import argparse
import math
import numpy as np
import pandas as pd

"""
Parse an Nsight Compute CSV export and generate a text summary.

This script aggregates per-kernel profiler rows into step-level totals and
derived metrics that are useful for roofline analysis, including:
- total floating-point operations
- total DRAM traffic
- achieved GFLOP/s
- achieved GB/s
- arithmetic intensity
- time-weighted SM and DRAM utilization

Typical usage:
    ncu --import report.ncu-rep --csv > report.csv
    python3 roofline_analysis/parse_ncu_csv.py report.csv \
        --label resnet18_bs128_L4 \
        --out summary.txt
"""

METRIC_SM   = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_TIME = "gpu__time_duration.sum"

METRIC_FADD = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
METRIC_FMUL = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
METRIC_FFMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"

METRIC_DRAM_READ  = "dram__bytes_read.sum"
METRIC_DRAM_WRITE = "dram__bytes_write.sum"

def to_float(v):
    """
    Normalize a profiler-provided metric string and convert it to float.

    Nsight Compute may format values with percent signs or thousands
    separators, so this helper strips those decorations first.
    """
    try:
        s = str(v).strip().replace("%", "").replace(",", "")
        return float(s)
    except Exception:
        return np.nan

def detect_time_scale(unit_raw: str) -> float:
    """
    Map time unit to seconds scaling factor.
    """
    u = (unit_raw or "").strip().lower()

    if u in {"usecond", "us", "µs", "microsecond", "microseconds"}:
        return 1e-6
    if u in {"ms", "millisecond", "milliseconds"}:
        return 1e-3
    if u in {"ns", "nanosecond", "nanoseconds"}:
        return 1e-9
    if u in {"s", "sec", "second", "seconds"}:
        return 1.0

    return 1e-6

def detect_bytes_scale(unit_raw: str) -> float:
    """
    Map byte units (byte, Kbyte, Mbyte, etc.) to bytes.
    """
    u = (unit_raw or "").strip().lower()

    if u in {"byte", "bytes"}:
        return 1.0
    if u in {"kbyte", "kbytes", "kb"}:
        return 1024.0
    if u in {"mbyte", "mbytes", "mb"}:
        return 1024.0 ** 2
    if u in {"gbyte", "gbytes", "gb"}:
        return 1024.0 ** 3

    return 1.0

def build_join_keys(df: pd.DataFrame):
    """
    Columns that identify a kernel instance.
    """
    candidates = ["Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC"]
    
    return [c for c in candidates if c in df.columns]

def weighted_avg_pct(df: pd.DataFrame, metric_name: str, join_keys):
    """
    Time-weighted average of a percentage metric over kernels.

    Each kernel contributes in proportion to how long it ran.
    This is more representative than a plain average because short-lived
    kernels should not influence the final utilization as much as kernels
    that dominate the profiled step.
    """

    # Metric values
    sub = df[df["Metric Name"] == metric_name][join_keys + ["Metric Value"]].copy()
    
    if sub.empty:
        return math.nan
    
    sub = sub.rename(columns={"Metric Value": "metric"})
    sub["metric"] = sub["metric"].apply(to_float)
    sub = sub.dropna(subset=["metric"])

    # Time values
    tdf = df[df["Metric Name"] == METRIC_TIME][join_keys + ["Metric Value"]].copy()
    
    if tdf.empty:
        return math.nan
    
    tdf = tdf.rename(columns={"Metric Value": "time_val"})
    tdf["time_val"] = tdf["time_val"].apply(to_float)
    tdf = tdf.dropna(subset=["time_val"])

    merged = pd.merge(sub, tdf, on=join_keys, how="inner")
    merged = merged[(merged["time_val"] > 0) & merged["metric"].notna()]

    if merged.empty:
        return math.nan

    return float((merged["metric"] * merged["time_val"]).sum() / merged["time_val"].sum())

def sum_step_time_seconds(df: pd.DataFrame) -> float:
    """
    Sum gpu__time_duration.sum over kernels, converted to seconds.
    """
    tdf = df[df["Metric Name"] == METRIC_TIME].copy()
    
    if tdf.empty:
        return math.nan

    unit_col = "Metric Unit" if "Metric Unit" in tdf.columns else (
        "Unit" if "Unit" in tdf.columns else None
    )
    
    if unit_col:
        unit_values = tdf[unit_col].dropna()
        unit_mode = unit_values.mode().iat[0] if not unit_values.empty else "usecond"
    else:
        unit_mode = "usecond"

    scale = detect_time_scale(unit_mode)

    tdf = tdf.rename(columns={"Metric Value": "time_val"})
    tdf["time_val"] = tdf["time_val"].apply(to_float)
    tdf = tdf.dropna(subset=["time_val"])

    step_time_raw = float(tdf["time_val"].sum())
    step_time_s = step_time_raw * scale
    
    return step_time_s

def sum_metric(df: pd.DataFrame, metric_name: str) -> float:
    """
    Sum Metric Value over all rows for a given metric (no unit scaling).
    """
    sub = df[df["Metric Name"] == metric_name].copy()
    
    if sub.empty:
        return 0.0
    
    vals = sub["Metric Value"].apply(to_float)
    vals = vals.dropna()
    
    return float(vals.sum())

def sum_bytes_metric(df: pd.DataFrame, metric_name: str) -> float:
    """
    Sum byte-type metrics, respecting Metric Unit, into raw bytes.
    """
    sub = df[df["Metric Name"] == metric_name].copy()
    
    if sub.empty:
        return 0.0

    total = 0.0
    unit_col = "Metric Unit" if "Metric Unit" in sub.columns else (
        "Unit" if "Unit" in sub.columns else None
    )
    
    for _, row in sub.iterrows():
        val = to_float(row["Metric Value"])
        
        if np.isnan(val):
            continue
        
        unit_raw = row.get(unit_col, "byte") if unit_col else "byte"
        scale = detect_bytes_scale(unit_raw)
        total += val * scale
    
    return float(total)

def main():
    ap = argparse.ArgumentParser(description="Parse Nsight Compute CSV and aggregate FLOPs/bytes/time.")

    ap.add_argument("csv", help="Nsight Compute --csv export (per-kernel rows)")
    ap.add_argument("--label", default="workload", help="Label/name for this run")
    ap.add_argument("--out", default="summary.txt", help="Output text summary file")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    join_keys = build_join_keys(df)

    if not join_keys:
        raise SystemExit(
            "Could not find kernel identity columns "
            "(Kernel Name/Context/Stream/Block/Grid/Device/CC)."
        )

    # 1) Time-weighted SM% and DRAM%
    sm_pct = weighted_avg_pct(df, METRIC_SM, join_keys)
    dram_pct = weighted_avg_pct(df, METRIC_DRAM, join_keys)

    # 2) Total step time (seconds)
    step_time_s = sum_step_time_seconds(df)

    # 3) FLOP operation counts
    # One FFMA instruction performs both a multiply and an add, so it
    # contributes 2 floating-point operations.
    fadd_total = sum_metric(df, METRIC_FADD)
    fmul_total = sum_metric(df, METRIC_FMUL)
    ffma_total = sum_metric(df, METRIC_FFMA)
    flops_total = fadd_total + fmul_total + 2.0 * ffma_total  # ffma = 2 FLOPs

    # 4) DRAM bytes (converted to raw bytes)
    bytes_read_total = sum_bytes_metric(df, METRIC_DRAM_READ)
    bytes_write_total = sum_bytes_metric(df, METRIC_DRAM_WRITE)
    bytes_total = bytes_read_total + bytes_write_total

    # 5) Rates and arithmetic intensity
    if step_time_s > 0:
        gflops_per_s = flops_total / step_time_s / 1e9
        gb_per_s = bytes_total / step_time_s / 1e9
    else:
        gflops_per_s = math.nan
        gb_per_s = math.nan

    if bytes_total > 0:
        AI_flop_per_byte = flops_total / bytes_total
    else:
        AI_flop_per_byte = math.nan

    # Write human-readable summary
    with open(args.out, "w") as f:
        f.write(f"Label: {args.label}\n")
        f.write(f"Input CSV: {args.csv}\n\n")

        f.write("=== Totals (aggregated across all kernels in the profiled step) ===\n")
        f.write(f"Total fadd instructions : {fadd_total:.3e}\n")
        f.write(f"Total fmul instructions : {fmul_total:.3e}\n")
        f.write(f"Total ffma instructions : {ffma_total:.3e}\n")
        f.write(f"Total FLOPs             : {flops_total:.3e}\n\n")

        f.write(f"Total DRAM bytes read   : {bytes_read_total:.3e}\n")
        f.write(f"Total DRAM bytes write  : {bytes_write_total:.3e}\n")
        f.write(f"Total DRAM bytes total  : {bytes_total:.3e}\n\n")

        f.write("=== Time and rates ===\n")
        f.write(f"Step time (seconds)     : {step_time_s:.6e}\n")
        f.write(f"Achieved GFLOP/s        : {gflops_per_s:.3f}\n")
        f.write(f"Achieved GB/s           : {gb_per_s:.3f}\n")
        f.write(f"Arithmetic intensity    : {AI_flop_per_byte:.6f} FLOP/byte\n\n")

        f.write("=== Utilization (time-weighted, elapsed) ===\n")
        f.write(f"SM% pct_of_peak_sustained_elapsed   : {sm_pct:.3f}\n")
        f.write(f"DRAM% pct_of_peak_sustained_elapsed : {dram_pct:.3f}\n")

    print(f"Wrote summary to {args.out}")
    print(f"Label: {args.label}")
    print(f"Step time (s): {step_time_s:.6e}")
    print(f"GFLOP/s:       {gflops_per_s:.3f}")
    print(f"GB/s:          {gb_per_s:.3f}")
    print(f"AI (FLOP/B):   {AI_flop_per_byte:.6f}")
    print(f"SM%:           {sm_pct:.3f}")
    print(f"DRAM%:         {dram_pct:.3f}")

if __name__ == "__main__":
    main()
