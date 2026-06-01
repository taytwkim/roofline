import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

"""
Plot a single-point roofline chart from precomputed summary values.

The workload point comes from profiler-derived values such as arithmetic
intensity and achieved GFLOP/s. The hardware peak values passed to this
script (peak compute throughput and peak memory bandwidth) are known GPU
specifications and are not extracted from the profiler output.

Typical usage:
    python3 roofline_analysis/plot_roofline.py \
        --ai 15.624057 \
        --gflops 1466.556 \
        --peak-compute 30.0 \
        --peak-bw 300.0 \
        --label resnet18_bs128_L4 \
        --out resnet18_bs128_L4.png
"""

def logspace(a, b, n):
    """
    Log-spaced samples between a and b (inclusive) with n points.
    """
    return [10 ** (math.log10(a) + i * (math.log10(b / a) / (n - 1))) for i in range(n)]

def main():
    ap = argparse.ArgumentParser(description="Plot a single-point roofline given precomputed AI and GFLOP/s.")
    ap.add_argument("--ai", type=float, required=True, help="Arithmetic intensity (FLOP/byte) for the workload point.")
    ap.add_argument("--gflops", type=float, required=True, help="Attained performance (GFLOP/s) for the workload point.")
    ap.add_argument("--peak-compute", type=float, required=True, help="Peak compute throughput (TFLOP/s).")
    ap.add_argument("--peak-bw", type=float, required=True, help="Peak memory bandwidth (GB/s).")
    ap.add_argument("--label", default="Workload", help="Label for the plotted point.")
    ap.add_argument("--sm-pct", type=float, default=None, help="Time-weighted SM%% of peak sustained (optional, for annotation).")
    ap.add_argument("--dram-pct", type=float, default=None, help="Time-weighted DRAM%% of peak sustained (optional, for annotation).")
    ap.add_argument("--out", default="roofline.png", help="Output PNG filename.")
    args = ap.parse_args()

    # Inputs
    AI = args.ai                                        # FLOP/byte
    Y = args.gflops                                     # GFLOP/s
    peak_compute_gflops = args.peak_compute * 1000.0    # TFLOP/s -> GFLOP/s
    peak_bw_gbps = args.peak_bw                         # GB/s

    # Knee: where compute roof meets bandwidth roof
    knee_AI = peak_compute_gflops / peak_bw_gbps

    # Choose an x-range that keeps both the workload point and the knee visible.
    # This avoids zooming in too tightly around just one of them.
    x_min = max(1e-3, min(AI / 5.0, knee_AI / 20.0))
    x_max = max(AI * 5.0, knee_AI * 2.0)
    xs = logspace(x_min, x_max, 300)

    # The roofline is bandwidth-limited on the sloped segment (x * peak_bw)
    # and compute-limited once it hits the flat compute ceiling.
    ys = [min(peak_compute_gflops, x * peak_bw_gbps) for x in xs]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Theoretical roof
    ax.plot(xs, ys, linewidth=2, label="Theoretical roof")

    # Compute peak (horizontal line)
    ax.hlines(
        peak_compute_gflops,
        x_min,
        x_max,
        linestyles="--",
        linewidth=1,
        label="Compute peak",
    )

    # Knee (vertical line)
    ax.vlines(
        knee_AI,
        max(ys[0] * 0.5, 1.0),
        peak_compute_gflops,
        linestyles=":",
        linewidth=1,
        label="Knee",
    )

    # Workload point
    ax.scatter([AI], [Y], s=40)

    # Annotate the measured workload point with the summary values that
    # produced it, plus optional utilization metrics from the profiler.
    annot_lines = [
        f"{args.label}",
        f"AI={AI:.3f} FLOP/B",
        f"Y={Y:.1f} GFLOP/s",
    ]

    if args.sm_pct is not None:
        annot_lines.append(f"SM%={args.sm_pct:.1f}")
    
    if args.dram_pct is not None:
        annot_lines.append(f"DRAM%={args.dram_pct:.1f}")

    ax.annotate(
        "\n".join(annot_lines),
        (AI, Y),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=8,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(max(1.0, ys[0] * 0.5), peak_compute_gflops * 1.5)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Attainable Performance (GFLOP/s)")
    ax.set_title(f"Roofline (Peak: {args.peak_compute:.2f} TFLOP/s, {args.peak_bw:.0f} GB/s)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)
    
    print(f"Saved roofline plot to {args.out}")
    print(f"Point: AI={AI:.3f} FLOP/byte, Y={Y:.1f} GFLOP/s, knee AI={knee_AI:.3f}")

if __name__ == "__main__":
    main()
