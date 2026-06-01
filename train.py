import argparse, os, random, time, math
import torch
import numpy as _np
from contextlib import nullcontext
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# These are the known mean and std of the CIFAR-10 training set, used to normalize the dataset.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def set_random_seed(seed: int):
    random.seed(seed)                           # seeds Python's built-in RNG
    torch.manual_seed(seed)                     # seeds PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)            # seeds CUDA RNG
    
    # Deterministic=False + benchmark=True is a good perf default for CNNs w/ fixed shapes
    torch.backends.cudnn.deterministic = False  # allow cuDNN to use fast, non-deterministic kernels; fast but may not be reproducible
    torch.backends.cudnn.benchmark = True       # let cuDNN autotune the fastest kernel for your input shapes

def log_device_info(device: torch.device, amp_flag: bool):
    """
    Print summary of runtime environment and available accelerator

    NVIDIA GPU → prints CUDA info and whether AMP is enabled.
    Apple Silicon Mac → prints MPS info and notes AMP is disabled.
    Otherwise → CPU.

    MPS (Metal Performance Shader) : Apple's GPU backend
    AMP (Automatic Mixed Precision) : Use mixed precision to speed up math and reduce GPU memory
    """
    print(f"[env] torch={torch.__version__}")
    
    try:
        print(f"[env] numpy={_np.__version__}")
    except Exception:
        print("[env] numpy=NOT INSTALLED")

    if device.type == "cuda":
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        print(f"[device] CUDA available: True | gpus={n} | current='{name}'")
        print(f"[amp] enabled={amp_flag}")
    elif device.type == "mps":
        print("[device] MPS (Apple Silicon) available: True | using MPS device")
        print("[amp] disabled on MPS (float32 training)")
    else:
        print("[device] CPU")
        print("[amp] disabled on CPU")

def get_device():
    """
    Decides which accelerator to use, in order of preference
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")

    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = torch.device("mps")
    
    else:
        dev = torch.device("cpu")
    
    return dev

def make_loaders(data_dir, batch_size, workers, pin_memory: bool):
    """
    Builds training & test DataLoaders; how data is transformed, batched, and fed to the GPU/CPU.
    """
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    train = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=tf_train)
    test  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf_test)
    
    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=pin_memory,
        persistent_workers=workers > 0
    )
    
    test_dl = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_memory,
        persistent_workers=workers > 0
    )
    
    return train_dl, test_dl

def accuracy(logits, y):
    """
    Top-1 accuracy helper used for reporting performance.
    This is separate from the loss function that drives learning.

    logits has shape [batch, classes] and stores one score per class for each example.
    y has shape [batch] and stores the true class ID for each example.

    Returns the fraction of correct predictions in the batch.
    """
    return (logits.argmax(1) == y).float().mean().item()
 
def save_ckpt(path, model, opt, sched, epoch, best_acc):
    """
    Write a training checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "best_acc": best_acc,
    }, path)

def load_ckpt(path, model, opt=None, sched=None, map_location="cpu"):
    """
    Load a training checkpoint
    """
    blob = torch.load(path, map_location=map_location)
    model.load_state_dict(blob["model"])
    
    if opt and "opt" in blob and blob["opt"] is not None:
        opt.load_state_dict(blob["opt"])
    
    if sched and "sched" in blob and blob["sched"] is not None:
        sched.load_state_dict(blob["sched"])
    
    return blob.get("epoch", 0), blob.get("best_acc", 0.0)

# ---- NVTX helpers (no-ops when not on CUDA) ----
"""
    Use NVTX to profle one training step
    * Profile GPU work inside the NVTX window (matmul/conv, fused ops, optimizer kernels)
    * Attribution to sub-ranges (“forward”, “backward”, etc.)
    * Bounded to one batch thanks to warmup + syncs + early exit

    Using NVTX
    * nvtx push/pop labels “start/stop” markers on the host timeline.
    * nvtx.range_push("name") tells the profiler “start a region called name now.”
    * nvtx.range_pop() says “end the most recent region.”

    They’re annotations that Nsight tools read later to:
    * Align your labeled regions with actual CUDA kernel launches (which are async).
    * Let you filter and attribute metrics to specific phases (forward/backward/opt).
"""
def _nvtx_push(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)

def _nvtx_pop(enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_pop()

def _cuda_sync(enabled: bool):
    if enabled:
        torch.cuda.synchronize()

def main(args):
    set_random_seed(args.seed)
    device = get_device()
    use_cuda = (device.type == "cuda")    # are we on NVIDIA?
    amp_on = args.amp and use_cuda        # use AMP?
    pin_mem = use_cuda                    # pin_memory, only helps on CUDA
    profiling_enabled = bool(args.profile_one_step) and use_cuda
    log_device_info(device, amp_on)

    # DataLoaders
    train_dl, test_dl = make_loaders(args.data, args.batch_size, args.workers, pin_mem)

    # Model
    if args.model.lower() == "resnet18":
        model = models.resnet18(num_classes=10)
    elif args.model.lower() == "resnet34":
        model = models.resnet34(num_classes=10)
    else:
        raise ValueError(f"Unknown model '{args.model}'. Choose from ['resnet18', 'resnet34'].")
    model.to(device)

    # Optim, loss, sched
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Cosine schedule with warmup
    warmup = max(0, args.warmup)
    total_steps = args.epochs * math.ceil(len(train_dl.dataset) / args.batch_size)
    
    # Learning rate multiplier
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)     # learning rate scheduler, automatically changes the optimizer's lr during training
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)     # loss function

    # Scaler helps training stay numerically stable when using lower-precision math.
    # With AMP, some values can become very small and gradients can underflow.
    # The scaler temporarily multiplies the loss by a large factor before backprop.
    # This makes the gradients larger too, so they are less likely to underflow in lower precision.
    # Before the optimizer update, PyTorch divides those gradients back down to their original scale.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if use_cuda else None
    
    # Context manager that turns mixed precision on for the forward pass.
    # Outside the "with autocast" block, normal precision rules apply.
    # Inside the block, PyTorch can run some operations in lower precision
    # when it is supported and safe to do so.
    autocast_ctx = (lambda: torch.amp.autocast("cuda", enabled=amp_on)) if use_cuda else (lambda: nullcontext())

    start_epoch, best_acc = 0, 0.0

    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_acc = load_ckpt(args.resume, model, opt, sched, map_location="cpu")
        print(f"[resume] from {args.resume} at epoch {start_epoch}, best_acc={best_acc:.3f}")

    warmup_iters = max(0, args.warmup_iters)    # how many iterations to run before profiling starts.
    profile_iter = max(1, args.profile_iter)    # which single iteration after warmup to profile (1 = the first one after warmup).
    did_profile = False                         # a boolean flag indicating whether the script actually profiled that one step (used to exit early and skip eval).

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # -------- train --------
        model.train()
        total, loss_sum, acc_sum = 0, 0.0, 0.0

        it = 0
        for xb, yb in train_dl:
            it += 1
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            # Pick which iteration to profile: decide whether this iteration is the NVTX-profiled one
            profile_this_step = profiling_enabled and (it == warmup_iters + profile_iter)

            if profiling_enabled and it == 1:
                print(f"[profile] warmup iters={warmup_iters}, profiling iter={warmup_iters + profile_iter} (1-based within epoch)")
            
            if profile_this_step:
                _cuda_sync(True) # torch.cuda.synchronize() ensures your “end” truly captures all GPU work inside the range (CUDA is async by default).
                _nvtx_push("train_step", True)

            with autocast_ctx():
                if profile_this_step: _nvtx_push("forward", True)
                logits = model(xb)
                if profile_this_step: _nvtx_pop(True)

                if profile_this_step: _nvtx_push("loss", True)
                loss = loss_fn(logits, yb)
                if profile_this_step: _nvtx_pop(True)

            if scaler is not None:
                if profile_this_step: _nvtx_push("backward", True)
                scaler.scale(loss).backward()
                if profile_this_step: _nvtx_pop(True)

                if profile_this_step: _nvtx_push("optimizer_step", True)
                scaler.step(opt)
                scaler.update()
                if profile_this_step: _nvtx_pop(True)

            else:
                if profile_this_step: _nvtx_push("backward", True)
                loss.backward()
                if profile_this_step: _nvtx_pop(True)

                if profile_this_step: _nvtx_push("optimizer_step", True)
                opt.step()
                if profile_this_step: _nvtx_pop(True)

            sched.step()

            if profile_this_step:
                _cuda_sync(True)
                _nvtx_pop(True)
                did_profile = True
                print("[profile] captured one training step via NVTX; exiting early from training loop.")
                
                # Update metrics for the profiled batch before exiting
                bs = xb.size(0)
                total += bs
                loss_sum += loss.detach().item() * bs
                acc_sum += accuracy(logits.detach(), yb) * bs
                break  # early-exit the training loop

            # Normal metrics accumulation
            bs = xb.size(0)
            total += bs
            loss_sum += loss.detach().item() * bs
            acc_sum += accuracy(logits.detach(), yb) * bs

        # If we profiled a single step, skip eval and end after this epoch.
        if did_profile:
            tr_loss, tr_acc = loss_sum / max(1, total), acc_sum / max(1, total)
            dt = time.time() - t0
            print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | (profile run) | {dt:.1f}s")
            print("[profile] Done. Skipping eval and remaining epochs.")
            return

        # -------- eval --------
        model.eval()
        total_t, loss_t, acc_t = 0, 0.0, 0.0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast_ctx():
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                bs = xb.size(0)
                total_t += bs
                loss_t += loss.item() * bs
                acc_t += accuracy(logits, yb) * bs

        tr_loss, tr_acc = loss_sum / total, acc_sum / total
        te_loss, te_acc = loss_t / total_t, acc_t / total_t
        dt = time.time() - t0
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| test loss {te_loss:.4f} acc {te_acc:.3f} | {dt:.1f}s")

        # Save best
        if te_acc > best_acc:
            best_acc = te_acc
            save_ckpt(os.path.join(args.out_dir, f"{args.model}_cifar10_best.pt"),
                      model, opt, sched, epoch, best_acc)

        # Optional epoch snapshots
        if args.save_every and (epoch % args.save_every == 0):
            save_ckpt(os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"),
                      model, opt, sched, epoch, best_acc)

    print(f"best test acc: {best_acc:.3f}")
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"{args.model}_final_weights.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./artifacts")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--amp", action="store_true", help="use mixed precision on CUDA")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500, help="warmup steps for cosine schedule")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="", help="path to checkpoint")
    p.add_argument("--save-every", type=int, default=0, help="save snapshot every N epochs (0=off)")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34"], help="model architecture")
    p.add_argument("--profile-one-step", action="store_true", help="warm up then profile exactly one training step via NVTX and exit early (CUDA only)")
    p.add_argument("--warmup-iters", type=int, default=20, help="number of warmup iterations before the profiled step")
    p.add_argument("--profile-iter", type=int, default=1, help="which iteration after warmup to profile (1 = first after warmup)")
    args = p.parse_args()
    main(args)
