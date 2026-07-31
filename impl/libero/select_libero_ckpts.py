"""Pick a checkpoint per LIBERO task by validation loss.

Rollouts are disabled during LIBERO training (robomimic cannot construct a
LIBERO env reliably), so there is no best-rollout-success-rate checkpoint to
read. This parses the training log for per-epoch validation loss, reports the
best epoch, and maps it to the nearest checkpoint actually on disk. The chosen
checkpoints are what the native-env rollout harness then evaluates.
"""
import argparse, glob, json, os, re

VAL_RE = re.compile(r"Validation Epoch (\d+)")
LOSS_RE = re.compile(r"Loss\s*:\s*([0-9.eE+-]+)")


def parse(log_path):
    """Return {epoch: val_loss} from a robomimic launch log."""
    out, cur = {}, None
    if not os.path.exists(log_path):
        return out
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                cur = int(m.group(1)); continue
            if cur is not None:
                m2 = LOSS_RE.search(line)
                if m2:
                    try:
                        out[cur] = float(m2.group(1))
                    except ValueError:
                        pass
                    cur = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for d in sorted(glob.glob(os.path.join(args.training_dir, "dp_libero_*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        ck = sorted(glob.glob(os.path.join(d, "**", "models", "*.pth"), recursive=True))
        epochs = {}
        for c in ck:
            m = re.search(r"epoch_(\d+)", os.path.basename(c))
            if m:
                epochs[int(m.group(1))] = c
        val = parse(os.path.join(args.training_dir, f"{name}.launch.log"))
        best_ep = min(val, key=val.get) if val else None
        chosen = None
        if epochs:
            key = (min(epochs, key=lambda e: abs(e - best_ep)) if best_ep is not None
                   else max(epochs))
            chosen = epochs[key]
        res[name] = {
            "n_checkpoints": len(epochs),
            "checkpoint_epochs": sorted(epochs),
            "best_val_epoch": best_ep,
            "best_val_loss": val.get(best_ep) if best_ep is not None else None,
            "chosen_checkpoint": chosen,
            "selection": "nearest saved checkpoint to best validation epoch",
        }
        print(f"{name[:52]:54s} ckpts={len(epochs):3d} best_val_ep={best_ep} "
              f"-> {os.path.basename(chosen) if chosen else 'NONE'}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"CKPT_SELECTION_WRITTEN {args.out} ({len(res)} tasks)")


if __name__ == "__main__":
    main()
