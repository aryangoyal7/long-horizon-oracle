"""Prepare LIBERO-10 for diffusion policy training.

Two jobs, both idempotent:
  1. add a mask/{train,valid} 90/10 split by demo (LIBERO ships no filter keys)
  2. write one robomimic diffusion-policy config per task

Differences from the panda configs, and why:
  - obs keys are LIBERO's own (agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori,
    gripper_states), not robomimic's (agentview_image, robot0_eef_pos, ...)
  - images are 128x128 rather than 84x84, so the crop randomizer is 116 (~90%)
  - rollout is DISABLED. robomimic cannot construct a LIBERO env reliably (the
    wrapper path is nondeterministic, see ftle_labeler_libero.py), so success
    rates come from a separate native-env harness and checkpoint selection here
    is by validation loss.
  - batch 128 rather than 100: LIBERO frames are 128x128, so this is already a
    larger memory footprint per step than any panda run, and on 13k transitions
    a much bigger batch would just cut the number of distinct gradient steps.
"""
import argparse, copy, json, os, glob
import h5py
import numpy as np

LOW_DIM = ["ee_pos", "ee_ori", "gripper_states"]
RGB = ["agentview_rgb", "eye_in_hand_rgb"]


def add_mask(path, val_frac=0.1, seed=0):
    with h5py.File(path, "a") as f:
        if "mask" in f and "train" in f["mask"] and "valid" in f["mask"]:
            return "skip"
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(len(demos) * val_frac)))
        val = set(rng.choice(len(demos), n_val, replace=False).tolist())
        tr = np.array([d.encode() for i, d in enumerate(demos) if i not in val])
        va = np.array([d.encode() for i, d in enumerate(demos) if i in val])
        if "mask" in f:
            del f["mask"]
        g = f.create_group("mask")
        g.create_dataset("train", data=tr)
        g.create_dataset("valid", data=va)
        return f"{len(tr)}/{len(va)}"


def make_config(template, dataset, task, out_dir, epochs, batch):
    c = copy.deepcopy(template)
    c["experiment"]["name"] = f"dp_libero_{task}"
    c["experiment"]["rollout"]["enabled"] = False
    c["experiment"]["save"]["on_best_rollout_success_rate"] = False
    c["experiment"]["save"]["on_best_validation"] = True
    c["experiment"]["save"]["every_n_epochs"] = 50
    c["experiment"]["validate"] = True
    c["train"]["data"] = dataset
    c["train"]["output_dir"] = os.path.join(out_dir, f"dp_libero_{task}")
    c["train"]["batch_size"] = batch
    c["train"]["num_epochs"] = epochs
    c["train"]["hdf5_filter_key"] = "train"
    c["train"]["hdf5_validation_filter_key"] = "valid"
    o = c["observation"]["modalities"]["obs"]
    o["low_dim"] = list(LOW_DIM)
    o["rgb"] = list(RGB)
    enc = c["observation"]["encoder"]["rgb"]["obs_randomizer_kwargs"]
    enc["crop_height"] = 116
    enc["crop_width"] = 116
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/mnt/scratch/lh/data/libero/libero_10")
    ap.add_argument("--template", required=True)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    template = json.load(open(args.template))
    os.makedirs(args.config_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.hdf5")))
    print(f"found {len(files)} datasets")
    for p in files:
        task = os.path.basename(p).replace("_demo.hdf5", "")
        st = add_mask(p)
        cfg = make_config(template, p, task, args.output_dir, args.epochs, args.batch)
        cp = os.path.join(args.config_dir, f"dp_libero_{task}.json")
        with open(cp, "w") as f:
            json.dump(cfg, f, indent=1)
        with h5py.File(p, "r") as f:
            n = len(f["data"]); tot = int(f["data"].attrs["total"])
        print(f"  {task[:48]:50s} demos={n} steps={tot} mask={st}")
    print("PREP_DONE")


if __name__ == "__main__":
    main()
