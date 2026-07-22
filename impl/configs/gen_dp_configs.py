"""Generate robomimic Diffusion Policy image-training configs for the 4 ph tasks.

Starts from robomimic's own diffusion_policy.json template (the benchmark config:
To=2, Ta=8, Tp=16, UNet 256/512/1024, DDPM-100, AdamW 1e-4 cosine) and overrides ONLY:
  - dataset path / output dir / experiment name
  - observation keys (images + eef proprio) and crop-randomizer size per camera res
  - rollout horizon per task (robomimic benchmark values)
Checkpoints + logs go to the persistent CIFS share (survives instance restarts).

Usage: python gen_dp_configs.py  (writes impl/configs/dp_<task>.json)
"""

import json
import os

TEMPLATE = "/mnt/scratch/lh/repos/robomimic/robomimic/exps/templates/diffusion_policy.json"
DATA_ROOT = "/mnt/scratch/lh/data/robomimic"
OUT_ROOT = ("/mnt/batch/tasks/shared/LS_root/mounts/clusters/garyan181/code/"
            "Users/garyan18/long-horizon/results/training")
CFG_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = {
    #  task       main camera   px   rollout horizon
    "lift":      ("agentview", 84, 400),
    "can":       ("agentview", 84, 400),
    "square":    ("agentview", 84, 400),
    "tool_hang": ("sideview", 240, 700),
}


def main():
    with open(TEMPLATE) as f:
        base = json.load(f)

    for task, (cam, px, horizon) in TASKS.items():
        cfg = json.loads(json.dumps(base))  # deep copy

        cfg["experiment"]["name"] = f"dp_{task}_image"
        cfg["experiment"]["validate"] = True
        cfg["experiment"]["save"]["enabled"] = True
        cfg["experiment"]["save"]["every_n_epochs"] = 50
        cfg["experiment"]["save"]["on_best_rollout_success_rate"] = True
        cfg["experiment"]["rollout"]["horizon"] = horizon
        cfg["experiment"]["rollout"]["rate"] = 100   # eval every 100 epochs (20 rounds)
        cfg["experiment"]["render_video"] = False

        cfg["train"]["data"] = f"{DATA_ROOT}/{task}/ph/image_v15.hdf5"
        cfg["train"]["output_dir"] = f"{OUT_ROOT}/dp_{task}"
        cfg["train"]["num_data_workers"] = 0   # cache-all + fork workers => EGL segfault after first rollout
        # whole image dataset cached in RAM (36GB total across tasks, box has 1.7TB);
        # low_dim caching left epochs dataloader-bound at ~6 min vs ~20 s compute
        cfg["train"]["hdf5_cache_mode"] = "all"
        cfg["train"]["hdf5_filter_key"] = "train"           # official 90/10 split masks
        cfg["train"]["hdf5_validation_filter_key"] = "valid"

        mods = cfg["observation"]["modalities"]["obs"]
        mods["low_dim"] = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        mods["rgb"] = [f"{cam}_image", "robot0_eye_in_hand_image"]

        enc = cfg["observation"]["encoder"]["rgb"]
        enc["obs_randomizer_class"] = "CropRandomizer"
        crop = {84: 76, 240: 216}[px]      # robomimic image-benchmark crop sizes
        enc["obs_randomizer_kwargs"] = {
            "crop_height": crop, "crop_width": crop,
            "num_crops": 1, "pos_enc": False,
        }

        out = os.path.join(CFG_DIR, f"dp_{task}.json")
        with open(out, "w") as f:
            json.dump(cfg, f, indent=2)
        print("wrote", out, f"(cams={mods['rgb']}, crop={int(px*0.9)}, horizon={horizon},"
              f" epochs={cfg['train']['num_epochs']}, rollout_n={cfg['experiment']['rollout']['n']}"
              f" every {cfg['experiment']['rollout']['rate']} epochs)")


if __name__ == "__main__":
    main()
