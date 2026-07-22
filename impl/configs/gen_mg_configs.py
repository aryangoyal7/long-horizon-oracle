"""Generate Diffusion Policy configs for the Stage 2 MimicGen long-horizon tasks.

Same recipe as gen_dp_configs.py (robomimic diffusion_policy.json template, only
dataset/obs/output overridden) with two Stage 2 differences:
  - in-training rollout evaluation is DISABLED: mimicgen envs need robosuite 1.4,
    which only the mg venv has; the lh venv that owns diffusion policy is on 1.5.
    Evaluation happens post hoc through the mg venv.
  - no train/valid filter masks: the mimicgen hdf5s ship without mask groups.

epoch_every_n_steps is 100 in the template, so num_epochs 2000 is the same
200k-gradient-step budget as the Panda ph trainings regardless of dataset size.

Usage: python gen_mg_configs.py  (writes impl/configs/dp_mg_<task>.json)
"""

import json
import os

TEMPLATE = "/mnt/scratch/lh/repos/robomimic/robomimic/exps/templates/diffusion_policy.json"
DATA_ROOT = "/mnt/scratch/lh/data/mimicgen"
OUT_ROOT = ("/mnt/batch/tasks/shared/LS_root/mounts/clusters/garyan181/code/"
            "Users/garyan18/long-horizon/results/training")
CFG_DIR = os.path.dirname(os.path.abspath(__file__))

# all mimicgen core tasks render agentview + robot0_eye_in_hand at 84 px
TASKS = ["three_piece_assembly_d0", "nut_assembly_d0", "kitchen_d0",
         "coffee_preparation_d0"]


def main():
    with open(TEMPLATE) as f:
        base = json.load(f)

    for task in TASKS:
        cfg = json.loads(json.dumps(base))  # deep copy

        cfg["experiment"]["name"] = f"dp_mg_{task}_image"
        cfg["experiment"]["validate"] = False
        cfg["experiment"]["save"]["enabled"] = True
        cfg["experiment"]["save"]["every_n_epochs"] = 50
        cfg["experiment"]["save"]["on_best_rollout_success_rate"] = False
        cfg["experiment"]["rollout"]["enabled"] = False
        cfg["experiment"]["render_video"] = False

        cfg["train"]["data"] = f"{DATA_ROOT}/{task}_image.hdf5"
        cfg["train"]["output_dir"] = f"{OUT_ROOT}/dp_mg_{task}"
        cfg["train"]["num_data_workers"] = 0
        cfg["train"]["hdf5_cache_mode"] = "all"
        cfg["train"]["hdf5_filter_key"] = None
        cfg["train"]["hdf5_validation_filter_key"] = None

        mods = cfg["observation"]["modalities"]["obs"]
        mods["low_dim"] = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        mods["rgb"] = ["agentview_image", "robot0_eye_in_hand_image"]

        enc = cfg["observation"]["encoder"]["rgb"]
        enc["obs_randomizer_class"] = "CropRandomizer"
        enc["obs_randomizer_kwargs"] = {
            "crop_height": 76, "crop_width": 76, "num_crops": 1, "pos_enc": False,
        }

        out = os.path.join(CFG_DIR, f"dp_mg_{task}.json")
        with open(out, "w") as f:
            json.dump(cfg, f, indent=2)
        print("wrote", out,
              f"(epochs={cfg['train']['num_epochs']}, rollout=disabled)")


if __name__ == "__main__":
    main()
