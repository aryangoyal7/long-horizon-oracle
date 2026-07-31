"""One captioned sanity rollout video per RoboCasa task.

Loads the policy once, then for each task runs a single episode at k=16 with
the fork's VideoRecordingWrapper, records the episode's language instruction
(annotation.human.task_description), and renames the video to
<task>__<success|fail>.mp4 with a prompts.json sidecar.
"""
import glob
import json
import os
import shutil

import numpy as np

CKPT = "/mnt/scratch/lh/data/robocasa/ckpt/gr00t_n1-5/multitask_learning/checkpoint-120000"
OUT = "/mnt/scratch/lh/rollouts/groot_sanity"
TASKS = ["OpenCabinet", "OpenDrawer", "PickPlaceCounterToCabinet", "TurnOnSinkFaucet"]
ACTION_KEYS = ["action.gripper_close", "action.end_effector_position",
               "action.end_effector_rotation", "action.base_motion",
               "action.control_mode"]
LANG_KEY = "annotation.human.task_description"


def policy_obs(obs):
    """The policy's batching transform indexes every entry; plain strings and
    0-dim arrays (the env's language annotation) must become 1-element lists,
    matching the offline dataset format."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            out[k] = [v.item()]
        else:
            out[k] = v
    return out


def main():
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from gr00t.eval.simulation import (MultiStepConfig, SimulationConfig,
                                       VideoConfig, _create_single_env)
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    dc = DATA_CONFIG_MAP["panda_omron"]
    policy = Gr00tPolicy(
        model_path=CKPT, modality_config=dc.modality_config(),
        modality_transform=dc.transform(), embodiment_tag="new_embodiment",
        denoising_steps=4)

    prompts = {}
    for task in TASKS:
        vdir = os.path.join(OUT, f"_rec_{task}")
        os.makedirs(vdir, exist_ok=True)
        horizon = get_task_horizon(task)
        cfg = SimulationConfig(
            env_name=f"robocasa/{task}", split="pretrain",
            n_episodes=1, n_envs=1,
            video=VideoConfig(video_dir=vdir, steps_per_render=1),
            multistep=MultiStepConfig(n_action_steps=1,
                                      max_episode_steps=horizon))
        env = _create_single_env(cfg, 0)
        success, steps, lang = False, 0, ""
        for attempt in range(3):
            obs, _ = env.reset()
            lang = obs.get(LANG_KEY, "")
            if not isinstance(lang, str):
                lang = str(np.ravel(lang)[0]) if np.size(lang) else ""
            print(f"{task} attempt {attempt}: prompt = {lang!r}", flush=True)
            success, steps = False, 0
            chunk, executed, chunk_len = None, 0, 0
            for t in range(horizon):
                if chunk is None or executed >= chunk_len:
                    chunk = policy.get_action(policy_obs(obs))
                    chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
                    executed = 0
                act = {k: np.asarray(chunk[k])[executed: executed + 1]
                       for k in ACTION_KEYS}
                obs, _r, term, trunc, info = env.step(act)
                executed += 1
                steps = t + 1
                if bool(np.ravel(info.get("success", False))[0]):
                    success = True
                    break
                if term or trunc:
                    break
            print(f"{task} attempt {attempt}: "
                  f"{'success' if success else 'fail'} at {steps}", flush=True)
            if success:
                break
        env.reset()   # finalizes the last video file
        env.close()
        vids = sorted(glob.glob(os.path.join(vdir, "**", "*.mp4"),
                                recursive=True), key=os.path.getmtime)
        tag = "success" if success else "fail"
        dst = os.path.join(OUT, f"{task}__{tag}.mp4")
        if vids:
            shutil.copy(vids[-1], dst)
        prompts[task] = {"prompt": lang, "success": success, "steps": steps,
                         "video": dst}
        print(f"{task}: {tag} at {steps} steps -> {dst}", flush=True)

    with open(os.path.join(OUT, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)
    print("SANITY_VIDEOS_DONE", flush=True)


if __name__ == "__main__":
    main()
