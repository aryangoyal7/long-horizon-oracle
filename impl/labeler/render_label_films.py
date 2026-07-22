"""Stitch a long per-task film of labeled demonstrations.

Each frame shows the replayed demo (offscreen EGL render), the current
open-loop and closed-loop label class with its lambda value, the length of
the current open-loop band, and two full-episode color timelines (OL, CL)
with a moving cursor. Demos play in index order, so demos 0-49 (the ones
with closed-loop labels) come first. Run in the lh venv.

Colors match the label report: blue = stable (lam < -delta),
light gray = deadband, red = unstable (lam > delta), white = no label.
"""
import argparse, json, os
import h5py
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

C_STABLE = (33, 102, 172)
C_DEAD = (224, 224, 224)
C_UNSTABLE = (178, 24, 43)
C_NONE = (255, 255, 255)
C_BG = (24, 24, 28)
C_TEXT = (235, 235, 235)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
F_BIG = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 22)
F_MED = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 16)
F_SMALL = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans.ttf", 13)

W, H = 660, 470
VID = 320          # rendered frame is upscaled to VID x VID
PANEL_X = VID + 20
STRIP_H = 24
STRIP_W = W - 130


def classify(lam, delta):
    if lam is None or np.isnan(lam):
        return None
    if lam < -delta:
        return "stable"
    if lam > delta:
        return "unstable"
    return "deadband"


def color_of(cls):
    return {None: C_NONE, "stable": C_STABLE, "deadband": C_DEAD,
            "unstable": C_UNSTABLE}[cls]


def per_step(stamp_t, stamp_lam, stride, T):
    """Expand stamps (each covering [t, t+stride)) to a per-step lambda
    array of length T, NaN where no stamp covers the step."""
    lam = np.full(T, np.nan)
    for t, v in zip(stamp_t, stamp_lam):
        lam[t:min(t + stride, T)] = v
    return lam


def runs(classes):
    """List of (cls, start, end_exclusive) contiguous runs."""
    out = []
    s = 0
    for i in range(1, len(classes) + 1):
        if i == len(classes) or classes[i] != classes[s]:
            out.append((classes[s], s, i))
            s = i
    return out


def draw_strip(draw, x, y, classes, T, cur_t, label):
    draw.text((x - 118, y + 4), label, font=F_SMALL, fill=C_TEXT)
    for cls, s, e in runs(classes):
        x0 = x + int(s / T * STRIP_W)
        x1 = x + max(int(e / T * STRIP_W), x0 + 1)
        draw.rectangle([x0, y, x1, y + STRIP_H], fill=color_of(cls))
    draw.rectangle([x, y, x + STRIP_W, y + STRIP_H], outline=(90, 90, 90))
    cx = x + int(cur_t / T * STRIP_W)
    draw.line([cx, y - 3, cx, y + STRIP_H + 3], fill=(255, 255, 0), width=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ol", required=True)
    ap.add_argument("--cl", required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--target-frames", type=int, default=14400)  # 12 min at 20 fps
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {"low_dim": ["robot0_eef_pos", "robot0_eef_quat",
                     "robot0_gripper_qpos", "object"], "rgb": []})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True)

    ol = np.load(args.ol)
    cl = np.load(args.cl)
    f = h5py.File(args.dataset, "r")
    writer = imageio.get_writer(args.out, fps=20, quality=7,
                                macro_block_size=None)
    total = 0
    demo = 0
    stats = {"demos": 0, "frames": 0}
    while total < args.target_frames and f"demo_{demo}" in f["data"]:
        states = f[f"data/demo_{demo}/states"][()]
        T = len(states)
        m_ol = ol["demo_id"] == demo
        m_cl = cl["demo_id"] == demo
        if not m_ol.any():
            break  # past the labeled range
        lam_ol = per_step(ol["t"][m_ol], ol["lambda_task"][m_ol], 2, T)
        lam_cl = (per_step(cl["t"][m_cl], cl["lambda_cl_task"][m_cl], 10, T)
                  if m_cl.any() else np.full(T, np.nan))
        cls_ol = [classify(v, args.delta) for v in lam_ol]
        cls_cl = [classify(v, args.delta) for v in lam_cl]
        run_of_step = {}
        for cls, s, e in runs(cls_ol):
            for t in range(s, e):
                run_of_step[t] = (cls, s, e)

        # 0.4 s separator card
        card = Image.new("RGB", (W, H), C_BG)
        d = ImageDraw.Draw(card)
        d.text((W // 2 - 90, H // 2 - 20), f"{args.task}  demo {demo}",
               font=F_BIG, fill=C_TEXT)
        for _ in range(8):
            writer.append_data(np.asarray(card))

        for t in range(T):
            env.reset_to({"states": states[t]})
            fr = env.render(mode="rgb_array", height=256, width=256,
                            camera_name=args.camera)
            img = Image.new("RGB", (W, H), C_BG)
            img.paste(Image.fromarray(fr).resize((VID, VID), Image.NEAREST),
                      (10, 10))
            d = ImageDraw.Draw(img)
            d.text((PANEL_X, 12), f"{args.task}", font=F_BIG, fill=C_TEXT)
            d.text((PANEL_X, 42), f"demo {demo}   step {t + 1}/{T}",
                   font=F_MED, fill=C_TEXT)

            y = 78
            for name, lam, cls in [("open-loop", lam_ol[t], cls_ol[t]),
                                   ("closed-loop", lam_cl[t], cls_cl[t])]:
                d.text((PANEL_X, y), name, font=F_MED, fill=C_TEXT)
                d.rectangle([PANEL_X, y + 22, PANEL_X + 26, y + 48],
                            fill=color_of(cls), outline=(90, 90, 90))
                txt = ("no label" if cls is None
                       else f"{cls}   λ = {lam:+.3f}")
                d.text((PANEL_X + 36, y + 26), txt, font=F_MED, fill=C_TEXT)
                y += 62
            cls, s, e = run_of_step[t]
            d.text((PANEL_X, y),
                   f"current OL band: {e - s} steps ({s}..{e - 1})",
                   font=F_SMALL, fill=C_TEXT)
            d.text((PANEL_X, y + 22),
                   f"δ = {args.delta:.4f} (same for both)",
                   font=F_SMALL, fill=C_TEXT)
            y += 52
            for cname, cls2 in [("stable  λ < -δ", "stable"),
                                ("deadband  |λ| ≤ δ", "deadband"),
                                ("unstable  λ > δ", "unstable")]:
                d.rectangle([PANEL_X, y, PANEL_X + 16, y + 16],
                            fill=color_of(cls2), outline=(90, 90, 90))
                d.text((PANEL_X + 24, y + 1), cname, font=F_SMALL, fill=C_TEXT)
                y += 22

            draw_strip(d, 128, H - 96, cls_ol, T, t, "open-loop")
            draw_strip(d, 128, H - 52, cls_cl, T, t, "closed-loop")
            writer.append_data(np.asarray(img))
            total += 1
        stats["demos"] += 1
        stats["frames"] = total
        demo += 1
        if demo % 10 == 0:
            print(f"[{args.task}] demo {demo}, {total} frames", flush=True)
    writer.close()
    print(f"RENDER_{args.task}_DONE demos={stats['demos']} "
          f"frames={total} minutes={total / 20 / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
