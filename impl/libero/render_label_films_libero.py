"""Stitch a long per-task film of labeled LIBERO demonstrations.

Same idea as impl/labeler/render_label_films.py, but LIBERO needs its own
frame source: the robomimic wrapper path is nondeterministic for LIBERO (see
ftle_labeler_libero.py), so frames come from stepping the native
OffScreenRenderEnv from the recorded init state rather than from reset_to on
each stored state.

The other difference is the labels. LIBERO has open-loop labels only, no
closed-loop pass, so the second lambda row of the robomimic film is replaced
by the window contact flags, which the LIBERO label files do carry.

Colors match the label report: blue = stable (lam < -delta),
light gray = deadband, red = unstable (lam > delta), white = no label.
Run in the mg venv (robosuite 1.4.1), which is the one LIBERO imports against.
"""
import argparse
import os
import sys

import h5py
import imageio
import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "labeler"))
from render_label_films import (  # noqa: E402
    C_BG, C_TEXT, C_DEAD, F_BIG, F_MED, F_SMALL, H, PANEL_X, STRIP_H, STRIP_W,
    VID, W, classify, color_of, runs)

sys.path.insert(0, _HERE)
from ftle_labeler_libero import resolve_bddl  # noqa: E402

C_ON = (60, 160, 90)
C_OFF = (60, 60, 66)


def per_step(stamp_t, stamp_vals, T):
    """Forward-fill stamps to a per-step array, NaN before the first stamp.
    Stride-agnostic, unlike the fixed-stride expansion the robomimic film uses."""
    out = np.full(T, np.nan)
    order = np.argsort(stamp_t)
    ts, vs = np.asarray(stamp_t)[order], np.asarray(stamp_vals, float)[order]
    idx = np.searchsorted(ts, np.arange(T), side="right") - 1
    ok = idx >= 0
    out[ok] = vs[idx[ok]]
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


def draw_flag_strip(draw, x, y, flags, T, cur_t, label):
    """Same geometry as draw_strip but for a boolean per-step series."""
    draw.text((x - 118, y + 4), label, font=F_SMALL, fill=C_TEXT)
    vals = ["on" if v else "off" for v in flags]
    for v, s, e in runs(vals):
        x0 = x + int(s / T * STRIP_W)
        x1 = x + max(int(e / T * STRIP_W), x0 + 1)
        draw.rectangle([x0, y, x1, y + STRIP_H], fill=C_ON if v == "on" else C_OFF)
    draw.rectangle([x, y, x + STRIP_W, y + STRIP_H], outline=(90, 90, 90))
    cx = x + int(cur_t / T * STRIP_W)
    draw.line([cx, y - 3, cx, y + STRIP_H + 3], fill=(255, 255, 0), width=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="short name shown on the film")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--target-frames", type=int, default=14400)  # 12 min at 20 fps
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-demos", type=int, default=0, help="0 = no cap")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # match the dataset's frame orientation before the env is constructed
    with h5py.File(args.dataset, "r") as f:
        conv = f["data"].attrs.get("macros_image_convention", b"opencv")
        conv = conv.decode() if isinstance(conv, bytes) else conv
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = conv

    from libero.libero.envs import OffScreenRenderEnv

    z = np.load(args.labels, allow_pickle=True)
    lab_did, lab_t, lab_lam = z["demo_id"], z["t"], z["lambda_task"]
    wg = z.get("win_contact_gripper")
    wt = z.get("win_contact_table")
    wf = z.get("win_contact_fixture")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, quality=7,
                                macro_block_size=None)
    total = 0
    n_demos = 0
    with h5py.File(args.dataset, "r") as f:
        bddl = resolve_bddl(f["data"].attrs["bddl_file_name"])
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.size,
                                 camera_widths=args.size, ignore_done=True)
        env.reset()
        demos = sorted(set(lab_did.tolist()))
        if args.max_demos:
            demos = demos[: args.max_demos]
        for did in demos:
            if total >= args.target_frames:
                break
            g = f[f"data/demo_{did}"]
            states, actions = g["states"][()], g["actions"][()]
            T = int(actions.shape[0])
            sel = lab_did == did
            if not sel.any():
                continue
            ts_d = lab_t[sel]
            lam = per_step(ts_d, lab_lam[sel], T)
            cls_ol = [classify(v, args.delta) for v in lam]
            fg = per_step(ts_d, wg[sel], T) > 0.5 if wg is not None else np.zeros(T, bool)
            ft = per_step(ts_d, wt[sel], T) > 0.5 if wt is not None else np.zeros(T, bool)
            ff = per_step(ts_d, wf[sel], T) > 0.5 if wf is not None else np.zeros(T, bool)
            run_of_step = {}
            for cls, s, e in runs(cls_ol):
                for t in range(s, e):
                    run_of_step[t] = (cls, s, e)

            # 0.4 s separator card
            card = Image.new("RGB", (W, H), C_BG)
            d = ImageDraw.Draw(card)
            d.text((W // 2 - 150, H // 2 - 20), f"{args.task}  demo {did}",
                   font=F_BIG, fill=C_TEXT)
            for _ in range(8):
                writer.append_data(np.asarray(card))

            env.set_init_state(states[0])
            for t in range(T):
                obs, _, _, _ = env.step(actions[t])
                fr = obs.get("agentview_image")
                if fr is None:
                    key = [k for k in obs if k.endswith("_image")][0]
                    fr = obs[key]
                fr = np.ascontiguousarray(fr)
                img = Image.new("RGB", (W, H), C_BG)
                img.paste(Image.fromarray(fr).resize((VID, VID), Image.NEAREST),
                          (10, 10))
                d = ImageDraw.Draw(img)
                d.text((PANEL_X, 12), args.task, font=F_MED, fill=C_TEXT)
                d.text((PANEL_X, 40), f"demo {did}   step {t + 1}/{T}",
                       font=F_MED, fill=C_TEXT)

                y = 74
                d.text((PANEL_X, y), "open-loop", font=F_MED, fill=C_TEXT)
                d.rectangle([PANEL_X, y + 22, PANEL_X + 26, y + 48],
                            fill=color_of(cls_ol[t]), outline=(90, 90, 90))
                txt = ("no label" if cls_ol[t] is None
                       else f"{cls_ol[t]}   λ = {lam[t]:+.3f}")
                d.text((PANEL_X + 36, y + 26), txt, font=F_MED, fill=C_TEXT)
                y += 62

                cls, s, e = run_of_step[t]
                d.text((PANEL_X, y), f"current OL band: {e - s} steps ({s}..{e - 1})",
                       font=F_SMALL, fill=C_TEXT)
                d.text((PANEL_X, y + 20), f"δ = {args.delta:.4f}",
                       font=F_SMALL, fill=C_TEXT)
                y += 46

                d.text((PANEL_X, y), "window contact", font=F_SMALL, fill=C_TEXT)
                y += 18
                for cname, on in [("gripper", fg[t]), ("table", ft[t]),
                                  ("fixture", ff[t])]:
                    d.rectangle([PANEL_X, y, PANEL_X + 16, y + 14],
                                fill=C_ON if on else C_OFF, outline=(90, 90, 90))
                    d.text((PANEL_X + 24, y), cname, font=F_SMALL, fill=C_TEXT)
                    y += 19
                y += 6
                for cname, cls2 in [("stable  λ < -δ", "stable"),
                                    ("deadband  |λ| ≤ δ", "deadband"),
                                    ("unstable  λ > δ", "unstable")]:
                    d.rectangle([PANEL_X, y, PANEL_X + 16, y + 14],
                                fill=color_of(cls2), outline=(90, 90, 90))
                    d.text((PANEL_X + 24, y), cname, font=F_SMALL, fill=C_TEXT)
                    y += 19

                draw_strip(d, 128, H - 96, cls_ol, T, t, "open-loop")
                draw_flag_strip(d, 128, H - 52, fg, T, t, "gripper contact")
                writer.append_data(np.asarray(img))
                total += 1
                if total >= args.target_frames:
                    break
            n_demos += 1
            print(f"[{args.task}] demo {did}, {total} frames", flush=True)
    writer.close()
    print(f"RENDER_{args.task}_DONE demos={n_demos} frames={total} "
          f"minutes={total / args.fps / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
