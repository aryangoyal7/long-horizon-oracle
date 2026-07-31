"""Side-by-side demo videos for the composed long-horizon task.

Left panel: fixed k=16. Right panel: predictor switching (16,1).
Overlays: current chunk length at all times, timestep, stage indicator,
a SWITCH banner when the executed chunk length changes, and a colored border
on the switching panel (green while k=16, red while k=1). Episodes end with a
frozen SUCCESS/FAIL card. Frame j of a recording corresponds to loop step
j-1 (the first frame is the door-closing noop step).
"""
import glob
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon"
OUT = os.path.join(ROOT, "results/robocasa/sequence_videos")
DEMO = os.path.join(OUT, "demos")
BAR = 64
GREEN, RED, GRAY = (27, 175, 122), (235, 104, 52), (120, 120, 120)
FLASH_FRAMES = 12


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F_BIG, F_MED = font(26), font(19)


def load_cell(name):
    d = json.load(open(os.path.join(OUT, f"{name}.json")))
    vids = sorted(glob.glob(os.path.join(OUT, f"vid_{name}", "*.mp4")),
                  key=os.path.getmtime)
    eps = d["episodes"]
    assert len(vids) >= len(eps), (len(vids), len(eps))
    return d, eps, vids[: len(eps)]


def k_series(ep, mode, k_fixed=None, k_stable=16):
    n = ep["steps"]
    if mode == "fixed":
        return np.full(n, k_fixed, dtype=int)
    ks = np.full(n, k_stable, dtype=int)
    rs = ep.get("replans", [])
    for i, r in enumerate(rs):
        t_end = rs[i + 1]["t"] if i + 1 < len(rs) else n
        ks[r["t"]: t_end] = r["k"]
    return ks


def annotate(frame, title, k, t, stage, outcome, border, flash_text):
    # upscale small sim frames so the text bar has room
    if frame.shape[1] < 500:
        im = Image.fromarray(frame)
        frame = np.asarray(im.resize((im.width * 2, im.height * 2),
                                     Image.LANCZOS))
    h, w = frame.shape[:2]
    img = Image.new("RGB", (w, h + BAR), (18, 18, 18))
    img.paste(Image.fromarray(frame), (0, BAR))
    dr = ImageDraw.Draw(img)
    dr.text((10, 6), title, font=F_MED, fill=(230, 230, 230))
    kcol = GREEN if k >= 16 else RED
    dr.text((10, 34), f"chunk k = {k}", font=F_BIG, fill=kcol)
    ttxt = f"t = {t}"
    dr.text((w - 10 - dr.textlength(ttxt, font=F_MED), 6), ttxt,
            font=F_MED, fill=(200, 200, 200))
    dr.text((w - 10 - dr.textlength(stage, font=F_MED), 38), stage,
            font=F_MED, fill=(255, 214, 90))
    if flash_text:
        tw = dr.textlength(flash_text, font=F_BIG)
        dr.rectangle([w // 2 - tw / 2 - 12, BAR + 6,
                      w // 2 + tw / 2 + 12, BAR + 46], fill=(255, 214, 90))
        dr.text((w // 2 - tw / 2, BAR + 12), flash_text, font=F_BIG,
                fill=(0, 0, 0))
    if outcome:
        col = GREEN if "SUCCESS" in outcome else RED
        tw = dr.textlength(outcome, font=F_BIG)
        dr.rectangle([w // 2 - tw / 2 - 16, h // 2 + BAR - 26,
                      w // 2 + tw / 2 + 16, h // 2 + BAR + 26], fill=col)
        dr.text((w // 2 - tw / 2, h // 2 + BAR - 14), outcome,
                font=F_BIG, fill=(255, 255, 255))
    a = np.asarray(img).copy()
    a[:6, :] = border; a[-6:, :] = border; a[:, :6] = border; a[:, -6:] = border
    return a


def panel_frames(vid, ep, ks, title, fixed_border):
    frames = imageio.mimread(vid, memtest=False)
    out = []
    n = len(frames)
    flash_until, flash_text = -1, ""
    for j, fr in enumerate(frames):
        t = min(max(j - 1, 0), len(ks) - 1)
        k = int(ks[t])
        if fixed_border is None:
            if t > 0 and ks[t] != ks[t - 1] and j > 1:
                flash_until = j + FLASH_FRAMES
                flash_text = f"SWITCH {int(ks[t-1])} -> {int(ks[t])}"
            border = GREEN if k >= 16 else RED
        else:
            border = fixed_border
        s1 = ep.get("stage1_end")
        stage = ("stage 1: open cabinet" if (s1 is None or t < s1)
                 else "stage 2: pick and place")
        out.append(annotate(np.asarray(fr), title, k, t, stage, None,
                            border, flash_text if j <= flash_until else ""))
    outcome = (f"SUCCESS at t={ep['steps']}" if ep["success"]
               else f"FAIL (horizon {ep['steps']})")
    last = annotate(np.asarray(frames[-1]), title, int(ks[-1]),
                    ep["steps"], "", outcome,
                    fixed_border if fixed_border is not None
                    else (GREEN if ks[-1] >= 16 else RED), "")
    return out, last


def compose(fixed_ep, fixed_vid, pred_ep, pred_vid, path):
    kf = k_series(fixed_ep, "fixed", k_fixed=16)
    kp = k_series(pred_ep, "predictor")
    L, Llast = panel_frames(fixed_vid, fixed_ep, kf,
                            "FIXED k = 16", GRAY)
    R, Rlast = panel_frames(pred_vid, pred_ep, kp,
                            "SWITCHING 16 -> 1", None)
    n = max(len(L), len(R)) + 30
    h = max(L[0].shape[0], R[0].shape[0])

    def fit(a):
        if a.shape[0] != h:
            img = Image.fromarray(a)
            w2 = int(a.shape[1] * h / a.shape[0])
            a = np.asarray(img.resize((w2, h)))
        return a

    w = imageio.get_writer(path, fps=10, codec="h264", quality=7)
    for j in range(n):
        lf = fit(L[j] if j < len(L) else Llast)
        rf = fit(R[j] if j < len(R) else Rlast)
        w.append_data(np.concatenate([lf, rf], axis=1))
    w.close()
    print("wrote", path, flush=True)


def main():
    os.makedirs(DEMO, exist_ok=True)
    fd, feps, fvids = load_cell("seqvid_fixed_16")
    pd_, peps, pvids = load_cell("seqvid_pred_16-1")
    p_succ = [i for i, e in enumerate(peps) if e["success"]]
    f_fail = [i for i, e in enumerate(feps) if not e["success"]]
    f_succ = [i for i, e in enumerate(feps) if e["success"]]
    pairs = []
    if f_fail and p_succ:
        pairs.append((f_fail[0], p_succ[0]))
    if f_succ and p_succ:
        pairs.append((f_succ[0], p_succ[1 % len(p_succ)]))
    used_p = {p for _, p in pairs}
    rest_p = [i for i in p_succ if i not in used_p] or p_succ
    rest_f = (f_fail[1:] or f_fail or f_succ)
    if rest_f and rest_p:
        pairs.append((rest_f[0], rest_p[0]))
    for n_, (fi, pi) in enumerate(pairs[:3]):
        compose(feps[fi], fvids[fi], peps[pi], pvids[pi],
                os.path.join(DEMO, f"demo{n_+1}_fixed16_vs_switch16-1.mp4"))


if __name__ == "__main__":
    main()
