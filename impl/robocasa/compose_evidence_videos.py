"""Composed evidence videos for the switching hypothesis (RoboCasa, GR00T).

Episodes are selected from the replan logs by criteria, not at random:
  PP ep1, ep4:  fixed-16 fails; oracle (16,1) succeeds after a single fire.
  TOSF ep1:     fixed-16 fails; oracle succeeds with 5 fires at the lever.
  PP ep3, TOSF ep7: oracle never fires, leaves the policy alone, success.
  OD ep0:  learned head fires 78/120 replans, episode still fails (honesty).
  PP ep7:  oracle fires 57/101 replans and still fails (honesty).

Side-by-side panels are SEPARATE episodes: scene inits are not paired across
runs (verified visually on first frames), and the note bar says so.
Frame j of each recording is executed step j (counts match exactly).
"""
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon"
EV = os.path.join(ROOT, "results/robocasa/evidence_videos")
OUT = os.path.join(EV, "composed")
PANEL, BAR, NOTE_H = 512, 76, 34
GREEN, RED, GRAY, WHITE = (27, 175, 122), (225, 64, 40), (60, 60, 60), (245, 245, 245)
FPS, FREEZE, FLASH, END_HOLD = 20, 12, 16, 50


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F_BIG, F_MED, F_SML = font(26), font(21), font(17)


def load(name):
    d = json.load(open(os.path.join(EV, f"{name}.json")))
    vids = sorted(glob.glob(os.path.join(EV, f"vid_{name}", "*.mp4")),
                  key=os.path.getmtime)
    assert len(vids) == len(d["episodes"]), (name, len(vids))
    return d, vids


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f[:, :, ::-1].copy())
    cap.release()
    return fr


def panel_spec(run, vids, ep_idx, title, sub, delta, acted=True):
    ep = run["episodes"][ep_idx]
    n = ep["steps"]
    ks = np.full(n, 16, int)
    lam = np.full(n, np.nan)
    rs = ep.get("replans", [])
    for i, r in enumerate(rs):
        t2 = rs[i + 1]["t"] if i + 1 < len(rs) else n
        ks[r["t"]:t2] = r["k"]
        lam[r["t"]:t2] = r["lam"]
    fires = [r["t"] for r in rs if r["k"] == 1]
    frames = read_frames(vids[ep_idx])
    assert len(frames) == n, (title, len(frames), n)
    return {"title": title, "sub": sub, "frames": frames, "ks": ks,
            "lam": lam, "delta": delta, "fires": fires, "steps": n,
            "success": ep["success"], "n_replans": len(rs),
            "n_fired": len(fires), "acted": acted}


def draw_panel(p, t, flash, ended):
    img = Image.new("RGB", (PANEL, BAR + PANEL), GRAY)
    fr = Image.fromarray(p["frames"][min(t, p["steps"] - 1)]).resize(
        (PANEL, PANEL), Image.LANCZOS)
    img.paste(fr, (0, BAR))
    d = ImageDraw.Draw(img, "RGBA")
    k = int(p["ks"][min(t, p["steps"] - 1)])
    lam = p["lam"][min(t, p["steps"] - 1)]
    d.text((10, 6), p["title"], font=F_BIG, fill=WHITE)
    kcol = RED if (k == 1 and p["acted"]) else WHITE
    lam_s = "--" if np.isnan(lam) else f"{lam:.3f}"
    d.text((10, 42), f"t {min(t, p['steps'] - 1):>3}   k={k}   "
           f"lam {lam_s} vs delta {p['delta']:.3f}", font=F_MED, fill=kcol)
    bcol = RED if (k == 1 and p["acted"]) else GREEN
    for w in range(6):
        d.rectangle([w, BAR + w, PANEL - 1 - w, BAR + PANEL - 1 - w],
                    outline=bcol)
    if flash:
        d.rectangle([0, BAR + 190, PANEL, BAR + 250], fill=(225, 64, 40, 215))
        d.text((PANEL // 2, BAR + 220),
               f"FIRE  lam {lam_s} > {p['delta']:.3f}  ->  k=1",
               font=F_BIG, fill=WHITE, anchor="mm")
    if ended:
        col = GREEN if p["success"] else RED
        d.rectangle([0, BAR + 180, PANEL, BAR + 290], fill=col + (225,))
        d.text((PANEL // 2, BAR + 212),
               "SUCCESS" if p["success"] else "FAIL",
               font=F_BIG, fill=WHITE, anchor="mm")
        nk1 = int((p["ks"] == 1).sum())
        stat = f"fired {p['n_fired']}/{p['n_replans']} replans"
        if p["acted"] and p["n_fired"]:
            stat += f", {nk1} of {p['steps']} steps at k=1"
        d.text((PANEL // 2, BAR + 252), stat, font=F_SML, fill=WHITE,
               anchor="mm")
    return np.asarray(img)


def render(out_name, panels, fire_panel=None, note=""):
    W = PANEL * len(panels) + 8 * (len(panels) - 1)
    H = BAR + PANEL + (NOTE_H if note else 0)
    path = os.path.join(OUT, out_name)
    vw = None
    for cc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), FPS, (W, H))
        if vw.isOpened():
            break
    assert vw is not None and vw.isOpened()
    L = max(p["steps"] for p in panels)
    fp = panels[fire_panel] if fire_panel is not None else None
    freeze_at = set(fp["fires"]) if fp and fp["n_fired"] <= 8 else set()
    flash_ranges = ([(ft, ft + FLASH) for ft in fp["fires"]]
                    if fp and fp["n_fired"] <= 12 else [])

    def emit(t, flash):
        tiles = []
        for i, p in enumerate(panels):
            fl = flash and i == fire_panel
            tiles.append(draw_panel(p, t, fl, ended=(t >= p["steps"])))
            if i < len(panels) - 1:
                tiles.append(np.full((BAR + PANEL, 8, 3), 60, np.uint8))
        row = np.concatenate(tiles, axis=1)
        if note:
            strip = Image.new("RGB", (W, NOTE_H), (25, 25, 25))
            ImageDraw.Draw(strip).text((W // 2, NOTE_H // 2), note,
                                       font=F_SML, fill=(200, 200, 200),
                                       anchor="mm")
            row = np.concatenate([row, np.asarray(strip)], axis=0)
        vw.write(row[:, :, ::-1])

    for t in range(L):
        if t in freeze_at:
            for _ in range(FREEZE):
                emit(t, True)
        emit(t, any(a <= t < b for a, b in flash_ranges))
    for _ in range(END_HOLD):
        emit(L, False)
    vw.release()
    print(f"{out_name}: {L} steps -> {os.path.getsize(path) / 1e6:.1f} MB",
          flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    pp_sh, pp_sh_v = load("PP_shadow16_vid")
    pp_or, pp_or_v = load("PP_oracle161_vid")
    tf_sh, tf_sh_v = load("TOSF_shadow16_vid")
    tf_or, tf_or_v = load("TOSF_oracle161_vid")
    od_pr, od_pr_v = load("OD_pred161_vid")
    D_PP, D_TF, D_OD = 0.035765, 0.077113, 0.054712
    NOTE = "separate episodes: scene inits are not paired across runs"

    for ep in (1, 4):
        render(f"PP_rescue_pair_ep{ep}.mp4", [
            panel_spec(pp_sh, pp_sh_v, ep, "fixed k=16",
                       "PickPlace", D_PP, acted=False),
            panel_spec(pp_or, pp_or_v, ep, "oracle (16,1)",
                       "PickPlace", D_PP)], fire_panel=1, note=NOTE)
    render("TOSF_rescue_pair_ep1.mp4", [
        panel_spec(tf_sh, tf_sh_v, 1, "fixed k=16",
                   "SinkFaucet", D_TF, acted=False),
        panel_spec(tf_or, tf_or_v, 1, "oracle (16,1)",
                   "SinkFaucet", D_TF)], fire_panel=1, note=NOTE)
    render("PP_leftalone_ep3.mp4", [
        panel_spec(pp_or, pp_or_v, 3, "oracle (16,1) - 0 fires",
                   "PickPlace", D_PP)], fire_panel=0)
    render("TOSF_leftalone_ep7.mp4", [
        panel_spec(tf_or, tf_or_v, 7, "oracle (16,1) - 0 fires",
                   "SinkFaucet", D_TF)], fire_panel=0)
    render("OD_highfire_fail_ep0.mp4", [
        panel_spec(od_pr, od_pr_v, 0, "learned head (16,1)",
                   "OpenDrawer", D_OD)], fire_panel=0)
    render("PP_highfire_fail_ep7.mp4", [
        panel_spec(pp_or, pp_or_v, 7, "oracle (16,1)",
                   "PickPlace", D_PP)], fire_panel=0)


if __name__ == "__main__":
    main()
