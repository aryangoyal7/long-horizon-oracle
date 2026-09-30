"""Render the label-video clips: nominal + N perturbed branches from one
labeled demonstration stamp, composited into one H.264 clip.

Reproduces the labeled branches exactly rather than approximately. The
labelers reuse one env across a demo's stamps, and robosuite carries state
across reset_to (gripper command ramp, observable timers), so a branch rolled
out from a cold reset is not the branch that was labeled. Each job therefore
replays the labeler's own call sequence from the demo's reset up to the target
stamp (same env constructor, same reset_to calls, same RNG draws), records the
target stamp's branches, and recomputes lambda_bar from them. No GL context
exists during that rollout (on RoboCasa even attaching one shifted lambda_bar at
the 1e-5 level); frames are rendered afterwards by restoring each recorded
state. The recomputed value is stored next to the label's value; a clip whose
branches do not reproduce the label is flagged in the manifest.

  robomimic, MimicGen  impl/labeler/ftle_labeler.py      (lh / mg venv)
  RoboCasa             impl/rc_relabel/ftle_labeler_rc_v2.py + ftle_labeler_robocasa.py (rc venv),
                       replaying the labeling attempt that wrote the kept row (rc_attempt)

Frame j (j = 1..K) is the state after the j-th replayed action. The image is
the nominal branch blended 50/50 with the mean of the perturbed branches, so
where branches separate the scene ghosts. On top: the nominal end-effector
path in white, the perturbed paths in the class colour (the paper's
\\definecolor RGB), and the task, lambda_bar, class and K in the top-left.
K frames at 20 fps, then the last frame held for one second.

usage (inside the matching venv, MUJOCO_GL=egl):
  python render_label_videos.py --platform robomimic [--only lift] [--verify-only]
"""
import argparse
import importlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
ROOT = os.path.dirname(SITE)
for p in ("impl/labeler", "impl/robocasa", "impl/rc_relabel"):
    sys.path.insert(0, os.path.join(ROOT, p))

W, H, FPS, HOLD_S = 640, 480, 20, 1.0
# \definecolor{unst}{RGB}{204,102,92}, {stab}{88,152,120}, {dead}{135,135,140}
CLASS_RGB = {"unstable": (204, 102, 92), "stable": (88, 152, 120),
             "deadband": (135, 135, 140)}
PLATFORM_NAME = {"robomimic": "robomimic", "mimicgen": "MimicGen",
                 "robocasa": "RoboCasa"}
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
DIV_FLOOR = 1e-12


def camera_for(platform, task):
    # the camera each dataset records its third-person images from
    if platform == "robocasa":
        return "robot0_agentview_left"
    return "agentview"


# ----------------------------------------------------------------- rollouts --
class Recorder:
    """Records the target stamp's branches during the labeled rollout: eef
    positions and task vectors from the observations, and the flattened sim
    state after every step. Nothing is rendered here; frames are drawn
    afterwards from the recorded states (render_recorded), so no GL context
    or render call ever runs inside the rollout that reproduces the label."""

    def __init__(self, env, cam):
        self.env, self.cam = env, cam
        self.eef, self.task, self.states, self.start_state = [], [], [], None

    def _state(self):
        return np.array(self.env.env.sim.get_state().flatten(), copy=True)

    def start_branch(self, obs):
        if self.start_state is None:
            self.start_state = self._state()
        self.eef.append([np.asarray(obs["robot0_eef_pos"]).copy()])
        self.task.append([])
        self.states.append([])

    def step(self, obs, tv, nominal=False):
        self.eef[-1].append(np.asarray(obs["robot0_eef_pos"]).copy())
        self.task[-1].append(tv)
        self.states[-1].append(self._state())


def render_recorded(rec, job):
    """Frames and per-frame camera matrices from the recorded states."""
    import robosuite.utils.camera_utils as CU
    attach_renderer(rec.env, job)
    sim = rec.env.env.sim

    def at(state):
        sim.set_state_from_flattened(state)
        sim.forward()

    def cam():
        return CU.get_camera_transform_matrix(sim, rec.cam, H, W)

    at(rec.start_state)
    rec.cams = [cam()]
    rec.frames = []
    for b, states in enumerate(rec.states):
        fb = []
        for st in states:
            at(st)
            fb.append(np.asarray(rec.env.render(mode="rgb_array", height=H, width=W,
                                                camera_name=rec.cam)).copy())
            if b == 0:
                rec.cams.append(cam())
        rec.frames.append(fb)


def attach_renderer(env, job):
    """Offscreen GL context for sim.render (its constructor calls sim.forward()),
    attached only after the labeled rollout is complete."""
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    MjRenderContextOffscreen(env.env.sim, device_id=int(job["gpu"]), max_width=W, max_height=H)


def rollout_robomimic(job, render):
    """ftle_labeler.process_demo, call for call, up to the target stamp."""
    FL = importlib.import_module("ftle_labeler")
    import h5py
    import robomimic.utils.file_utils as FileUtils
    ds, cfg, clip = job["sim_dataset"], job["cfg"], job["clip"]
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=ds)
    with h5py.File(ds, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        dk = f"demo_{clip['demo_id']}"
        i = demos.index(dk)                     # labeler seed = seed + i
        g = f["data"][dk]
        model_xml, states, actions = g.attrs["model_file"], g["states"][()], g["actions"][()]
    K, N, stride = cfg["K"], cfg["N"], cfg["stride"]
    sigma, npd = cfg["sigma_u"], 3               # pos_only: translation dims
    env = FL.make_env(env_meta)
    rng = np.random.default_rng(int(cfg["seed"]) + i)
    env.reset()
    env.reset_to({"model": model_xml})
    T = actions.shape[0]
    rec = None
    for t0 in range(0, T - K, stride):
        target = t0 == clip["t"]
        env.reset_to({"states": states[t0]})           # contact-flag pass
        obs = env.reset_to({"states": states[t0]})
        if target:
            rec = Recorder(env, job["camera"])
            rec.start_branch(obs)
        nom = []
        for j in range(K):
            obs, _, _, _ = env.step(actions[t0 + j])
            nom.append(FL.task_vec(obs))
            if target:
                rec.step(obs, nom[-1], True)
        for n in range(N):
            a0 = actions[t0].copy()
            a0[:npd] += sigma * rng.standard_normal(npd)
            obs = env.reset_to({"states": states[t0]})
            if target:
                rec.start_branch(obs)
            for j in range(K):
                obs, _, _, _ = env.step(a0 if j == 0 else actions[t0 + j])
                if target:
                    rec.step(obs, FL.task_vec(obs), False)
        if target:
            return rec
    raise RuntimeError(f"stamp t={clip['t']} not reached (T={T})")


def rc_attempt(task, demo_id, t, lam):
    """The call history of the labeling attempt whose row became the label.

    The RoboCasa demo pass wrote one JSONL row per stamp and resumed demos
    across rounds and relaunches, so a stamp can have been labeled by a fresh
    attempt (replay gate, then stamps from t=0) or by a resumed one (no replay
    gate, first pending stamp onward), sometimes both. The consolidated label
    kept one row; this finds it and returns (ran_replay_gate, [t0, ...] up to
    and including t) for the attempt that wrote it."""
    import glob
    C = importlib.import_module("common_rc")
    for f in sorted(glob.glob(f"{C.RUN_DIR}/demo_stamps/{task}_w*.jsonl")):
        rows = [r for r in map(json.loads, open(f)) if r.get("demo_id") == demo_id]
        attempts, cur = [], None
        for r in rows:
            if r["type"] == "replay" or cur is None or (
                    r["type"] == "stamp" and cur["ts"] and r["t"] <= cur["ts"][-1]):
                cur = {"replay": r["type"] == "replay", "ts": [], "lam": []}
                attempts.append(cur)
            if r["type"] == "stamp":
                cur["ts"].append(r["t"]); cur["lam"].append(r["lambda_mean"])
        for a in attempts:
            for i, (tt, ll) in enumerate(zip(a["ts"], a["lam"])):
                if tt == t and abs(ll - lam) < 5e-7:
                    return a["replay"], a["ts"][: i + 1], os.path.basename(f)
    raise RuntimeError(f"no labeling row for {task} demo {demo_id} t={t} lambda={lam}")


def rollout_robocasa(job, render):
    """ftle_labeler_rc_v2.process_demo, replaying the attempt that wrote the label."""
    FR = importlib.import_module("ftle_labeler_robocasa")
    C = importlib.import_module("common_rc")
    import h5py
    ds, cfg, clip = job["sim_dataset"], job["cfg"], job["clip"]
    task = job["task"]
    with h5py.File(ds, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        g = f["data"][f"demo_{clip['demo_id']}"]
        model_xml, ep_meta = g.attrs["model_file"], g.attrs["ep_meta"]
        states = g["states"][()]
        actions = FR.lerobot_to_env(g["actions"][()])
    K, N, stride, sigma = cfg["K"], cfg["N"], cfg["stride"], cfg["sigma_u"]
    env = FR.make_env(env_meta)
    FR.reset_demo(env, model_xml, ep_meta)
    p0 = FR.arm_pos_indices(env)
    ran_gate, history, src = rc_attempt(task, clip["demo_id"], clip["t"], clip["lambda_bar"])
    job["history"] = {"replay_gate": ran_gate, "first_t": history[0], "worker_file": src}
    if ran_gate:
        FR.replay_error(env, states, actions)
    for t0 in history:
        target = t0 == clip["t"]
        rng = np.random.default_rng(C.derive_seed(task, clip["demo_id"], t0, "sigma-rmse-v2"))
        FR.set_state(env, states[t0])
        obs = env.reset_to({"states": np.asarray(states[t0])})
        if target:
            rec = Recorder(env, job["camera"])
            rec.start_branch(obs)
        nom = []
        for j in range(K):
            obs, _, _, _ = env.step(actions[t0 + j])
            nom.append(FR.task_vec(obs))
            if target:
                rec.step(obs, nom[-1], True)
        for n in range(N):
            a0 = actions[t0].copy()
            a0[p0:p0 + 3] += sigma * rng.standard_normal(3)
            obs = env.reset_to({"states": np.asarray(states[t0])})
            if target:
                rec.start_branch(obs)
            for j in range(K):
                obs, _, _, _ = env.step(a0 if j == 0 else actions[t0 + j])
                if target:
                    rec.step(obs, FR.task_vec(obs), False)
        if target:
            return rec
    raise RuntimeError(f"stamp t={clip['t']} not reached (T={T})")


def lambda_from(rec):
    """Per-branch OLS slope of log distance to the nominal, as the labelers fit it."""
    nom = np.asarray(rec.task[0])
    taus = np.arange(1, nom.shape[0] + 1, dtype=np.float64)
    tc = taus - taus.mean()
    slopes = []
    for b in rec.task[1:]:
        y = np.log(np.maximum(np.linalg.norm(np.asarray(b) - nom, axis=1), DIV_FLOOR))
        slopes.append(float((tc * (y - y.mean())).sum() / (tc * tc).sum()))
    s = np.asarray(slopes)
    return float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s))), slopes


# ---------------------------------------------------------------- drawing ----
def _fonts():
    from PIL import ImageFont
    return (ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 19),
            ImageFont.truetype(f"{FONT_DIR}/DejaVuSans.ttf", 17),
            ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 17))


def project(cam, pts):
    import robosuite.utils.camera_utils as CU
    rc = CU.project_points_from_world_to_camera(np.asarray(pts), cam, H, W)
    return rc  # (row, col)


def compose(rec, job, lam_label):
    from PIL import Image, ImageDraw
    cls, K = job["clip"]["class"], job["cfg"]["K"]
    col = CLASS_RGB[cls]
    f_title, f_body, f_bold = _fonts()
    SS = 2  # supersampled overlay for anti-aliased lines
    out = []
    # pixel paths: every branch projected with the nominal camera at that frame
    paths = []
    for b in range(len(rec.eef)):
        pts = []
        for j, p in enumerate(rec.eef[b]):
            r, c = project(rec.cams[j], p[None])[0]
            pts.append((c * SS + SS / 2, r * SS + SS / 2))
        paths.append(pts)
    title = f"{PLATFORM_NAME[job['platform']]} · {job['task']}"
    for j in range(1, K + 1):
        nom = rec.frames[0][j - 1].astype(np.float32)
        pert = np.mean([rec.frames[b][j - 1] for b in range(1, len(rec.frames))], axis=0)
        base = Image.fromarray(np.clip(0.5 * nom + 0.5 * pert, 0, 255).astype(np.uint8))
        ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for b in list(range(1, len(paths))) + [0]:
            seg = paths[b][: j + 1]
            rgb = (255, 255, 255) if b == 0 else col
            wid = (5 if b == 0 else 4) * SS // 2
            d.line(seg, fill=(0, 0, 0, 110), width=wid + 3 * SS // 2, joint="curve")
            d.line(seg, fill=rgb + (255,), width=wid, joint="curve")
            x, y = seg[-1]
            rr = (5 if b == 0 else 4) * SS // 2
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=rgb + (255,),
                      outline=(0, 0, 0, 160), width=SS)
        x0, y0 = paths[0][0]
        d.ellipse([x0 - 4 * SS, y0 - 4 * SS, x0 + 4 * SS, y0 + 4 * SS],
                  outline=(255, 255, 255, 255), width=SS)
        ov = ov.resize((W, H), Image.LANCZOS)
        img = Image.alpha_composite(base.convert("RGBA"), ov)
        # text block, top-left
        tb = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        td = ImageDraw.Draw(tb)
        lam_txt = f" = {lam_label:+.3f}"
        lines_h = 3 * 26 + 12
        lw = max(td.textlength(title, font=f_title),
                 td.textlength("λ" + lam_txt + "   " + cls.upper(), font=f_bold) + 4) + 20
        td.rounded_rectangle([8, 8, 8 + lw, 8 + lines_h], radius=6, fill=(0, 0, 0, 165))
        x, y = 18, 14
        td.text((x, y), title, font=f_title, fill=(255, 255, 255, 255))
        y += 27
        # lambda with a drawn macron (combining U+0304 is not laid out by PIL)
        lx0, ly0, lx1, _ = td.textbbox((x, y), "λ", font=f_bold)
        td.text((x, y), "λ", font=f_bold, fill=(255, 255, 255, 255))
        td.line([(lx0 + 1, ly0 - 3), (lx1 - 1, ly0 - 3)], fill=(255, 255, 255, 255), width=2)
        x2 = x + td.textlength("λ", font=f_bold)
        td.text((x2, y), lam_txt, font=f_bold, fill=(255, 255, 255, 255))
        x3 = x2 + td.textlength(lam_txt + "   ", font=f_bold)
        td.text((x3, y), cls.upper(), font=f_bold, fill=tuple(min(255, int(c * 1.25)) for c in col) + (255,))
        y += 26
        td.text((x, y), f"K = {K}", font=f_body, fill=(255, 255, 255, 255))
        img = Image.alpha_composite(img, tb).convert("RGB")
        out.append(np.asarray(img))
    out += [out[-1]] * int(round(HOLD_S * FPS))
    return out


def encode(frames, path):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "18",
           "-movflags", "+faststart", path + ".part.mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed for {path}"
    os.replace(path + ".part.mp4", path)


# ------------------------------------------------------------------- jobs ----
def run_job(job):
    t_start = time.time()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(job["gpu"])
    render = not job["verify_only"]
    try:
        fn = rollout_robocasa if job["platform"] == "robocasa" else rollout_robomimic
        rec = fn(job, render)
        lam, se, slopes = lambda_from(rec)          # from the un-rendered rollout
        if render:
            render_recorded(rec, job)
        clip = job["clip"]
        res = dict(file=clip["file"], demo_id=clip["demo_id"], t=clip["t"],
                   lambda_label=clip["lambda_bar"], lambda_se_label=clip["lambda_se"],
                   lambda_rendered=lam, lambda_se_rendered=se, branch_slopes=slopes,
                   abs_diff=abs(lam - clip["lambda_bar"]), camera=job["camera"],
                   history=job.get("history", {"replay_gate": None, "first_t": 0,
                                               "note": "demo replayed from its reset"}))
        if render:
            frames = compose(rec, job, clip["lambda_bar"])
            path = os.path.join(job["out_dir"], clip["file"])
            encode(frames, path)
            res.update(n_frames=len(frames), bytes=os.path.getsize(path))
        res["seconds"] = round(time.time() - t_start, 1)
        return res
    except Exception:
        import traceback
        return dict(file=job["clip"]["file"], error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True, choices=["robomimic", "mimicgen", "robocasa"])
    ap.add_argument("--stamps", default=os.path.join(HERE, "stamps.json"))
    ap.add_argument("--out-dir", default=os.path.join(SITE, "static", "videos"))
    ap.add_argument("--log-dir", default=os.path.join(HERE, "render_log"))
    ap.add_argument("--only", nargs="*", default=None, help="task names")
    ap.add_argument("--files", nargs="*", default=None, help="clip file names")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-render clips that already have an mp4 and a successful log entry")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    jobs = []
    for s in json.load(open(args.stamps)):
        if s["platform"] != args.platform or (args.only and s["task"] not in args.only):
            continue
        cfg = {k: s[k] for k in ("K", "N", "stride", "sigma_u", "seed")}
        for c in s["clips"]:
            if not c["available"] or (args.files and c["file"] not in args.files):
                continue
            jobs.append(dict(platform=s["platform"], task=s["task"], clip=c, cfg=cfg,
                             sim_dataset=s["sim_dataset"], camera=camera_for(s["platform"], s["task"]),
                             out_dir=args.out_dir, verify_only=args.verify_only,
                             gpu=len(jobs) % 8))
    tag = "verify" if args.verify_only else "render"
    log = os.path.join(args.log_dir, f"{tag}_{args.platform}.jsonl")
    if not args.verify_only and not args.force and os.path.exists(log):
        done = {r["file"] for r in map(json.loads, open(log)) if "error" not in r}
        jobs = [j for j in jobs if not (j["clip"]["file"] in done and os.path.exists(
            os.path.join(args.out_dir, j["clip"]["file"])))]
    print(f"{len(jobs)} clips to do on {args.platform}", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, max(1, len(jobs)))) as pool, open(log, "a") as fo:
        for r in pool.imap_unordered(run_job, jobs):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            if "error" in r:
                print(f"FAIL {r['file']}\n{r['error']}", flush=True)
            else:
                print(f"{r['file']}: lambda label {r['lambda_label']:+.6f} rendered "
                      f"{r['lambda_rendered']:+.6f} |diff| {r['abs_diff']:.2e}"
                      + (f" {r.get('bytes', 0)/1e6:.2f} MB" if 'bytes' in r else "")
                      + f" ({r['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
