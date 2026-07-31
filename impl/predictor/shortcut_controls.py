"""Refined shortcut controls.

The 'any contact' flag is true at every stamp (the object rests on the table),
so the meaningful contact partition is the gripper flag.  The clock shortcut
('later in the episode means less stable') is controlled by computing AUROC
WITHIN episode-progress deciles: if the head is only a clock, its within-stratum
AUROC collapses to chance.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/impl/predictor")
from train_head import AttentiveHead, auroc, spearman

FEAT = "/mnt/scratch/lh/features"
POOL = [f"{FEAT}/dino_{n}.npz" for n in
        ["lift", "can", "square", "tool_hang", "rollouts_lift", "rollouts_can",
         "mg_stack_d0", "mg_square_d0"]]
R = "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/results/predictor"
DELTA = 0.03545861691236496

F, P, lam, demo, t, cg, ct, cf, src = [], [], [], [], [], [], [], [], []
off = 0
for i, p in enumerate(POOL):
    z = np.load(p, allow_pickle=True)
    F.append(z["features"]); P.append(z["proprio"]); lam.append(z["lambda_task"])
    demo.append(z["demo_id"] + off); t.append(z["t"])
    cg.append(z["win_contact_gripper"]); ct.append(z["win_contact_table"])
    cf.append(z["win_contact_fixture"]); src.append(np.full(len(z["t"]), i))
    off += int(z["demo_id"].max()) + 1
F = np.concatenate(F); P = np.concatenate(P).astype(np.float32)
lam = np.concatenate(lam).astype(np.float32); demo = np.concatenate(demo)
t = np.concatenate(t); cg = np.concatenate(cg); ct = np.concatenate(ct)
cf = np.concatenate(cf); src = np.concatenate(src)

val = np.isin(demo, list(set(json.load(open(f"{R}/run1_val_demos.json")))))
prog = np.zeros(len(t), dtype=np.float32)
for d in np.unique(demo):
    m = demo == d
    tm = t[m].astype(np.float32)
    prog[m] = (tm - tm.min()) / max(tm.max() - tm.min(), 1)

y = (lam > DELTA).astype(np.float32)
conf = np.abs(lam) > DELTA

ck = torch.load(f"{R}/ablation/A_di_full_s0/head.pt", map_location="cpu",
                weights_only=False)
P = (P - ck["prop_mu"]) / ck["prop_sd"]
model = AttentiveHead(dim=F.shape[-1], prop_dim=P.shape[-1])
model.load_state_dict(ck["model"]); model.eval().cuda()
idx = np.where(val)[0]
outs = []
with torch.no_grad():
    for i in range(0, len(idx), 512):
        b = idx[i:i+512]
        lg, lh = model(torch.from_numpy(F[b]).float().cuda(),
                       torch.from_numpy(P[b]).cuda())
        outs.append(torch.stack([torch.sigmoid(lg), lh], 1).cpu().numpy())
o = np.concatenate(outs)
phat, lhat = o[:, 0], o[:, 1]
yv, cv, lv = y[val], conf[val], lam[val]
progv, cgv, ctv, cfv = prog[val], cg[val], ct[val], cf[val]

print(f"val stamps {val.sum()}   confident {cv.sum()}")
print(f"flag rates: gripper {cgv.mean():.3f}  table {ctv.mean():.3f}  "
      f"fixture {cfv.mean():.3f}")

def au(score, m):
    m = m & cv
    if m.sum() < 30 or len(np.unique(yv[m])) < 2:
        return None, int(m.sum())
    return auroc(score[m], yv[m]), int(m.sum())

print("\n--- 1. TRIVIAL BASELINES vs HEAD (confident stamps) ---")
allm = np.ones(len(yv), bool)
for nm, s in [("episode progress t", progv.astype(float)),
              ("gripper contact flag", cgv.astype(float)),
              ("fixture contact flag", cfv.astype(float)),
              ("HEAD p(unstable)", phat)]:
    a, n = au(s, allm)
    print(f"  {nm:24s} n={n:6d}  AUROC={a:.3f}")

print("\n--- 2. PARTITION BY GRIPPER CONTACT ---")
for nm, m in [("gripper contact", cgv), ("no gripper contact", ~cgv)]:
    a, n = au(phat, m)
    ap, _ = au(progv.astype(float), m)
    base = yv[m & cv].mean() if (m & cv).sum() else float("nan")
    print(f"  {nm:22s} n={n:6d}  head={a if a is None else round(a,3)}  "
          f"progress={ap if ap is None else round(ap,3)}  P(unstable)={base:.3f}")

print("\n--- 3. CONTROLLING THE CLOCK: AUROC within progress deciles ---")
bins = np.quantile(progv, np.linspace(0, 1, 11))
aus, ns = [], []
for i in range(10):
    m = (progv >= bins[i]) & (progv <= bins[i + 1] if i == 9 else progv < bins[i + 1])
    a, n = au(phat, m)
    if a is not None:
        aus.append(a); ns.append(n)
        print(f"  decile {i} [{bins[i]:.2f},{bins[i+1]:.2f})  n={n:5d}  AUROC={a:.3f}")
w = np.array(ns) / sum(ns)
print(f"  weighted mean within-decile AUROC = {float((np.array(aus)*w).sum()):.3f}"
      f"   (unstratified 0.863; chance 0.5)")

print("\n--- 4. PER-TASK breakdown (confident stamps) ---")
names = ["lift", "can", "square", "tool_hang", "roll_lift", "roll_can",
         "mg_stack", "mg_square"]
for i, nm in enumerate(names):
    m = src[val] == i
    a, n = au(phat, m)
    ap, _ = au(progv.astype(float), m)
    if a is not None:
        print(f"  {nm:10s} n={n:6d}  head={a:.3f}  progress={ap:.3f}")
    else:
        print(f"  {nm:10s} n={n:6d}  (not in val)")
