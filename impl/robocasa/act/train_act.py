"""Train ACT on a rendered RoboCasa task dataset."""
import argparse
import io
import os
import sys
import time

import h5py
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from act_model import ACT, CHUNK  # noqa: E402

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class Demos(torch.utils.data.Dataset):
    def __init__(self, path):
        self.frames, self.prop, self.acts, self.index = [], [], [], []
        with h5py.File(path, "r") as f:
            for name in f["data"]:
                g = f["data"][name]
                i = len(self.frames)
                self.frames.append([bytes(b) for b in g["frames"][()]])
                self.prop.append(g["proprio"][()])
                self.acts.append(g["actions"][()])
                self.index += [(i, t) for t in range(len(self.prop[-1]))]
        allp = np.concatenate(self.prop)
        alla = np.concatenate(self.acts)
        self.p_mean, self.p_std = allp.mean(0), allp.std(0) + 1e-6
        self.a_mean, self.a_std = alla.mean(0), alla.std(0) + 1e-6

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, t = self.index[idx]
        img = np.asarray(Image.open(io.BytesIO(self.frames[i][t])),
                         np.float32) / 255.0
        img = ((img - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1)
        prop = (self.prop[i][t] - self.p_mean) / self.p_std
        T = len(self.acts[i])
        a = np.zeros((CHUNK, self.acts[i].shape[1]), np.float32)
        n = min(CHUNK, T - t)
        a[:n] = (self.acts[i][t:t + n] - self.a_mean) / self.a_std
        pad = np.zeros(CHUNK, bool)
        pad[n:] = True
        return (img.astype(np.float32), prop.astype(np.float32), a, pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = Demos(args.data)
    print(f"dataset: {len(ds)} steps", flush=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=8,
        pin_memory=True, drop_last=True, persistent_workers=True)
    model = ACT().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    stats = {"p_mean": ds.p_mean, "p_std": ds.p_std,
             "a_mean": ds.a_mean, "a_std": ds.a_std}

    def save(tag):
        torch.save({"model": model.state_dict(), "stats": stats},
                   os.path.join(args.out, f"act_{tag}.pt"))

    step, t0, run_l1, run_kl = 0, time.time(), 0.0, 0.0
    while step < args.steps:
        for img, prop, acts, pad in dl:
            img, prop = img.cuda(non_blocking=True), prop.cuda(non_blocking=True)
            acts, pad = acts.cuda(non_blocking=True), pad.cuda(non_blocking=True)
            a_hat, mu, logvar = model(img, prop, acts, pad)
            l1 = (torch.abs(a_hat - acts).mean(-1)[~pad]).mean()
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                  ).sum(-1).mean()
            loss = l1 + args.kl * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_l1 += l1.item(); run_kl += kl.item(); step += 1
            if step % 500 == 0:
                print(f"step {step} l1 {run_l1/500:.4f} kl {run_kl/500:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                run_l1 = run_kl = 0.0
            if step % 10000 == 0:
                save(step)
            if step >= args.steps:
                break
    save("final")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
