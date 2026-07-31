"""Compact faithful ACT (Zhao et al. 2023): CVAE + DETR-style transformer.

Hyperparameters follow the paper: d_model 512, 8 heads, ff 3200, 4 encoder
and 7 decoder layers, z dim 32, KL weight 10, ResNet18 backbone, one camera,
chunk length 16, 12-dim actions, 9-dim proprio.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision

CHUNK = 16
ADIM = 12
PDIM = 9
D = 512
ZDIM = 32


def sine_pos_2d(d, h, w):
    """(h*w, d) fixed 2D sine embedding, half dims per axis."""
    def one(n, dd):
        pos = torch.arange(n).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dd, 2).float()
                        * (-np.log(10000.0) / dd))
        pe = torch.zeros(n, dd)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe
    ph = one(h, d // 2).unsqueeze(1).expand(h, w, d // 2)
    pw = one(w, d // 2).unsqueeze(0).expand(h, w, d // 2)
    return torch.cat([ph, pw], dim=-1).reshape(h * w, d)


def sine_pos_1d(d, n):
    pos = torch.arange(n).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ACT(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            bb = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        except Exception:
            bb = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(bb.children())[:-2])
        self.img_proj = nn.Conv2d(512, D, 1)
        self.register_buffer("img_pos", sine_pos_2d(D, 7, 7))

        # CVAE style encoder
        self.enc_cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.enc_prop = nn.Linear(PDIM, D)
        self.enc_act = nn.Linear(ADIM, D)
        self.register_buffer("enc_pos", sine_pos_1d(D, CHUNK + 2))
        layer = nn.TransformerEncoderLayer(D, 8, 3200, 0.1, batch_first=True)
        self.style_enc = nn.TransformerEncoder(layer, 4)
        self.z_head = nn.Linear(D, 2 * ZDIM)

        # main transformer
        self.z_proj = nn.Linear(ZDIM, D)
        self.prop_proj = nn.Linear(PDIM, D)
        self.extra_pos = nn.Parameter(torch.randn(2, D) * 0.02)
        self.trunk = nn.Transformer(
            D, 8, num_encoder_layers=4, num_decoder_layers=7,
            dim_feedforward=3200, dropout=0.1, batch_first=True)
        self.queries = nn.Parameter(torch.randn(CHUNK, D) * 0.02)
        self.head = nn.Linear(D, ADIM)

    def encode_style(self, proprio, actions, is_pad):
        B = actions.shape[0]
        toks = torch.cat([self.enc_cls.expand(B, 1, D),
                          self.enc_prop(proprio).unsqueeze(1),
                          self.enc_act(actions)], dim=1)
        toks = toks + self.enc_pos.unsqueeze(0)
        mask = torch.cat([torch.zeros(B, 2, dtype=torch.bool,
                                      device=actions.device), is_pad], dim=1)
        out = self.style_enc(toks, src_key_padding_mask=mask)
        mu, logvar = self.z_head(out[:, 0]).chunk(2, dim=-1)
        return mu, logvar

    def forward(self, img, proprio, actions=None, is_pad=None):
        B = img.shape[0]
        if actions is not None:
            mu, logvar = self.encode_style(proprio, actions, is_pad)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            mu = logvar = None
            z = torch.zeros(B, ZDIM, device=img.device)
        feat = self.img_proj(self.backbone(img))            # B,D,7,7
        feat = feat.flatten(2).transpose(1, 2) + self.img_pos.unsqueeze(0)
        src = torch.cat([(self.z_proj(z) + self.extra_pos[0]).unsqueeze(1),
                         (self.prop_proj(proprio)
                          + self.extra_pos[1]).unsqueeze(1),
                         feat], dim=1)
        out = self.trunk(src, self.queries.unsqueeze(0).expand(B, -1, -1))
        return self.head(out), mu, logvar
