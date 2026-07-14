"""
nbvae.py — a compact, batch-conditional Negative-Binomial VAE in pure PyTorch.

Generative model (scVI-style, observed library size, batch-conditional decoder & encoder):
    z_i ~ N(0, I_d)
    rho_i = softmax(f_dec([z_i, b_i]))          (gene proportions; b_i = one-hot batch)
    mu_ig = library_i * rho_ig                  (NB mean)
    x_ig ~ NegBinomial(mean=mu_ig, inverse_dispersion=theta_g)
      Var(x_ig) = mu_ig + mu_ig^2 / theta_g

Conditioning the decoder (and encoder) on the batch pushes technical variation out of z, so
the Fisher-Rao pullback metric computed from d rho/d z reflects biology, not batch.  We keep
the decoder BatchNorm-free so it is `torch.func` (jacfwd/vmap) friendly for the metric.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def nb_nll(x, mu, theta, eps=1e-8):
    """Negative binomial negative log-likelihood (mean / inverse-dispersion param)."""
    theta = theta + eps; mu = mu + eps
    log_theta_mu = torch.log(theta + mu)
    ll = (theta * (torch.log(theta) - log_theta_mu)
          + x * (torch.log(mu) - log_theta_mu)
          + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0))
    return -ll.sum(-1)


class MLP(nn.Module):
    def __init__(self, sizes, act=nn.ReLU, norm=True, last_act=False):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 or last_act:
                if norm:
                    layers.append(nn.LayerNorm(sizes[i + 1]))
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NBVAE(nn.Module):
    def __init__(self, n_genes, n_batch=1, d_latent=10, hidden=128, n_layers=1, dropout=0.1):
        super().__init__()
        self.n_genes = n_genes; self.d = d_latent; self.n_batch = n_batch
        self.encoder = MLP([n_genes + n_batch] + [hidden] * n_layers, norm=True, last_act=True)
        self.enc_mu = nn.Linear(hidden, d_latent)
        self.enc_lv = nn.Linear(hidden, d_latent)
        self.drop = nn.Dropout(dropout)
        self.decoder = MLP([d_latent + n_batch] + [hidden] * n_layers + [n_genes], norm=True)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def encode(self, x_log, b):
        h = self.drop(self.encoder(torch.cat([x_log, b], -1)))
        return self.enc_mu(h), self.enc_lv(h)

    @staticmethod
    def reparam(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def rho(self, z, b):
        """Gene proportions softmax(decoder([z, b])). Pure fn of z (b held fixed)."""
        return F.softmax(self.decoder(torch.cat([z, b], -1)), dim=-1)

    def theta(self):
        return F.softplus(self.log_theta) + 1e-4

    def decode_mu(self, z, library, b):
        return library * self.rho(z, b)

    def forward(self, x_log, x_counts, library, b):
        qm, qlv = self.encode(x_log, b)
        z = self.reparam(qm, qlv)
        mu = self.decode_mu(z, library, b)
        theta = self.theta().expand_as(mu)
        recon = nb_nll(x_counts, mu, theta)
        kl = -0.5 * torch.sum(1 + qlv - qm.pow(2) - qlv.exp(), dim=-1)
        return recon, kl, qm, z


def _batch_onehot(adata):
    if "batch" in adata.obs:
        cats = adata.obs["batch"].astype("category")
        codes = cats.cat.codes.values
        nb = len(cats.cat.categories)
    else:
        codes = np.zeros(adata.n_obs, dtype=int); nb = 1
    B = np.zeros((len(codes), nb), dtype=np.float32)
    B[np.arange(len(codes)), codes] = 1.0
    return B, nb


def fit_nbvae(adata, d_latent=10, hidden=128, n_layers=1, epochs=250, lr=1e-3,
              batch_size=256, kl_warmup=50, beta=1.0, weight_decay=1e-5,
              seed=0, device="cpu", verbose=True):
    """Train a batch-conditional NBVAE; returns (model, Z_mean, library, batch_onehot)."""
    torch.manual_seed(seed); np.random.seed(seed)
    C = adata.layers["counts"]
    C = C.toarray() if hasattr(C, "toarray") else np.asarray(C)
    C = torch.tensor(np.asarray(C), dtype=torch.float32)
    lib = C.sum(1, keepdim=True).clamp_min(1.0)
    x_log = torch.log1p(C)
    mu_g = x_log.mean(0, keepdim=True); sd_g = x_log.std(0, keepdim=True).clamp_min(1e-3)
    x_in = (x_log - mu_g) / sd_g
    B_np, nb = _batch_onehot(adata)
    B = torch.tensor(B_np, dtype=torch.float32)

    n, G = C.shape
    model = NBVAE(G, n_batch=nb, d_latent=d_latent, hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dl = DataLoader(TensorDataset(x_in, C, lib, B), batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        bw = beta * min(1.0, (ep + 1) / max(1, kl_warmup))
        tot = 0.0
        for xb, cb, lb, bb in dl:
            xb, cb, lb, bb = (t.to(device) for t in (xb, cb, lb, bb))
            recon, kl, _, _ = model(xb, cb, lb, bb)
            loss = (recon + bw * kl).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * xb.size(0)
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  loss/cell {tot / n:8.2f}  beta {bw:.2f}", flush=True)

    model.eval()
    with torch.no_grad():
        qm, _ = model.encode(x_in.to(device), B.to(device))
    return model, qm.cpu().numpy().astype(np.float32), lib.cpu().numpy().ravel(), B_np


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load
    A = load("pbmc3k")
    model, Z, lib, B = fit_nbvae(A, epochs=80, verbose=True)
    print("latent", Z.shape, "n_batch", B.shape[1],
          "theta", float(model.theta().min()), float(model.theta().max()))
