"""
geometry.py — the Fisher-Rao pullback metric of the NB-VAE.

The statistical manifold is the family {NB(mu(z), theta)}.  The Fisher information of a
negative binomial wrt its mean mu (theta fixed) is

        I(mu) = theta / (mu (mu + theta)) = 1 / Var(x).            (derived analytically)

Pulling this metric back to the latent space z through the decoder mean mu_g(z)=ell*rho_g(z):

        G(z) = sum_g I(mu_g) (d mu_g/dz)(d mu_g/dz)^T
             = ell^2 * J_rho(z)^T diag(w) J_rho(z),   w_g = theta_g/(mu_g(mu_g+theta_g)),

where J_rho = d rho / d z is the (genes x d) decoder Jacobian, computed exactly and cheaply
with torch.func.jacrev + vmap because d is small (~10).  No full per-cell Jacobian is ever
materialised beyond a single minibatch, which resolves the scalability concern.

Local (near-neighbour) squared Fisher-Rao distance uses the symmetrised midpoint metric,
a second-order-accurate geodesic surrogate:

        delta^2(i,j) = (z_i - z_j)^T ((G_i + G_j)/2) (z_i - z_j).
"""
from __future__ import annotations
import numpy as np
import torch
from torch.func import jacfwd, vmap


@torch.no_grad()
def _theta_np(model):
    return model.theta().detach().cpu().numpy().astype(np.float64)


def compute_metric_tensors(model, Z, lib, B=None, batch=256, device="cpu", ridge=1e-4,
                           use_fisher=True):
    """
    Return per-cell Fisher-Rao metric tensors G (N,d,d), plus diagnostics:
      logdet  : 0.5*log det G  (Riemannian volume element / "magnification factor")
      trace   : tr(G)          (local statistical stretch)

    use_fisher=True  -> G = ell^2 J_rho^T diag(1/Var) J_rho   (NB Fisher-Rao pullback)
    use_fisher=False -> G = ell^2 J_rho^T J_rho               (Euclidean pullback ablation)
    B : (N, n_batch) one-hot batch matrix (held fixed when differentiating wrt z).
    """
    model = model.to(device).eval()
    Zt = torch.tensor(np.asarray(Z), dtype=torch.float32, device=device)
    libt = torch.tensor(np.asarray(lib), dtype=torch.float32, device=device).reshape(-1, 1)
    if B is None:
        B = np.ones((Zt.shape[0], model.n_batch), dtype=np.float32)
        B[:, 1:] = 0.0
    Bt = torch.tensor(np.asarray(B), dtype=torch.float32, device=device)
    theta = model.theta().detach()                        # (G,)
    d = Zt.shape[1]

    def rho_bz(z, b):                                     # z:(d,), b:(nb,) -> rho:(G,)
        return model.rho(z, b)

    # forward-mode Jacobian wrt z only (b fixed); jacfwd avoids the GxG softmax Jacobian.
    jac_fn = vmap(jacfwd(rho_bz, argnums=0))              # (B,d),(B,nb) -> (B,G,d)

    Gs, logdets, traces = [], [], []
    N = Zt.shape[0]
    for s in range(0, N, batch):
        zb = Zt[s:s + batch]; lb = libt[s:s + batch]; bb = Bt[s:s + batch]
        with torch.no_grad():
            J = jac_fn(zb, bb)                            # (B,G,d)
            rho = model.rho(zb, bb)                       # (B,G)
            mu = lb * rho
            if use_fisher:
                w = theta / (mu * (mu + theta) + 1e-12)   # 1/Var
            else:
                w = torch.ones_like(mu)                   # Euclidean pullback
            wl = (lb ** 2) * w
            Gb = torch.einsum("bgi,bg,bgj->bij", J, wl, J)
            Gb = Gb + ridge * torch.eye(d, device=device).unsqueeze(0)
            _, ld = torch.linalg.slogdet(Gb)
            Gs.append(Gb.cpu().numpy())
            logdets.append((0.5 * ld).cpu().numpy())
            traces.append(torch.diagonal(Gb, dim1=1, dim2=2).sum(-1).cpu().numpy())
    G = np.concatenate(Gs).astype(np.float64)
    return G, np.concatenate(logdets), np.concatenate(traces)


def midpoint_fr_dist2(Z, G, i, j):
    """Vectorised squared Fisher-Rao distance for edge lists i, j."""
    diff = (Z[i] - Z[j])                                   # (E,d)
    Gm = 0.5 * (G[i] + G[j])                               # (E,d,d)
    return np.einsum("ed,edf,ef->e", diff, Gm, diff)


def fr_knn_indices(Z, G, k=30, k_cand=None):
    """Fisher-Rao kNN indices (candidate-and-rerank). Returns (N,k) neighbour indices."""
    from sklearn.neighbors import NearestNeighbors
    n = Z.shape[0]
    k_cand = k_cand or min(4 * k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_cand + 1).fit(Z)
    _, cand = nn.kneighbors(Z)
    cand = cand[:, 1:]
    D2 = local_fr_dist2_to_candidates(Z, G, cand)
    order = np.argsort(D2, axis=1)[:, :k]
    return np.take_along_axis(cand, order, axis=1)


def local_fr_dist2_to_candidates(Z, G, cand):
    """
    Z:(N,d), G:(N,d,d), cand:(N,k) candidate neighbour indices.
    Returns D2:(N,k) squared Fisher-Rao distances (midpoint metric).
    """
    N, k = cand.shape
    d = Z.shape[1]
    Zc = Z[cand]                                           # (N,k,d)
    diff = Z[:, None, :] - Zc                              # (N,k,d)
    Gm = 0.5 * (G[:, None, :, :] + G[cand])                # (N,k,d,d)
    D2 = np.einsum("nkd,nkde,nke->nk", diff, Gm, diff)
    return np.clip(D2, 0, None)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load
    from nbvae import fit_nbvae
    A = load("pbmc3k")
    model, Z, lib = fit_nbvae(A, epochs=60, verbose=False)
    G, logdet, trace = compute_metric_tensors(model, Z, lib)
    print("G shape", G.shape)
    ev = np.linalg.eigvalsh(G)
    print("eigenvalue range:", float(ev.min()), float(ev.max()))
    print("logdet(vol) range:", float(logdet.min()), float(logdet.max()))
    print("condition number median:", float(np.median(ev[:, -1] / np.clip(ev[:, 0], 1e-9, None))))
