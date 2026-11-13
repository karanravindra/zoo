"""
e-means in PyTorch (single file, no local imports).

Implements Algorithm 1 of "e-means: outperforming k-means (in one-tenth the
epochs)", Bessa et al., arXiv:2611.09501:

  * mini-batch k-means with EMA updates of per-cluster counts N_j and sums S_j
  * cosine-annealed step alpha = 1 - gamma, from alpha0 down to 0.1 * alpha0
  * dead-cluster revival every 5 batches by splitting a large "donor" cluster
  * last-epoch-mean finalization
  * batch size B and peak step alpha0 set by the paper's scaling laws

Usage:
    em = EMeans(k=1024, nepochs=3)
    centroids, labels = em.fit(x)          # x: (n, d) tensor
    test_labels = assign(x_test, centroids)

Run `python emeans.py` for a comparison against Lloyd's k-means.
"""

import math
import time

import torch


def scaling_laws(n: int, k: int, nepochs: float) -> tuple[int, float]:
    """Batch size B and peak step alpha0 from the problem size (paper eqs. 6-7)."""
    B = 0.035 * k**0.13 * n**0.88 * nepochs**0.81
    if n > 2_000_000:          # the paper divides the batch law by 8 at this scale
        B /= 8
    B = 2 ** round(math.log2(max(B, 1.0)))      # nearest power of two
    B = int(min(B, n))                           # capped at n
    alpha0 = min(0.824 * k**-0.46 * B**0.90 * n**-0.45 * nepochs**-0.11, 1.0)
    return B, alpha0


@torch.no_grad()
def assign(x: torch.Tensor, centroids: torch.Tensor, max_elems: int = 1 << 25) -> torch.Tensor:
    """Index of the nearest centroid for each row of x (exact, chunked)."""
    dev, dt = centroids.device, centroids.dtype
    cn = (centroids * centroids).sum(1)[None]            # (1, k)
    step = max(1, max_elems // centroids.shape[0])
    out = torch.empty(x.shape[0], dtype=torch.long, device=dev)
    for i in range(0, x.shape[0], step):
        xb = x[i:i + step].to(dev, dt)
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; ||x||^2 does not change the argmin
        out[i:i + step] = torch.addmm(cn, xb, centroids.T, alpha=-2.0).argmin(1)
    return out


@torch.no_grad()
def mse(x: torch.Tensor, centroids: torch.Tensor, chunk: int = 1 << 16) -> float:
    """Mean squared distance from each point to its nearest centroid."""
    total = 0.0
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk].to(centroids.device, centroids.dtype)
        total += (xb - centroids[assign(xb, centroids)]).pow(2).sum().double().item()
    return total / x.shape[0]


class EMeans:
    def __init__(
        self,
        k: int,
        nepochs: int = 5,
        batch_size: int | None = None,    # None -> scaling law
        alpha0: float | None = None,      # None -> scaling law
        floor: float = 0.1,               # cosine schedule ends at floor * alpha0
        tau: float = 0.1,                 # revive clusters with N_j < tau * mean(N)
        revive_every: int = 5,
        split_eps: float = 1.0 / 1024,    # Faiss-style perturbation when splitting
        seed: int = 0,
        device: str | torch.device | None = None,
        verbose: bool = False,
    ):
        self.k, self.nepochs = k, nepochs
        self.batch_size, self.alpha0 = batch_size, alpha0
        self.floor, self.tau = floor, tau
        self.revive_every, self.split_eps = revive_every, split_eps
        self.seed, self.verbose = seed, verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.centroids = None

    @torch.no_grad()
    def _revive(self, C, S, N, gen) -> int:
        """Replace dead clusters by splitting donors sampled proportionally to size."""
        dead = (N < self.tau * N.mean()).nonzero().squeeze(1)
        if dead.numel() == 0:
            return 0
        w = N.clone()
        w[dead] = 0
        m = min(dead.numel(), int((w > 0).sum()))
        if m == 0:
            return 0
        dead = dead[:m]
        donor = torch.multinomial(w.double().cpu(), m, replacement=False, generator=gen).to(N.device)
        # Split: both halves start at the donor centroid, nudged in opposite directions.
        sign = torch.ones(C.shape[1], device=C.device, dtype=C.dtype)
        sign[1::2] = -1
        sign *= self.split_eps
        c, half = C[donor], N[donor] / 2
        C[dead], C[donor] = c * (1 + sign), c * (1 - sign)
        N[dead], N[donor] = half, half
        S[dead], S[donor] = C[dead] * half[:, None], C[donor] * half[:, None]
        return m

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Train on x (n, d). Returns (centroids (k, d), last-epoch labels (n,))."""
        n, d = x.shape
        k, T, dev, dt = self.k, self.nepochs, self.device, torch.float32
        assert k <= n, "need at least k points"

        law_B, _ = scaling_laws(n, k, T)
        B = min(self.batch_size or law_B, n)
        a0 = self.alpha0 if self.alpha0 is not None else min(
            0.824 * k**-0.46 * B**0.90 * n**-0.45 * T**-0.11, 1.0)
        nb = max(1, n // B)            # batches per epoch
        M = T * nb                     # total updates
        self.batch_size_, self.alpha0_ = B, a0
        if self.verbose:
            print(f"[e-means] n={n} k={k} d={d} T={T}  B={B}  alpha0={a0:.4f}  updates={M}")

        gen = torch.Generator().manual_seed(self.seed)

        # Random init. N starts tiny so the init carries almost no weight.
        C = x[torch.randperm(n, generator=gen)[:k].to(x.device)].to(dev, dt).clone()
        N = torch.full((k,), 1e-10, device=dev, dtype=dt)
        S = C * N[:, None]

        labels = torch.empty(n, dtype=torch.long, device=dev)
        t = 0
        for epoch in range(T):
            last = epoch == T - 1
            if last:
                fin_S = torch.zeros(k, d, device=dev, dtype=dt)
                fin_N = torch.zeros(k, device=dev, dtype=dt)
            perm = torch.randperm(n, generator=gen)
            for b in range(nb):
                # the last batch of the epoch absorbs the remainder
                idx = perm[b * B: (b + 1) * B if b < nb - 1 else n]
                alpha = a0 * (self.floor + (1 - self.floor) * 0.5 * (1 + math.cos(math.pi * t / M)))
                gamma = 1.0 - alpha
                t += 1

                xb = x[idx.to(x.device)].to(dev, dt)
                a = assign(xb, C)
                nj = torch.bincount(a, minlength=k).to(dt)
                sj = torch.zeros(k, d, device=dev, dtype=dt).index_add_(0, a, xb)

                # EMA update of counts and sums; centroid = S / N
                N.mul_(gamma).add_(nj, alpha=alpha)
                S.mul_(gamma).add_(sj, alpha=alpha)
                hit = nj > 0           # S/N is unchanged for clusters with no points
                C[hit] = S[hit] / N[hit, None]

                if last:
                    labels[idx.to(dev)] = a
                    fin_N += nj
                    fin_S += sj

                if t % self.revive_every == 0:
                    r = self._revive(C, S, N, gen)
                    if self.verbose and r:
                        print(f"[e-means] update {t}/{M}: revived {r} clusters")

        # Finalization: exact mean of the points assigned during the last epoch.
        ok = fin_N > 0
        C[ok] = fin_S[ok] / fin_N[ok, None]
        self.centroids = C
        return C, labels


@torch.no_grad()
def lloyd(x: torch.Tensor, k: int, niter: int = 25, seed: int = 0) -> torch.Tensor:
    """Plain full-batch Lloyd k-means with random init, as a baseline."""
    gen = torch.Generator().manual_seed(seed)
    C = x[torch.randperm(x.shape[0], generator=gen)[:k]].clone().float()
    for _ in range(niter):
        a = assign(x, C)
        cnt = torch.bincount(a, minlength=k).float()
        s = torch.zeros_like(C).index_add_(0, a, x.float())
        ok = cnt > 0
        C[ok] = s[ok] / cnt[ok, None]
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    n, n_test, d, k = 200_000, 20_000, 32, 1024
    # Synthetic data: mixture of 300 Gaussians with uneven weights and scales.
    centers = torch.randn(300, d) * 3
    scales = torch.rand(300) * 1.5 + 0.2
    comp = torch.multinomial(torch.rand(300) ** 3, n + n_test, replacement=True)
    data = centers[comp] + torch.randn(n + n_test, d) * scales[comp, None]
    x, x_test = data[:n], data[n:]

    t0 = time.time()
    C_l = lloyd(x, k, niter=25)
    t_l = time.time() - t0
    base_tr, base_te = mse(x, C_l), mse(x_test, C_l)
    print(f"Lloyd   25 epochs: train {base_tr:.4f}  test {base_te:.4f}  ({t_l:.1f}s)")

    for T in (1, 2, 3, 5):
        em = EMeans(k, nepochs=T)
        t0 = time.time()
        C_e, _ = em.fit(x)
        t_e = time.time() - t0
        tr, te = mse(x, C_e), mse(x_test, C_e)
        print(f"e-means {T:2d} epochs: train {tr:.4f} ({100 * (tr / base_tr - 1):+.2f}%)  "
              f"test {te:.4f} ({100 * (te / base_te - 1):+.2f}%)  "
              f"({t_e:.1f}s, B={em.batch_size_}, alpha0={em.alpha0_:.3f})")
