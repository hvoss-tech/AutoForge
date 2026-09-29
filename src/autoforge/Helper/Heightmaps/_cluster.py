"""Clustering and colour helpers for the height-map initialisation, without
scikit-learn / scikit-image / SciPy: importing those cost ~0.6 s per run,
more than the clustering itself.

* ``rgb2lab``: skimage.color.rgb2lab's formula (sRGB, D65 2-degree white).
* ``kmeans``: Lloyd's k-means with k-means++ seeding, on the GPU when there
  is one; centroid sums are taken in a fixed order (sorted by label, float64
  prefix sums), so a run is reproducible.
* ``weighted_kmeans``: small weighted k-means in NumPy (k-means++ seeding).
* ``silhouette``: the mean silhouette coefficient (Euclidean).
"""
from typing import Optional

import numpy as np
import torch

_XYZ_FROM_RGB = np.array(
    [[0.412453, 0.357580, 0.180423],
     [0.212671, 0.715160, 0.072169],
     [0.019334, 0.119193, 0.950227]]
)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] (..., 3) -> CIELAB (..., 3), float64."""
    c = np.asarray(rgb, dtype=np.float64)
    lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = (lin @ _XYZ_FROM_RGB.T) / _WHITE_D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def cdist(a: np.ndarray, b: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """scipy.spatial.distance.cdist(a, b) (Euclidean only)."""
    assert metric == "euclidean"
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def _sq_dists(x: torch.Tensor, c: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """Squared distances [n, k] (clamped at 0)."""
    return (x.pow(2).sum(1, keepdim=True) - 2.0 * (x @ c.t()) + c2.view(1, -1)).clamp_(min=0.0)


def _assign(x: torch.Tensor, c: torch.Tensor, chunk: int) -> torch.Tensor:
    """Nearest centroid [N], chunked over x (|x|^2 is the same for every
    centroid, so |c|^2 - 2 x.c ranks them)."""
    c2 = c.pow(2).sum(1).view(1, -1)
    labels = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for lo in range(0, x.shape[0], chunk):
        xc = x[lo:lo + chunk]
        labels[lo:lo + chunk] = torch.addmm(c2.expand(xc.shape[0], -1), xc, c.t(), alpha=-2.0).argmin(dim=1)
    return labels


def _kmeanspp(x: torch.Tensor, k: int, gen: torch.Generator, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """k-means++ seeding with 2 + log(k) local trials per centre, as sklearn
    does it; on x's device without host syncs (``gen`` on that device)."""
    n = x.shape[0]
    w = torch.ones(n, dtype=x.dtype, device=x.device) if weights is None else weights.to(x.dtype)
    trials = 2 + int(np.log(k))
    centers = torch.empty(k, x.shape[1], dtype=x.dtype, device=x.device)
    first = torch.multinomial(w, 1, generator=gen)
    centers[0] = x[first[0]]
    d = (x - x[first]).pow(2).sum(1)
    for i in range(1, k):
        # A zero potential (every point on a centre) samples uniformly.
        p = d * w
        p = torch.where(p.sum() > 0, p, torch.ones_like(p))
        cand = torch.multinomial(p, trials, replacement=True, generator=gen)
        dc = torch.minimum(d.view(1, -1), (x.view(1, n, -1) - x[cand].view(trials, 1, -1)).pow(2).sum(-1))
        best = torch.argmin((dc * w.view(1, -1)).sum(1))
        centers[i] = x[cand[best]]
        d = dc[best]
    return centers


def _segment_sums(x: torch.Tensor, labels: torch.Tensor, k: int, weights: Optional[torch.Tensor] = None):
    """(per-label sums of x [k, D], per-label weight [k]) in a fixed order."""
    order = torch.argsort(labels, stable=True)
    xs = x[order].double()
    ws = (weights[order].double() if weights is not None else torch.ones(x.shape[0], dtype=torch.float64, device=x.device))
    ends = torch.cumsum(torch.bincount(labels, minlength=k), 0)
    # Rows are channels, so the scans run along the inner dimension (the
    # outer-dimension scan kernel is ~100x slower on an [N, 3] tensor).
    v = torch.cat([(xs * ws.view(-1, 1)).t(), ws.view(1, -1)], 0).contiguous()  # [D+1, N]
    cs = torch.cat([v.new_zeros(v.shape[0], 1), torch.cumsum(v, 1)], 1)[:, ends]  # [D+1, k]
    seg = torch.diff(cs, dim=1, prepend=cs.new_zeros(cs.shape[0], 1))
    return seg[:-1].t(), seg[-1]


def kmeans(
    pixels: np.ndarray,
    k: int,
    seed: int = 0,
    max_iter: int = 15,
    tol: float = 1e-4,
    init_size: int = 4096,
    device: Optional[torch.device] = None,
    chunk: int = 32768,
) -> tuple[np.ndarray, np.ndarray]:
    """k-means of ``pixels`` [N, D] -> (centroids [k, D] float64, labels [N]
    int64). Seeded by k-means++ on a random subsample of ``init_size`` pixels,
    then Lloyd iterations on all of them until the centres move less than
    ``tol`` times the data variance, at most ``max_iter`` of them (from
    k-means++ seeds, 10 already beat MiniBatchKMeans' inertia on the test
    images)."""
    if device is None:
        from autoforge.Helper.DeviceUtils import accelerator_device

        device = accelerator_device() or torch.device("cpu")
    dt = torch.float64 if device.type == "cpu" else torch.float32
    prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")  # exact distances, no TF32
    try:
        x = torch.as_tensor(np.ascontiguousarray(pixels), device=device).to(dt)
        n = x.shape[0]
        k = min(k, n)
        gen = torch.Generator(device=device).manual_seed(int(seed) % (2**63))
        sub = torch.randperm(n, generator=gen, device=device)[: min(n, max(init_size, 3 * k))]
        c = _kmeanspp(x[sub], k, gen)
        thr = tol * float(x.var(0).mean())
        for _ in range(max_iter):
            labels = _assign(x, c, chunk)
            sums, cnt = _segment_sums(x, labels, k)
            new = torch.where(cnt.view(-1, 1) > 0, sums / cnt.clamp(min=1).view(-1, 1), c.double()).to(dt)
            shift = float((new - c).pow(2).sum())
            c = new
            if shift <= thr:
                break
        labels = _assign(x, c, chunk)
        return c.double().cpu().numpy(), labels.cpu().numpy()
    finally:
        torch.set_float32_matmul_precision(prec)


def weighted_kmeans(
    X: np.ndarray, weights: np.ndarray, k: int, seed: int = 0, max_iter: int = 300, tol: float = 1e-4
) -> np.ndarray:
    """Weighted k-means of a few points (NumPy, float64) -> centroids [k, D]."""
    x = torch.as_tensor(np.asarray(X, dtype=np.float64))
    w = torch.as_tensor(np.asarray(weights, dtype=np.float64))
    k = min(k, x.shape[0])
    c = _kmeanspp(x, k, torch.Generator().manual_seed(int(seed)), weights=w)
    wm = w / w.sum()
    var = float((wm.view(-1, 1) * (x - (wm.view(-1, 1) * x).sum(0)).pow(2)).sum(0).mean())
    thr = tol * var
    for _ in range(max_iter):
        labels = torch.cdist(x, c).argmin(1)
        sums, ws = _segment_sums(x, labels, k, w)
        new = torch.where(ws.view(-1, 1) > 0, sums / ws.clamp(min=1e-300).view(-1, 1), c)
        shift = float((new - c).pow(2).sum())
        c = new
        if shift <= thr:
            break
    return c.numpy()


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (Euclidean); raises ValueError for fewer
    than 2 or more than n - 1 clusters, like sklearn."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    uniq, lab = np.unique(labels, return_inverse=True)
    n, nl = X.shape[0], len(uniq)
    if not 2 <= nl <= n - 1:
        raise ValueError(f"Number of labels is {nl}. Valid values are 2 to n_samples - 1 (inclusive)")
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))
    np.fill_diagonal(D, 0.0)
    counts = np.bincount(lab, minlength=nl).astype(np.float64)
    S = np.zeros((n, nl))
    np.add.at(S.T, lab, D)  # S[i, j] = sum of distances from i to cluster j
    own = counts[lab]
    a = S[np.arange(n), lab] / np.maximum(own - 1, 1)
    mean_other = S / counts[None, :]
    mean_other[np.arange(n), lab] = np.inf
    b = mean_other.min(1)
    s = (b - a) / np.maximum(a, b)
    s = np.where(own > 1, np.nan_to_num(s), 0.0)
    return float(s.mean())
