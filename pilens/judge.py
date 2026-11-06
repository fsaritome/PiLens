"""The judge: does a lens show anything in pi that it doesn't show in random controls?

For one lens and one source sequence of length n:

1. **Metrics.** The lens's own features plus generic image metrics (compressed
   size, pixel entropy, 2-D FFT power-spectrum slope / peakiness / flatness).
2. **Controls.** The same lens on >= 200 random sequences drawn from the
   source's null (``sources.make_control``), same n and base. Each metric gets a
   z-score against the controls and an empirical two-sided p-value.
3. **One p-value per lens (max-T).** T = max_j |z_j| over metrics; the
   empirical p of T among the controls' own T values. This corrects for the
   number of metrics inside a lens exactly, without assuming independence.
4. **Classifier.** Split pi into K chunks and one control sequence into K
   chunks; a small linear (ridge / regularised-LDA) classifier on the chunk
   metrics tries to tell them apart (5-fold CV accuracy). Permutation p-value
   from shuffled labels.
5. **Lens p** = min(1, 2 * min(p_maxT, p_classifier)) (Bonferroni over the two
   tests). Across lenses in a run, Benjamini-Hochberg q-values (``bh``).

The smallest attainable p is 1/(n_controls+1), so with m lenses you need
roughly n_controls > 2*m/0.05 for anything to survive BH at q = 0.05.
"""

from __future__ import annotations

import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from .lenses import get_lens, load_plugins
from .sources import Seq, make_control

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def to_gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3:
        g = img[..., :3].astype(np.float64).mean(axis=2)
        return g / 255.0 if img.dtype == np.uint8 else g
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def _downsample(g: np.ndarray, max_side: int = 256) -> np.ndarray:
    f = int(np.ceil(max(g.shape) / max_side))
    if f <= 1:
        return g
    h, w = (g.shape[0] // f) * f, (g.shape[1] // f) * f
    return g[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def image_metrics(img: np.ndarray) -> dict[str, float]:
    g = to_gray(img)
    q8 = np.clip(_downsample(g, 384) * 255, 0, 255).astype(np.uint8)
    raw = q8.tobytes()
    out = {"img_compress_ratio": len(zlib.compress(raw, 4)) / max(len(raw), 1)}
    h = np.bincount((np.clip(g, 0, 1) * 63).astype(np.int64).ravel(), minlength=64).astype(float)
    p = h[h > 0] / h.sum()
    out["img_entropy_bits"] = float(-(p * np.log2(p)).sum())
    d = _downsample(g)
    if min(d.shape) >= 8 and d.std() > 0:
        P = np.abs(np.fft.fftshift(np.fft.fft2(d - d.mean()))) ** 2
        cy, cx = np.array(P.shape) // 2
        yy, xx = np.indices(P.shape)
        r = np.hypot(yy - cy, xx - cx).astype(np.int64)
        radial = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
        rmax = min(cy, cx)
        f = np.arange(2, rmax)
        good = radial[f] > 0
        out["fft_slope"] = float(np.polyfit(np.log(f[good]), np.log(radial[f][good]), 1)[0]) \
            if good.sum() > 3 else 0.0
        Pn = P.copy()
        Pn[r <= 1] = 0
        med = np.median(Pn[r > 1])
        out["fft_peak_log10"] = float(np.log10(Pn.max() / med)) if med > 0 else 0.0
        v = P[r > 1] + 1e-12
        out["fft_flatness"] = float(np.exp(np.log(v).mean()) / v.mean())
    else:
        out.update(fft_slope=0.0, fft_peak_log10=0.0, fft_flatness=0.0)
    return out


def evaluate(lens_name: str, x: np.ndarray, base: int, params: dict | None = None):
    """Run a lens and return (image, metrics) where metrics are the tested numbers."""
    res = get_lens(lens_name)(x, base, **(params or {}))
    metrics = {k: v for k, v in res.features.items() if not k.startswith("_")}
    metrics.update(image_metrics(res.image))
    info = {k: v for k, v in res.features.items() if k.startswith("_")}
    return res.image, metrics, info


# ---------------------------------------------------------------------------
# controls (optionally in parallel)
# ---------------------------------------------------------------------------


def _control_job(args):
    lens_name, n, base, null, seed, params = args
    x = make_control(Seq("control", np.zeros(n, np.uint8), base, null), seed)
    _, m, _ = evaluate(lens_name, x, base, params)
    return m


def _pool_init():
    load_plugins()


_POOL: tuple[int, ProcessPoolExecutor] | None = None


def _map(fn, jobs_args, jobs: int):
    """Map in-process (jobs <= 1) or over a process pool that is reused between calls."""
    global _POOL
    if jobs <= 1:
        return [fn(a) for a in jobs_args]
    if _POOL is None or _POOL[0] != jobs:
        if _POOL is not None:
            _POOL[1].shutdown()
        _POOL = (jobs, ProcessPoolExecutor(max_workers=jobs, initializer=_pool_init))
    return list(_POOL[1].map(fn, jobs_args, chunksize=max(1, len(jobs_args) // (jobs * 4))))


def default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def bh(pvals) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def ridge_cv_permutation(X: np.ndarray, y: np.ndarray, n_perm: int, seed: int,
                         folds: int = 5, lam: float = 1.0) -> tuple[float, float]:
    """Cross-validated accuracy of a ridge (least-squares, ~regularised LDA) linear
    classifier, plus a label-permutation p-value.

    For fixed folds the test predictions are linear in the training labels
    (``pred = H @ y_train``), so all permutations cost one matrix product.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    # stratified fold assignment
    fold = np.empty(n, dtype=np.int64)
    for cls in np.unique(y):
        idx = rng.permutation(np.nonzero(y == cls)[0])
        fold[idx] = np.arange(len(idx)) % folds
    Y = np.empty((n, n_perm + 1))
    Y[:, 0] = np.where(y == 1, 1.0, -1.0)
    for j in range(1, n_perm + 1):
        Y[:, j] = rng.permutation(Y[:, 0])
    correct = np.zeros(n_perm + 1)
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        A = np.hstack([(X[tr] - mu) / sd, np.ones((tr.sum(), 1))])
        B = np.hstack([(X[te] - mu) / sd, np.ones((te.sum(), 1))])
        H = B @ np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T)
        pred = H @ Y[tr]
        correct += (np.sign(pred) == Y[te]).sum(axis=0)
    acc = correct / n
    p = (1 + (acc[1:] >= acc[0]).sum()) / (n_perm + 1)
    return float(acc[0]), float(p)


def _chunk_job(args):
    lens_name, x, base, params = args
    return evaluate(lens_name, x, base, params)[1]


def classify(lens_name: str, seq: Seq, params: dict, k_chunks: int, n_perm: int,
             seed: int, jobs: int = 1) -> tuple[float, float]:
    """Can a linear classifier tell chunks of ``seq`` from chunks of a control?"""
    n = len(seq)
    cl = n // k_chunks
    ctrl = make_control(seq, seed + 999_983)
    chunks = [(lens_name, src[i * cl : (i + 1) * cl], seq.base, params)
              for src in (seq.symbols, ctrl) for i in range(k_chunks)]
    ms = _map(_chunk_job, chunks, jobs)
    names = sorted(ms[0])
    rows = [[m[k] for k in names] for m in ms]
    X = np.nan_to_num(np.array(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([1] * k_chunks + [0] * k_chunks)
    keep = X.std(axis=0) > 0
    if not keep.any():
        return 0.5, 1.0
    return ridge_cv_permutation(X[:, keep], y, n_perm, seed)


@dataclass
class MetricScore:
    name: str
    value: float
    ctrl_mean: float
    ctrl_sd: float
    z: float
    p: float


@dataclass
class JudgeResult:
    lens: str
    params: dict
    source: str
    base: int
    n: int
    n_controls: int
    metrics: list[MetricScore]
    p_maxT: float
    clf_acc: float | None
    clf_p: float | None
    p_lens: float
    info: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    q_lens: float | None = None  # filled in by BH across lenses

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        top = max(self.metrics, key=lambda m: abs(m.z) if np.isfinite(m.z) else -1)
        clf = f" clf_acc={self.clf_acc:.2f} (p={self.clf_p:.3f})" if self.clf_acc is not None else ""
        q = f" q={self.q_lens:.3f}" if self.q_lens is not None else ""
        return (f"{self.lens:<11} {self.source} b{self.base} n={self.n}: p_maxT={self.p_maxT:.3f}"
                f"{clf} p_lens={self.p_lens:.3f}{q}  top: {top.name} z={top.z:+.2f}")


def judge(lens_name: str, seq: Seq, *, n_controls: int = 500, params: dict | None = None,
          seed: int = 0, classifier: bool = True, k_chunks: int = 32, n_perm: int | None = None,
          jobs: int = 1, return_images: bool = False):
    """Score one lens on one sequence against random controls. See module docstring."""
    t0 = time.time()
    params = dict(params or {})
    lens = get_lens(lens_name)
    params_full = {**lens.defaults, **params}
    img_pi, m_pi, info = evaluate(lens_name, seq.symbols, seq.base, params)
    names = list(m_pi)
    args = [(lens_name, len(seq), seq.base, seq.null, seed * 1_000_003 + i + 1, params)
            for i in range(n_controls)]
    ctrl = _map(_control_job, args, jobs)
    C = np.array([[c[k] for k in names] for c in ctrl], dtype=float)
    x = np.array([m_pi[k] for k in names], dtype=float)

    mu = np.nanmean(C, axis=0)
    sd = np.nanstd(C, axis=0, ddof=1)
    ok = (sd > 0) & np.isfinite(x)
    z = np.where(ok, (x - mu) / np.where(sd > 0, sd, 1), 0.0)
    Zc = np.where(ok, (C - mu) / np.where(sd > 0, sd, 1), 0.0)
    Zc = np.nan_to_num(Zc, nan=0.0)
    p_metric = (1 + (np.abs(Zc) >= np.abs(z)).sum(axis=0)) / (n_controls + 1)
    T, Tc = np.abs(z).max(), np.abs(Zc).max(axis=1)
    p_maxT = float((1 + (Tc >= T).sum()) / (n_controls + 1))

    metrics = [MetricScore(k, float(x[j]), float(mu[j]), float(sd[j]), float(z[j]), float(p_metric[j]))
               for j, k in enumerate(names)]
    clf_acc = clf_p = None
    p_lens = p_maxT
    if classifier:
        clf_acc, clf_p = classify(lens_name, seq, params, k_chunks, n_perm or n_controls, seed, jobs)
        p_lens = min(1.0, 2 * min(p_maxT, clf_p))
    res = JudgeResult(lens_name, params_full, seq.name, seq.base, len(seq), n_controls, metrics,
                      p_maxT, clf_acc, clf_p, p_lens, info, time.time() - t0)
    if return_images:
        img_ctrl, _, _ = evaluate(lens_name, make_control(seq, seed * 1_000_003 + 1), seq.base, params)
        return res, img_pi, img_ctrl
    return res


def judge_many(lens_names, seq: Seq, **kw) -> list[JudgeResult]:
    """Judge several lenses on one sequence and fill in BH q-values across them."""
    results = [judge(name, seq, **kw) for name in lens_names]
    for r, q in zip(results, bh([r.p_lens for r in results])):
        r.q_lens = float(q)
    return results
