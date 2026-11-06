"""Lenses: ways of turning a symbol sequence into a picture plus numbers.

The contract (also what phase-2 generated lenses must follow)::

    @lens("name", "one-line description")
    def my_lens(x: np.ndarray, base: int, *, param1=default1, ...) -> LensResult:
        ...

* ``x`` is a 1-D ``int64`` array with values in ``range(base)``; ``base`` is 2..36.
* Keyword parameters must all have defaults (they are read from the signature
  and shown in the gallery). Lenses must be deterministic and must not do I/O.
* Return a :class:`LensResult`: ``image`` is either a float ``HxW`` array in
  [0, 1] or a ``uint8`` ``HxWx3`` RGB array (keep it <= ~1024 px a side);
  ``features`` is a flat dict of finite floats with stable names. Names that
  start with ``_`` are informational only and are not tested by the judge.
* Feature values only need to be comparable between pi and the random controls
  of the same length and base, which the judge computes with the same lens.

Extra lenses are discovered from ``*.py`` files in ``lenses_ext/`` (or the
directory in ``$PILENS_LENS_DIR``); importing such a file registers its lenses.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage


@dataclass
class LensResult:
    image: np.ndarray
    features: dict[str, float]


@dataclass
class Lens:
    name: str
    fn: Callable[..., LensResult]
    description: str = ""
    defaults: dict = field(default_factory=dict)
    origin: str = "builtin"

    def __call__(self, x, base: int, **params) -> LensResult:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"lens {self.name!r} has no parameter(s) {sorted(unknown)}; "
                            f"known: {sorted(self.defaults)}")
        x = np.asarray(x, dtype=np.int64)
        res = self.fn(x, base, **params)
        res.features = {k: float(v) for k, v in res.features.items()}
        return res


LENSES: dict[str, Lens] = {}


def lens(name: str, description: str = ""):
    """Decorator that registers a lens function."""

    def deco(fn):
        sig = inspect.signature(fn)
        defaults = {k: p.default for k, p in sig.parameters.items()
                    if p.default is not inspect.Parameter.empty}
        mod = getattr(fn, "__module__", "")
        LENSES[name] = Lens(name, fn, description or (fn.__doc__ or "").strip().split("\n")[0],
                            defaults, "builtin" if mod == __name__ else mod)
        return fn

    return deco


def load_plugins(directory: str | os.PathLike | None = None) -> list[str]:
    """Import every ``*.py`` in the plugin directory; returns names of new lenses."""
    d = Path(directory or os.environ.get("PILENS_LENS_DIR",
                                         Path(__file__).resolve().parent.parent / "lenses_ext"))
    before = set(LENSES)
    if d.is_dir():
        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(f"pilens_ext_{f.stem}", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    return sorted(set(LENSES) - before)


def get_lens(name: str) -> Lens:
    if name not in LENSES:
        load_plugins()
    if name not in LENSES:
        raise KeyError(f"unknown lens {name!r}; known: {sorted(LENSES)}")
    return LENSES[name]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _chi2_z(counts: np.ndarray, expected) -> float:
    """Chi-square statistic standardised to ~N(0,1) under the null: (X2 - df)/sqrt(2 df)."""
    counts = counts.astype(np.float64)
    expected = np.broadcast_to(np.asarray(expected, dtype=np.float64), counts.shape).ravel()
    counts = counts.ravel()
    ok = expected > 0
    x2 = float((((counts - expected) ** 2)[ok] / expected[ok]).sum())
    df = max(int(ok.sum()) - 1, 1)
    return (x2 - df) / np.sqrt(2 * df)


def _entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def _raster_points(xy: np.ndarray, px: int, weights=None) -> np.ndarray:
    """Accumulate 2-D points into a px x px log-density image (aspect preserved)."""
    lo = xy.min(axis=0)
    span = float((xy.max(axis=0) - lo).max()) or 1.0
    ij = ((xy - lo) / span * (px - 1)).round().astype(np.int64)
    img = np.bincount(ij[:, 1] * px + ij[:, 0], weights=weights, minlength=px * px)
    img = np.log1p(img.reshape(px, px)[::-1])
    m = img.max()
    return (img / m if m > 0 else img).astype(np.float32)


# ---------------------------------------------------------------------------
# built-in lenses
# ---------------------------------------------------------------------------

def walk_scaling(pos: np.ndarray, min_window: int = 16) -> float:
    """Scaling exponent nu in R(L) ~ L^nu, from non-overlapping windows of length 2^k.

    Freely jointed / random walk: 0.5. 2-D self-avoiding walk: 0.75 (3-D: 0.588).
    Ballistic (biased) walk: 1. Compact (folded) 2-D globule: 0.5, 3-D: 1/3.
    """
    n = len(pos) - 1
    Ls, Rs = [], []
    L = min_window
    while L <= n // 8:
        d = pos[L::L] - pos[: -L : L][: len(pos[L::L])]
        Ls.append(L)
        Rs.append(np.sqrt((d ** 2).sum(axis=1).mean()))
        L *= 2
    if len(Ls) < 2:
        return float("nan")
    return float(np.polyfit(np.log(Ls), np.log(Rs), 1)[0])


@lens("walk", "Turtle walk: each symbol d is a unit step at angle 2*pi*d/base.")
def walk(x, base, *, px=512):
    ang = 2 * np.pi * x / base
    steps = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    pos = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
    n = len(x)
    mids = (pos[:-1] + pos[1:]) / 2  # half steps keep the drawn path continuous
    img = _raster_points(np.vstack([pos, mids]), px)
    cen = pos - pos.mean(axis=0)
    gyr = cen.T @ cen / len(pos)
    lam = np.sort(np.linalg.eigvalsh(gyr))
    cells = np.round(pos).astype(np.int64)
    key = cells[:, 0] * 4_000_003 + cells[:, 1]
    revisit = 1.0 - len(np.unique(key)) / len(key)
    return LensResult(img, {
        "nu": walk_scaling(pos),
        "end_to_end_norm": float(np.hypot(*pos[-1]) / np.sqrt(n)),
        "rg_norm": float(np.sqrt(lam.sum()) / np.sqrt(n)),
        "asphericity": float(((lam[1] - lam[0]) / lam.sum()) ** 2),
        "self_intersection": revisit,
    })


@lens("grid", "Colour grid: symbols laid out in rows of `width`, plus a period scan over widths.")
def grid(x, base, *, width=256, max_rows=512, fold_max=64):
    rows = min(len(x) // width, max_rows)
    g = x[: rows * width].reshape(rows, width)
    img = (g / max(base - 1, 1)).astype(np.float32)
    n = len(x)
    idx = np.arange(n)
    best_z, best_w = -np.inf, 0
    for w in range(2, fold_max + 1):
        c = np.bincount((idx % w) * base + x, minlength=w * base).reshape(w, base)
        z = _chi2_z(c, c.sum(axis=1, keepdims=True) / base)
        if z > best_z:
            best_z, best_w = z, w
    return LensResult(img, {
        "vert_match": float((g[1:] == g[:-1]).mean()),
        "horiz_match": float((g[:, 1:] == g[:, :-1]).mean()),
        "diag_match": float((g[1:, 1:] == g[:-1, :-1]).mean()),
        "fold_max_z": float(best_z),
        "_fold_best_width": float(best_w),
    })


@lens("chord", "Chord diagram: consecutive symbols joined by curves across a circle (Krzywinski-style).")
def chord(x, base, *, max_chords=2000, px=512, samples=256):
    m = min(len(x) - 1, max_chords)
    i = np.arange(m + 1)
    theta = 2 * np.pi * (x[: m + 1] + 0.08 + 0.84 * i / max(m, 1)) / base
    p = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    t = np.linspace(0, 1, samples)[:, None, None]
    curve = (1 - t) ** 2 * p[None, :-1] + t ** 2 * p[None, 1:]  # quadratic Bezier, control at centre
    pts = np.vstack([curve.reshape(-1, 2), [[-1.0, -1.0], [1.0, 1.0]]])
    img = _raster_points(pts, px)
    # whole-sequence transition statistics
    tr = np.bincount(x[:-1] * base + x[1:], minlength=base * base).reshape(base, base).astype(float)
    nt = tr.sum()
    exp_ind = tr.sum(1, keepdims=True) * tr.sum(0, keepdims=True) / nt
    pj = tr / nt
    pr, pc = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    nz = pj > 0
    mi = float((pj[nz] * np.log2(pj[nz] / (pr @ pc)[nz])).sum())
    step = np.abs(x[1:] - x[:-1])
    circ = np.minimum(step, base - step)
    return LensResult(img, {
        "transition_indep_z": _chi2_z(tr, exp_ind),
        "mutual_info_bits": mi,
        "mean_circular_step": float(circ.mean()),
    })


@lens("recurrence", "Recurrence plot: R[i,j] = 1 where the embedded symbols at i and j match.")
def recurrence(x, base, *, block=256, n_blocks=40, embed=2, max_lag=64):
    n = len(x)
    # embed: e_i = x_i*base^(embed-1) + ... + x_{i+embed-1}
    e = np.zeros(n - embed + 1, dtype=np.int64)
    for k in range(embed):
        e = e * base + x[k : n - embed + 1 + k]
    nb = max(1, min(n_blocks, len(e) // block))
    acc = np.zeros((block, block), dtype=np.float64)
    rr_list, det_list = [], []
    triu = np.triu(np.ones((block, block), dtype=bool), 1)
    for b in range(nb):
        s = e[b * block : (b + 1) * block]
        R = s[:, None] == s[None, :]
        acc += R
        Rt = R & triu
        prev = np.zeros_like(Rt)
        nxt = np.zeros_like(Rt)
        prev[1:, 1:] = Rt[:-1, :-1]
        nxt[:-1, :-1] = Rt[1:, 1:]
        nrec = Rt.sum()
        rr_list.append(nrec / triu.sum())
        det_list.append((Rt & (prev | nxt)).sum() / nrec if nrec else 0.0)
    img = acc / nb
    img = (img / img.max()).astype(np.float32) if img.max() > 0 else img.astype(np.float32)
    # lag profile on the whole sequence
    hist = np.bincount(x, minlength=base) / n
    p0 = float((hist ** 2).sum())
    zs = []
    for k in range(1, max_lag + 1):
        mk = (x[k:] == x[:-k]).mean()
        zs.append((mk - p0) / np.sqrt(p0 * (1 - p0) / (n - k)))
    zs = np.array(zs)
    return LensResult(img, {
        "recurrence_rate": float(np.mean(rr_list)),
        "determinism": float(np.mean(det_list)),
        "lag1_match_z": float(zs[0]),
        "lag_max_abs_z": float(np.abs(zs).max()),
        "_lag_best": float(np.abs(zs).argmax() + 1),
    })


@lens("pairs", "Pair/difference transform: deviation heatmaps of (x_i, x_i+k) for lags k=1..max_lag.")
def pairs(x, base, *, max_lag=8, cell=0):
    n = len(x)
    cell = cell or max(2, 80 // base)  # 0 = auto: keep panels ~80 px tall
    panels, zs = [], []
    for k in range(1, max_lag + 1):
        h = np.bincount(x[:-k] * base + x[k:], minlength=base * base).reshape(base, base)
        e = (n - k) / base ** 2
        zs.append(_chi2_z(h, e))
        dev = np.clip((h - e) / np.sqrt(e), -4, 4) / 8 + 0.5
        panels.append(np.kron(dev, np.ones((cell, cell))))
        panels.append(np.full((base * cell, 2), 1.0))
    img = np.hstack(panels[:-1]).astype(np.float32)
    d = (x[1:] - x[:-1]) % base
    dh = np.bincount(d, minlength=base)
    return LensResult(img, {
        "pair_lag1_z": float(zs[0]),
        "pair_max_z": float(np.max(zs)),
        "diff0_rate": float(dh[0] / len(d)),
        "diff_entropy_bits": _entropy_bits(dh),
        "symbol_entropy_bits": _entropy_bits(np.bincount(x, minlength=base)),
    })


def glyph_alphabet(base: int, size: int, style: str, seed: int) -> np.ndarray:
    """``base`` distinct size x size binary glyphs."""
    if style == "truchet":
        # diagonal tiles; orientation = d mod 2, thickness grows with d // 2
        g = np.zeros((base, size, size), dtype=bool)
        ii, jj = np.mgrid[0:size, 0:size]
        for d in range(base):
            w = 1 + (d // 2) % max(size // 2, 1)
            diag = np.abs(ii - jj) if d % 2 == 0 else np.abs(ii + jj - (size - 1))
            g[d] = diag < w
        return g
    rng = np.random.default_rng(seed * 1000 + base)
    seen, out = set(), []
    while len(out) < base:
        gl = rng.random((size, size)) < 0.5
        k = gl.tobytes()
        if k not in seen:
            seen.add(k)
            out.append(gl)
    return np.array(out)


@lens("glyph", "Glyph tiling: each symbol becomes a small black/white tile; tiles are laid out in a grid.")
def glyph(x, base, *, width=128, size=4, style="random", glyph_seed=0):
    rows = min(len(x) // width, width)
    alpha = glyph_alphabet(base, size, style, glyph_seed)
    tiles = alpha[x[: rows * width].reshape(rows, width)]  # rows, width, size, size
    img = tiles.transpose(0, 2, 1, 3).reshape(rows * size, width * size)
    lab, ncomp = ndimage.label(img)
    sizes = np.bincount(lab.ravel())[1:]
    fg = img.sum()
    return LensResult(img.astype(np.float32), {
        "fg_fraction": float(fg / img.size),
        "components_per_tile": float(ncomp / (rows * width)),
        "largest_component_frac": float(sizes.max() / fg) if ncomp else 0.0,
        "mean_component_size": float(sizes.mean()) if ncomp else 0.0,
    })
