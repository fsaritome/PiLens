"""Example plug-in lens (the shape phase-2 generated lenses will take).

Any ``*.py`` here is imported by ``pilens.lenses.load_plugins()``; each
``@lens`` it defines becomes available to the CLI, the judge and the gallery.
"""

import numpy as np

from pilens.lenses import LensResult, lens


@lens("spiral", "Square (Ulam-style) spiral: symbol i sits at the i-th cell of a spiral out from the centre.")
def spiral(x, base, *, side=255):
    n = min(len(x), side * side)
    # coordinates of the square spiral, vectorised
    k = np.arange(n)
    r = np.ceil((np.sqrt(k + 1) - 1) / 2).astype(np.int64)  # ring index
    t = 2 * r + 1
    m = t ** 2
    t = t - 1
    xs = np.empty(n, dtype=np.int64)
    ys = np.empty(n, dtype=np.int64)
    d = m - 1 - k  # distance back from the ring's last cell
    s1 = d < t
    s2 = (d >= t) & (d < 2 * t)
    s3 = (d >= 2 * t) & (d < 3 * t)
    s4 = d >= 3 * t
    xs[s1], ys[s1] = r[s1] - d[s1], r[s1]
    xs[s2], ys[s2] = -r[s2], r[s2] - (d[s2] - t[s2])
    xs[s3], ys[s3] = -r[s3] + (d[s3] - 2 * t[s3]), -r[s3]
    xs[s4], ys[s4] = r[s4], -r[s4] + (d[s4] - 3 * t[s4])
    c = side // 2
    img = np.zeros((side, side), dtype=np.float32)
    img[ys + c, xs + c] = x[:n] / max(base - 1, 1)
    g = np.full((side, side), -1, dtype=np.int64)
    g[ys + c, xs + c] = x[:n]
    filled = g >= 0
    v = filled[1:] & filled[:-1]
    h = filled[:, 1:] & filled[:, :-1]
    return LensResult(img, {
        "vert_neighbour_match": float((g[1:] == g[:-1])[v].mean()),
        "horiz_neighbour_match": float((g[:, 1:] == g[:, :-1])[h].mean()),
        "diag_match": float((g[1:, 1:] == g[:-1, :-1])[filled[1:, 1:] & filled[:-1, :-1]].mean()),
    })
