"""The formula hunt: scan a family of polynomial continued fractions for identities with pi,
with the same search run against a random constant as the null control.

Pipeline for one family:

1. enumerate all (a, b) coefficient vectors of the family;
2. float64 scan to depth 2000; keep fractions that clearly converge (>= 7 digits);
3. match each limit against the Mobius table (p0 + p1 c)/(q0 + q1 c), |coefs| <= M,
   for c = pi **and** for c = a seeded random real (the control);
4. verify every match at 600-bit precision to depth 16,000 (``gcf.verify``).

Any "verified" hit against the random constant would mean the judge is broken. A real
identity keeps converging to its target, while a coincidence stalls.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import gcf
from .sources import data_dir

KNOWN = {
    ((1, 2), (0, 0, 1)): "classical: 4/π = 1 + 1²/(3 + 2²/(5 + 3²/(7 + …)))",
    ((6,), (1, -4, 4)): "classical (Euler): π + 3 = 6 + 1²/(6 + 3²/(6 + 5²/(6 + …)))",
    ((2,), (1, -4, 4)): "Brouncker: 4/π + 1 = 2 + 1²/(2 + 3²/(2 + 5²/(2 + …)))",
}

PRESETS = {
    "classic": {"name": "classic: a(n) = a0 + a1 n, b(n) quadratic; |a0| <= 8, others <= 5",
                "a": [[-8, 8], [0, 4]], "b": [[-5, 5], [-5, 5], [-5, 5]]},
    "small": {"name": "small: a(n) linear, b(n) quadratic, |coef| <= 3",
              "a": [[-3, 3], [0, 3]], "b": [[-3, 3], [-3, 3], [-3, 3]]},
}

STATUS_RANK = {"verified": 2, "probable": 1, "inconclusive": 0, "rejected": -1}


def hunts_dir() -> Path:
    d = data_dir() / "hunts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def hunt_id(family: gcf.Family, constant: str, depth: int, M: int) -> str:
    h = hashlib.sha1(json.dumps([family.a, family.b, constant, depth, M]).encode()).hexdigest()[:8]
    return f"{constant}_{h}"


def run_hunt(family: gcf.Family, constant: str = "pi", depth: int = 2000, M: int = 6,
             null_seed: int = 7, max_verify: int = 400, save: bool = True, log=print) -> dict:
    t0 = time.time()
    A, B = family.enumerate()
    log(f"family {family.name or family.to_dict()}: {len(A):,} continued fractions")
    xs, hs, terms = [], [], []
    for s in range(0, len(A), 50_000):
        x, h, t = gcf.scan(A[s:s + 50_000], B[s:s + 50_000], depth)
        xs.append(x), hs.append(h), terms.append(t)
    x, h, term = np.concatenate(xs), np.concatenate(hs), np.concatenate(terms)
    with np.errstate(all="ignore"):
        scale = np.maximum(1.0, np.abs(x))
        diff = np.abs(x - h)
        conv = np.isfinite(x) & np.isfinite(h) & ~term & (diff < 1e-6 * scale) & (np.abs(x) < 1e6)
        digits = np.where(conv, np.clip(-np.log10(diff / scale + 1e-17), 0, 16), 0.0)
        tol = 2 * diff + 1e-13 * scale
    eligible = conv & (tol < 1e-7 * scale)
    log(f"  converged: {conv.sum():,}; precise enough to match: {eligible.sum():,}; "
        f"scan {time.time() - t0:.1f}s")

    sides = {}
    for side, (cname, seed) in {"target": (constant, None), "null": ("random", null_seed)}.items():
        import gmpy2
        with gmpy2.context(gmpy2.get_context(), precision=80):
            cval = float(gcf.constant_mpfr(cname, seed))
        table = gcf.mobius_table(cval, M)
        idx = np.nonzero(eligible)[0]
        pairs = gcf.match(table, x[idx], tol[idx])
        pairs = [(int(idx[i]), j) for i, j in pairs]
        pairs.sort(key=lambda p: -digits[p[0]])
        hits = []
        for i, j in pairs[:max_verify]:
            a, b = tuple(int(v) for v in A[i]), tuple(int(v) for v in B[i])
            v = gcf.verify(a, b, table.coefs[j], cname, seed)
            status = v.status
            if status == "rejected" and v.rate.startswith(("power", "exponential")):
                status = "inconclusive"
            ca, cb, sc = gcf.canonical(a, b)
            ca, cb = _trim(ca), _trim(cb)
            hits.append({
                "a": list(a), "b": list(b), "a_str": gcf.poly_str(a), "b_str": gcf.poly_str(b),
                "gcf": gcf.gcf_str(a, b), "coefs": [int(c) for c in table.coefs[j]],
                "relation": gcf.mobius_str(table.coefs[j], "π" if cname == "pi" else ("c" if cname == "random" else cname)),
                "status": status, "digits": round(v.digits, 1), "rate": v.rate,
                "scan_digits": round(float(digits[i]), 1),
                "canonical": [list(ca), list(cb)], "scaled_by": sc,
                "orbit": gcf.orbit_key(a, b),
                "core": gcf.family_key(a, b),
                "known": KNOWN.get((ca, cb), ""),
            })
        hits.sort(key=lambda r: (-STATUS_RANK[r["status"]], -r["digits"]))
        sides[side] = {"constant": cname, "seed": seed, "table_size": len(table.values),
                       "matches": len(pairs), "verified_checked": min(len(pairs), max_verify),
                       "count": {s: sum(r["status"] == s for r in hits) for s in STATUS_RANK},
                       "hits": hits}
        c = sides[side]["count"]
        log(f"  vs {cname:<6}: {len(pairs):,} raw matches -> verified {c['verified']}, "
            f"probable {c['probable']}, inconclusive {c['inconclusive']}, rejected {c['rejected']}")

    hid = hunt_id(family, constant, depth, M)
    res = {
        "id": hid, "family": family.to_dict(), "constant": constant, "depth": depth, "mobius_M": M,
        "n_candidates": int(len(A)), "n_converged": int(conv.sum()), "n_eligible": int(eligible.sum()),
        "target": sides["target"], "null": sides["null"],
        "distinct_relations": distinct(sides["target"]["hits"]),
        "orbits": sorted({r["orbit"] for r in distinct(sides["target"]["hits"])}),
        "cores": sorted({r["core"] for r in distinct(sides["target"]["hits"])}),
        "elapsed_s": round(time.time() - t0, 1), "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if save:
        d = hunts_dir() / hid
        d.mkdir(parents=True, exist_ok=True)
        axes = landscape_axes(A, B)
        res["landscape_axes"] = [axis_label(axes[0]), axis_label(axes[1])]
        for side in ("target", "null"):
            landscape(A, B, digits, conv, sides[side]["hits"], d / f"landscape_{side}.png", axes)
        (d / "hunt.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"  distinct verified/probable fractions: {len(res['distinct_relations'])} (up to scaling), "
        f"{len(res['orbits'])} up to scaling and index shift, {len(res['cores'])} after removing "
        f"polynomial equivalences; "
        f"{res['elapsed_s']}s")
    return res


def _trim(c):
    c = list(c)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    return tuple(c)


def distinct(hits: list[dict]) -> list[dict]:
    """One entry per canonical (a, b) among verified/probable hits."""
    seen, out = set(), []
    for r in hits:
        if r["status"] not in ("verified", "probable"):
            continue
        key = json.dumps(r["canonical"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def landscape_axes(A, B) -> tuple[tuple[str, int], tuple[str, int]]:
    """The two coefficients with the most distinct values (one from a, one from b if possible)."""
    cand = [(len(np.unique(A[:, k])), "a", k) for k in range(A.shape[1])] +            [(len(np.unique(B[:, k])), "b", k) for k in range(B.shape[1])]
    cand = [c for c in cand if c[0] > 1]
    cand.sort(reverse=True)
    best_a = next((c for c in cand if c[1] == "a"), None)
    best_b = next((c for c in cand if c[1] == "b"), None)
    if best_a and best_b:
        x, y = best_a, best_b
    else:
        x, y = (cand + cand)[:2] if len(cand) > 1 else (cand[0], cand[0])
    return (x[1], x[2]), (y[1], y[2])


def landscape(A, B, digits, conv, hits, path: Path, axes=None) -> None:
    """Map of the family on two coefficients (x, y): grey-blue brightness = how well the
    fractions in that cell converge (averaged over the other coefficients), gold = at least
    one verified identity in that cell (brighter = more)."""
    (xs_, xk), (ys_, yk) = axes or landscape_axes(A, B)
    xa = (A if xs_ == "a" else B)[:, xk]
    yb = (A if ys_ == "a" else B)[:, yk]
    xs, ys = np.unique(xa), np.unique(yb)
    xi, yi = np.searchsorted(xs, xa), np.searchsorted(ys, yb)
    W, H = len(xs), len(ys)
    tot = np.bincount(yi * W + xi, weights=digits, minlength=W * H)
    cnt = np.bincount(yi * W + xi, minlength=W * H)
    g = (tot / np.maximum(cnt, 1) / 16).reshape(H, W)
    rgb = np.stack([g * 70 + 20, g * 90 + 22, g * 140 + 30], -1)
    hitmap = np.zeros(H * W)
    for r in hits:
        if r["status"] in ("verified", "probable"):
            vx = (r["a"] if xs_ == "a" else r["b"])[xk]
            vy = (r["a"] if ys_ == "a" else r["b"])[yk]
            hitmap[int(np.searchsorted(ys, vy) * W + np.searchsorted(xs, vx))] += 1 if r["status"] == "verified" else 0.5
    hm = hitmap.reshape(H, W)
    lit = hm > 0
    inten = np.clip(0.55 + 0.45 * np.log1p(hm) / np.log1p(max(hm.max(), 1)), 0, 1)
    rgb[lit] = np.stack([255 * inten, 196 * inten, 60 * inten], -1)[lit]
    cell = max(4, 420 // max(W, H))
    img = np.kron(rgb[::-1].astype(np.uint8), np.ones((cell, cell, 1), dtype=np.uint8))
    Image.fromarray(img).save(path)


def axis_label(ax) -> str:
    side, k = ax
    return f"coefficient of n^{k} in {side}(n)" if k else f"constant term of {side}(n)"


def list_hunts() -> list[dict]:
    out = []
    for f in sorted(hunts_dir().glob("*/hunt.json")):
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return sorted(out, key=lambda r: r["saved"])


def migrate_cores() -> int:
    """Add the 'core' key (polynomial-equivalence class) to hunts saved before it existed."""
    n = 0
    for f in hunts_dir().glob("*/hunt.json"):
        h = json.loads(f.read_text(encoding="utf-8"))
        for side in ("target", "null"):
            for r in h[side]["hits"]:
                r["core"] = gcf.family_key(tuple(r["a"]), tuple(r["b"]))
        for r in h["distinct_relations"]:
            r["core"] = gcf.family_key(tuple(r["a"]), tuple(r["b"]))
        h["cores"] = sorted({r["core"] for r in h["distinct_relations"]})
        f.write_text(json.dumps(h, indent=1, ensure_ascii=False), encoding="utf-8")
        n += 1
    return n
