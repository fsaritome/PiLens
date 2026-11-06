"""Generalised continued fractions (GCFs) with polynomial terms: where pi's structure lives.

A GCF with polynomials a(n), b(n) is

    a(0) + b(1) / (a(1) + b(2) / (a(2) + b(3) / (a(3) + ...)))

Classical examples for pi:

    4/pi   = 1 + 1^2/(3 + 2^2/(5 + 3^2/(7 + ...)))      a(n) = 2n+1, b(n) = n^2
    pi + 3 = 6 + 1^2/(6 + 3^2/(6 + 5^2/(6 + ...)))      a(n) = 6,    b(n) = (2n-1)^2

The hunt (``pilens.hunt``) scans many (a, b) quickly in float64, matches each limit
against a table of simple Mobius transforms (p0 + p1*pi)/(q0 + q1*pi), and then
*verifies* candidates at high precision. A real identity keeps converging to its
target as the depth grows. A numerical coincidence stalls at a fixed distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import gmpy2
import numpy as np
from gmpy2 import mpfr, mpz

# ---------------------------------------------------------------------------
# polynomials and families
# ---------------------------------------------------------------------------


def poly_str(c: tuple[int, ...], var: str = "n") -> str:
    terms = []
    for k in range(len(c) - 1, -1, -1):
        v = c[k]
        if v == 0:
            continue
        mono = "" if k == 0 else (var if k == 1 else f"{var}^{k}")
        if mono and abs(v) == 1:
            s = ("-" if v < 0 else "+") + mono
        else:
            s = f"{v:+d}{mono}"
        terms.append(s)
    if not terms:
        return "0"
    out = "".join(terms)
    return out[1:] if out[0] == "+" else out


def poly_eval(c, n):
    return sum(ck * n ** k for k, ck in enumerate(c))


def gcf_str(a: tuple[int, ...], b: tuple[int, ...], terms: int = 3) -> str:
    """Human-readable nested fraction: 1 + 1/(3 + 4/(5 + 9/(7 + ...)))."""
    return f"{poly_eval(a, 0)} + " + _nest(a, b, 1, terms)


def _nest(a, b, n, terms):
    if n > terms:
        return "..."
    return f"{poly_eval(b, n)}/({poly_eval(a, n)} + {_nest(a, b, n + 1, terms)})"


@dataclass
class Family:
    """Ranges of integer polynomial coefficients: a[k] = (lo, hi) for the n^k coefficient."""

    a: list[tuple[int, int]]
    b: list[tuple[int, int]]
    name: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Family":
        return cls([tuple(r) for r in d["a"]], [tuple(r) for r in d["b"]], d.get("name", ""))

    def to_dict(self) -> dict:
        return {"name": self.name, "a": [list(r) for r in self.a], "b": [list(r) for r in self.b]}

    def size(self) -> int:
        return math.prod(hi - lo + 1 for lo, hi in self.a + self.b)

    def enumerate(self) -> tuple[np.ndarray, np.ndarray]:
        """All (a, b) coefficient vectors, removing the trivial sign symmetry a -> -a
        (which maps the limit L to -L) by requiring the leading nonzero a-coefficient > 0."""
        A = np.array(list(product(*[range(lo, hi + 1) for lo, hi in self.a])), dtype=np.int64)
        lead = np.zeros(len(A), dtype=np.int64)
        for k in range(A.shape[1] - 1, -1, -1):
            lead = np.where(lead == 0, A[:, k], lead)
        A = A[lead > 0]
        B = np.array(list(product(*[range(lo, hi + 1) for lo, hi in self.b])), dtype=np.int64)
        B = B[(B != 0).any(axis=1)]
        ia, ib = np.meshgrid(np.arange(len(A)), np.arange(len(B)), indexing="ij")
        return A[ia.ravel()], B[ib.ravel()]


# ---------------------------------------------------------------------------
# fast float64 scan of many GCFs at once
# ---------------------------------------------------------------------------


def scan(A: np.ndarray, B: np.ndarray, depth: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate many GCFs in parallel (float64, renormalised).

    Returns (x_depth, x_half, terminated): the convergents at ``depth`` and ``depth//2``,
    and a flag for b(n) == 0 at some n <= depth (finite, i.e. rational, fraction).
    """
    K = len(A)
    with np.errstate(all="ignore"):
        p_prev, q_prev = np.ones(K), np.zeros(K)
        p, q = poly_vals(A, 0).astype(float), np.ones(K)
        term = np.zeros(K, dtype=bool)
        half = None
        for n in range(1, depth + 1):
            an = poly_vals(A, n).astype(float)
            bn = poly_vals(B, n).astype(float)
            term |= bn == 0
            p, p_prev = an * p + bn * p_prev, p
            q, q_prev = an * q + bn * q_prev, q
            if n % 8 == 0:
                s = np.maximum.reduce([np.abs(p), np.abs(q), np.abs(p_prev), np.abs(q_prev)])
                s = np.where((s > 0) & np.isfinite(s), s, 1.0)
                p, q, p_prev, q_prev = p / s, q / s, p_prev / s, q_prev / s
            if n == depth // 2:
                half = p / q
        x = p / q
    return x, half, term


def poly_vals(C: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(len(C), dtype=np.int64)
    for k in range(C.shape[1] - 1, -1, -1):
        out = out * n + C[:, k]
    return out


# ---------------------------------------------------------------------------
# Mobius table of a constant
# ---------------------------------------------------------------------------


@dataclass
class MobiusTable:
    values: np.ndarray  # sorted
    coefs: np.ndarray  # (m, 4): p0, p1, q0, q1 with value = (p0 + p1 c)/(q0 + q1 c)


def mobius_table(c: float, M: int = 6) -> MobiusTable:
    r = np.arange(-M, M + 1)
    P0, P1, Q0, Q1 = [x.ravel() for x in np.meshgrid(r, r, r, r, indexing="ij")]
    ok = (P0 * Q1 - P1 * Q0 != 0) & ((P1 != 0) | (Q1 != 0))
    P0, P1, Q0, Q1 = P0[ok], P1[ok], Q0[ok], Q1[ok]
    den = Q0 + Q1 * c
    ok = np.abs(den) > 1e-9
    P0, P1, Q0, Q1 = P0[ok], P1[ok], Q0[ok], Q1[ok]
    vals = (P0 + P1 * c) / (Q0 + Q1 * c)
    # keep the simplest representative of each value (smallest max |coef|)
    cplx = np.max(np.abs(np.stack([P0, P1, Q0, Q1])), axis=0) * 100 + np.abs(np.stack([P0, P1, Q0, Q1])).sum(0)
    order = np.lexsort((cplx, np.round(vals, 12)))
    vals, coefs, rounded = vals[order], np.stack([P0, P1, Q0, Q1], 1)[order], np.round(vals[order], 12)
    first = np.ones(len(vals), dtype=bool)
    first[1:] = rounded[1:] != rounded[:-1]
    vals, coefs = vals[first], coefs[first]
    o = np.argsort(vals)
    return MobiusTable(vals[o], coefs[o])


def match(table: MobiusTable, x: np.ndarray, tol: np.ndarray) -> list[tuple[int, int]]:
    """Pairs (candidate index, table index) with |x - table| <= tol (closest entry only)."""
    out = []
    pos = np.searchsorted(table.values, x)
    for off in (-1, 0):
        j = np.clip(pos + off, 0, len(table.values) - 1)
        d = np.abs(table.values[j] - x)
        hit = np.nonzero(d <= tol)[0]
        out.extend(zip(hit.tolist(), j[hit].tolist()))
    best: dict[int, tuple[float, int]] = {}
    for i, j in out:
        d = abs(table.values[j] - x[i])
        if i not in best or d < best[i][0]:
            best[i] = (d, j)
    return [(i, j) for i, (_, j) in sorted(best.items())]


def mobius_str(coefs, cname: str = "π") -> str:
    """(p0 + p1 c)/(q0 + q1 c) as readable text, reduced and with a positive denominator."""
    p0, p1, q0, q1 = (int(v) for v in coefs)
    g = math.gcd(math.gcd(p0, p1), math.gcd(q0, q1)) or 1
    lead = q1 if q1 != 0 else q0
    if lead < 0:
        g = -g
    p0, p1, q0, q1 = p0 // g, p1 // g, q0 // g, q1 // g

    def lin(c0, c1):
        if c1 == 0:
            return str(c0)
        t = cname if abs(c1) == 1 else f"{abs(c1)}{cname}"
        if c0 == 0:
            return t if c1 > 0 else "-" + t
        if c1 > 0:
            return f"{t} {'+' if c0 > 0 else '-'} {abs(c0)}"
        return f"{c0} - {t}"

    num, den = lin(p0, p1), lin(q0, q1)
    if den == "1":
        return num
    if p0 != 0 and p1 != 0:
        num = f"({num})"
    if q0 != 0 and q1 != 0:
        den = f"({den})"
    return f"{num}/{den}"


# ---------------------------------------------------------------------------
# high-precision verification
# ---------------------------------------------------------------------------


def constant_mpfr(name: str, seed: int | None = None):
    """Target constant at the current gmpy2 precision. ``random`` = a seeded random real
    in (1, 4), the null control: no polynomial GCF should be a Mobius transform of it."""
    if name == "pi":
        return gmpy2.const_pi()
    if name == "e":
        return gmpy2.exp(1)
    if name == "sqrt2":
        return gmpy2.sqrt(2)
    if name == "random":
        rng = np.random.default_rng(seed)
        bits = gmpy2.get_context().precision + 16
        m = mpz(0)
        for w in rng.integers(0, 2**62, size=bits // 62 + 1).tolist():
            m = (m << 62) | w
        return 1 + 3 * mpfr(m) / mpfr(mpz(1) << (62 * (bits // 62 + 1)))
    raise ValueError(f"unknown constant {name!r}")


def deep_convergents(a, b, depths: list[int], prec: int = 400) -> list:
    """Convergents at the given depths, in mpfr with renormalisation."""
    out = []
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        p_prev, q_prev = mpfr(1), mpfr(0)
        p, q = mpfr(poly_eval(a, 0)), mpfr(1)
        want = sorted(depths)
        k = 0
        for n in range(1, want[-1] + 1):
            an, bn = poly_eval(a, n), poly_eval(b, n)
            p, p_prev = an * p + bn * p_prev, p
            q, q_prev = an * q + bn * q_prev, q
            if n % 64 == 0 and q != 0:
                s = abs(q)
                p, q, p_prev, q_prev = p / s, q / s, p_prev / s, q_prev / s
            if n == want[k]:
                out.append(p / q if q != 0 else mpfr("nan"))
                k += 1
                if k == len(want):
                    break
    return out


@dataclass
class Verdict:
    status: str  # "verified" | "probable" | "rejected"
    digits: float  # agreement with the target at the deepest depth (decimal digits)
    rate: str  # "exponential (x digits/term)" | "power law (error ~ N^-k)" | "stalled"
    errors: list[float]


def verify(a, b, coefs, constant: str = "pi", seed: int | None = None,
           depths=(250, 1000, 4000, 16000), prec: int = 600) -> Verdict:
    """Does the GCF really converge to the Mobius target, or does it stall?"""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        c = constant_mpfr(constant, seed)
        p0, p1, q0, q1 = (int(v) for v in coefs)
        target = (p0 + p1 * c) / (q0 + q1 * c)
        xs = deep_convergents(a, b, list(depths), prec)
        errs = []
        for x in xs:
            e = abs(x - target)
            errs.append(float(gmpy2.log10(e)) if e > 0 else -prec * 0.30103)
    floor = -prec * 0.30103 + 5
    ds = np.array(depths, dtype=float)
    le = np.array(errs)
    digits = float(min(-le[-1], -floor))
    if le[-1] <= floor:  # reached the working precision: fast (exponential) convergence
        with gmpy2.context(gmpy2.get_context(), precision=prec):
            e1, e2 = (abs(x - target) for x in deep_convergents(a, b, [20, 60], prec))
            l1 = float(gmpy2.log10(e1)) if e1 > 0 else -floor
            l2 = float(gmpy2.log10(e2)) if e2 > 0 else -floor
        rate = max((l1 - l2) / 40, 0.0)
        return Verdict("verified", digits, f"exponential (~{rate:.2f} digits/term)", errs)
    per_term = np.diff(le) / np.diff(ds)  # slope in (log10 error) vs depth
    if (per_term < 0).all() and np.ptp(per_term) < 0.3 * abs(per_term.mean()):
        status = "verified" if digits >= 30 else ("probable" if digits >= 8 else "rejected")
        return Verdict(status, digits, f"exponential (~{-per_term.mean():.3g} digits/term)", errs)
    loglog = np.diff(le) / np.diff(np.log10(ds))  # slope in log-log: power law
    if (loglog < -0.5).all() and np.ptp(loglog) < 0.35 * abs(loglog.mean()):
        status = "verified" if digits >= 30 else ("probable" if digits >= 8 else "rejected")
        return Verdict(status, digits, f"power law (error ~ N^{loglog.mean():.2f})", errs)
    return Verdict("rejected", digits, "stalled" if loglog[-1] > -0.5 else "irregular", errs)


def canonical(a, b) -> tuple[tuple, tuple, int]:
    """Undo the equivalence scaling a -> c*a, b -> c^2*b (which maps L -> c*L)."""
    g = 0
    for v in a:
        g = math.gcd(g, int(v))
    best = 1
    for c in range(g, 1, -1) if g > 1 else []:
        if g % c == 0 and all(int(v) % (c * c) == 0 for v in b):
            best = c
            break
    return tuple(int(v) // best for v in a), tuple(int(v) // (best * best) for v in b), best


def shift_poly(c, k: int) -> tuple[int, ...]:
    """Coefficients of p(n + k) (constant term first)."""
    out = [0] * len(c)
    for j, cj in enumerate(c):
        for i in range(j + 1):
            out[i] += cj * math.comb(j, i) * k ** (j - i)
    return tuple(out)


def orbit_key(a, b, max_shift: int = 12) -> str:
    """Canonical name for a fraction up to index shifts n -> n + k and the c-scaling.

    A shifted fraction (start the same fraction k terms later or earlier) has a limit
    that is a Mobius transform of the original's, so it is not a different identity.
    """
    best = None
    for k in range(-max_shift, max_shift + 1):
        sa, sb = shift_poly(a, k), shift_poly(b, k)
        if any(poly_eval(sb, n) == 0 for n in range(1, 60)):
            continue  # not a valid fraction (it would terminate)
        ca, cb, _ = canonical(sa, sb)
        key = (sum(map(abs, ca)) + sum(map(abs, cb)), ca, cb)
        if best is None or key < best:
            best = key
    return f"{poly_str(best[1])} | {poly_str(best[2])}"
