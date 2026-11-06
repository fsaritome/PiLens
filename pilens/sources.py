"""Sequences to look at: pi (and friends) plus the controls we compare them with.

Every source returns a :class:`Seq`: a 1-D array of integer *symbols* in
``range(base)`` plus a ``null`` tag that tells the judge how to draw random
controls that share the sequence's null hypothesis:

* ``"uniform"`` - i.i.d. uniform symbols (the null for digits of a normal number);
* ``"cf"``      - continued-fraction terms of a uniformly random real number
                  (the null for pi's continued fraction, Gauss-Kuzmin statistics).

Digits are computed with gmpy2 (MPFR/GMP) and cached in ``data/cache``.
10^6 decimal digits of pi take about a second.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import gmpy2
import numpy as np
from gmpy2 import mpz

DIGIT_BASES = (2, 4, 8, 10, 16)
CONSTANTS = ("pi", "e", "sqrt2")
SERIES = ("leibniz", "nilakantha", "bbp")
PLANTED_KINDS = ("bias", "period", "markov")


def data_dir() -> Path:
    d = Path(os.environ.get("PILENS_DATA", Path(__file__).resolve().parent.parent / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Seq:
    name: str
    symbols: np.ndarray  # 1-D, integer, values in range(base)
    base: int
    null: str = "uniform"  # how to draw controls: "uniform" | "cf"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.symbols)

    def head(self, n: int) -> "Seq":
        return Seq(self.name, self.symbols[:n], self.base, self.null, dict(self.meta))


# ---------------------------------------------------------------------------
# Digits of constants
# ---------------------------------------------------------------------------

_HEX = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate("0123456789abcdefghijklmnopqrstuvwxyz"):
    _HEX[ord(_c)] = _i


def _str_to_symbols(s: str) -> np.ndarray:
    out = _HEX[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
    assert (out != 255).all()
    return out


def _constant_mpfr(name: str):
    if name == "pi":
        return gmpy2.const_pi()
    if name == "e":
        return gmpy2.exp(1)
    if name == "sqrt2":
        return gmpy2.sqrt(2)
    raise ValueError(f"unknown constant {name!r}; choose from {CONSTANTS}")


def compute_digits(name: str, base: int, n: int) -> np.ndarray:
    """First ``n`` fractional digits of a constant in ``base`` (no caching)."""
    if not 2 <= base <= 36:
        raise ValueError("base must be in 2..36")
    guard = 30  # extra digits so the last requested ones are right
    bits = math.ceil((n + guard) * math.log2(base)) + 64
    with gmpy2.context(gmpy2.get_context(), precision=bits):
        x = _constant_mpfr(name)
        int_part = int(gmpy2.floor(x))
        m = mpz(gmpy2.floor(x * mpz(base) ** (n + guard)))
    s = m.digits(base)
    skip = len(mpz(int_part).digits(base)) if int_part else 0
    if int_part == 0:  # leading fractional zeros are dropped by digits()
        s = s.rjust(n + guard, "0")
    return _str_to_symbols(s[skip : skip + n])


def constant_digits(name: str = "pi", base: int = 10, n: int = 100_000) -> Seq:
    """Cached digits. Caches the largest computed length and slices from it."""
    cdir = cache_dir()
    best = None
    for f in cdir.glob(f"{name}_b{base}_n*.npy"):
        have = int(f.stem.split("_n")[-1])
        if have >= n and (best is None or have < best[0]):
            best = (have, f)
    if best is not None:
        sym = np.load(best[1])[:n]
    else:
        sym = compute_digits(name, base, n)
        np.save(cdir / f"{name}_b{base}_n{n}.npy", sym)
    return Seq(f"{name}", sym, base, "uniform", {"kind": "digits", "constant": name})


# ---------------------------------------------------------------------------
# Continued fractions
# ---------------------------------------------------------------------------

def _cf_of_interval(lo: mpz, hi: mpz, den: mpz, max_terms: int) -> list[int]:
    """CF terms common to lo/den and hi/den (so valid for any real in between)."""
    a, b = lo, den
    c, d = hi, den
    terms: list[int] = []
    while len(terms) < max_terms and b and d:
        q1, r1 = gmpy2.f_divmod(a, b)
        q2, r2 = gmpy2.f_divmod(c, d)
        if q1 != q2:
            break
        terms.append(int(q1))
        a, b = b, r1
        c, d = d, r2
    return terms


def _cf_bits(n_terms: int) -> int:
    # Levy: q_n ~ exp(1.1866 n); a real known to 2*log2(q_n) bits fixes n terms.
    return int(n_terms * 3.6) + 256


def cf_terms_of(x_num: mpz, bits: int, n_terms: int) -> list[int]:
    """Terms (after a0) of the real in [x_num, x_num+1) / 2^bits."""
    den = mpz(1) << bits
    t = _cf_of_interval(x_num, x_num + 1, den, n_terms + 1)
    return t[1 : n_terms + 1]


def pi_cf_terms(n_terms: int = 20_000) -> np.ndarray:
    f = cache_dir() / f"pi_cf_n{n_terms}.npy"
    if f.exists():
        return np.load(f)
    bits = _cf_bits(n_terms)
    with gmpy2.context(gmpy2.get_context(), precision=bits + 64):
        num = mpz(gmpy2.floor(gmpy2.const_pi() * (mpz(1) << bits)))
    terms = np.array(cf_terms_of(num, bits, n_terms), dtype=np.int64)
    if len(terms) < n_terms:
        raise RuntimeError("not enough precision for requested CF terms")
    np.save(f, terms)
    return terms


def random_cf_terms(n_terms: int, seed: int) -> np.ndarray:
    """CF terms of a uniformly random real: the exact null for pi's CF."""
    bits = _cf_bits(n_terms)
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2**63, size=bits // 63 + 1, dtype=np.int64)
    num = mpz(0)
    for w in raw.tolist():
        num = (num << 63) | w
    num &= (mpz(1) << bits) - 1
    terms = cf_terms_of(num, bits, n_terms)
    if len(terms) < n_terms:  # extremely unlikely; retry
        return random_cf_terms(n_terms, seed + 10_000_019)
    return np.array(terms, dtype=np.int64)


def cf_to_symbols(terms: np.ndarray, base: int) -> np.ndarray:
    """Symbol = min(a, base) - 1, so 1..base-1 map exactly and >=base share the top symbol."""
    return (np.minimum(terms, base) - 1).astype(np.uint8)


def pi_cf(n: int = 20_000, base: int = 10) -> Seq:
    t = pi_cf_terms(n)
    return Seq("pi_cf", cf_to_symbols(t, base), base, "cf", {"kind": "cf"})


# ---------------------------------------------------------------------------
# Series terms (trivially structured: useful as obvious positives)
# ---------------------------------------------------------------------------

def series_log_terms(name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(sign, log10|t_k|) of the k-th term of a series for pi."""
    k = np.arange(n, dtype=np.float64)
    if name == "leibniz":
        sign = np.where(k % 2 == 0, 1.0, -1.0)
        logt = np.log10(4.0) - np.log10(2 * k + 1)
    elif name == "nilakantha":
        kk = np.maximum(k, 1)
        sign = np.where(k == 0, 1.0, np.where(k % 2 == 1, 1.0, -1.0))
        logt = np.where(k == 0, np.log10(3.0),
                        np.log10(4.0) - np.log10(2 * kk) - np.log10(2 * kk + 1) - np.log10(2 * kk + 2))
    elif name == "bbp":
        inner = 4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6)
        sign = np.sign(inner)
        logt = -k * np.log10(16.0) + np.log10(np.abs(inner))
    else:
        raise ValueError(f"unknown series {name!r}; choose from {SERIES}")
    return sign, logt


def series(name: str, n: int = 100_000, base: int = 10) -> Seq:
    """Symbolise term k as the mantissa bin of |t_k| (fractional part of log10, B bins)."""
    _, logt = series_log_terms(name, n)
    frac = logt - np.floor(logt)
    sym = np.minimum((frac * base).astype(np.int64), base - 1).astype(np.uint8)
    return Seq(name, sym, base, "uniform", {"kind": "series"})


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def random_digits(n: int, base: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, base, size=n, dtype=np.uint8)


def random_seq(n: int, base: int = 10, seed: int = 12345) -> Seq:
    return Seq(f"random_s{seed}", random_digits(n, base, seed), base, "uniform", {"kind": "random"})


def planted(kind: str, strength: float, n: int, base: int = 10, seed: int = 777,
            period: int = 7) -> Seq:
    """Random digits with weak structure planted at ``strength`` in [0, 1].

    * ``bias``   - each digit is replaced by 0 with probability ``strength``;
    * ``period`` - each digit is replaced by ``pattern[i % period]`` w.p. ``strength``;
    * ``markov`` - each digit copies its predecessor w.p. ``strength``.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, base, size=n).astype(np.int64)
    hit = rng.random(n) < strength
    if kind == "bias":
        x[hit] = 0
    elif kind == "period":
        pattern = rng.integers(0, base, size=period)
        idx = np.nonzero(hit)[0]
        x[idx] = pattern[idx % period]
    elif kind == "markov":
        # sequential dependence: x[i] = x[i-1] where hit (resolve runs left to right)
        idx = np.nonzero(hit[1:])[0] + 1
        # positions where a run of copies starts take the value just before the run
        src = np.arange(n)
        src[idx] = -1
        # forward-fill the last non-copied index
        valid = np.where(src >= 0, src, 0)
        np.maximum.accumulate(valid, out=valid)
        x = x[valid]
    else:
        raise ValueError(f"unknown planted kind {kind!r}; choose from {PLANTED_KINDS}")
    return Seq(f"planted_{kind}_{strength:g}", x.astype(np.uint8), base, "uniform",
               {"kind": "planted", "planted": kind, "strength": strength, "seed": seed})


def make_control(seq: Seq, seed: int) -> np.ndarray:
    """One random control sequence drawn from ``seq``'s null hypothesis."""
    n = len(seq)
    if seq.null == "uniform":
        return random_digits(n, seq.base, seed)
    if seq.null == "cf":
        return cf_to_symbols(random_cf_terms(n, seed), seq.base)
    raise ValueError(f"unknown null {seq.null!r}")


def get_source(source: str, base: int, n: int, seed: int = 12345, strength: float = 0.03) -> Seq:
    """Resolve a source name used on the CLI.

    ``pi``, ``e``, ``sqrt2``: digits;  ``pi_cf``: continued fraction;
    ``leibniz``/``nilakantha``/``bbp``: series;  ``random``: seeded random digits;
    ``planted_bias``/``planted_period``/``planted_markov``: planted structure.
    """
    if source in CONSTANTS:
        return constant_digits(source, base, n)
    if source == "pi_cf":
        return pi_cf(n, base)
    if source in SERIES:
        return series(source, n, base)
    if source == "random":
        return random_seq(n, base, seed)
    if source.startswith("planted_"):
        return planted(source.removeprefix("planted_"), strength, n, base, seed)
    raise ValueError(f"unknown source {source!r}")
