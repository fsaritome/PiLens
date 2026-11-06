import numpy as np
import pytest

from pilens import sources as S


def as_str(sym):
    return "".join("0123456789abcdef"[d] for d in sym)


@pytest.mark.parametrize("base,expected", [
    (10, "14159265358979323846264338327950288419716939937510"),
    (16, "243f6a8885a308d313198a2e03707344a4093822299f31d008"),
    (2, "0010010000111111011010101000100010000101101000110000"),
    (4, "021003331222202020112203002031030103012120220232000313"),
])
def test_pi_digits(base, expected):
    assert as_str(S.compute_digits("pi", base, len(expected))) == expected


def test_other_constants():
    assert as_str(S.compute_digits("e", 10, 20)) == "71828182845904523536"
    assert as_str(S.compute_digits("sqrt2", 10, 20)) == "41421356237309504880"


def test_cache_slices_longer_run():
    a = S.constant_digits("pi", 10, 5000).symbols
    b = S.constant_digits("pi", 10, 1234).symbols
    assert (a[:1234] == b).all()
    assert len(b) == 1234


def test_pi_cf_prefix():
    # OEIS A001203 (without a0 = 3)
    assert S.pi_cf_terms(2000)[:20].tolist() == [7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2, 1]


def test_random_cf_follows_gauss_kuzmin():
    t = np.concatenate([S.random_cf_terms(3000, s) for s in range(3)])
    p1 = np.log2(4 / 3)  # P(a = 1) ~ 0.415
    assert abs((t == 1).mean() - p1) < 0.02
    sym = S.cf_to_symbols(t, 10)
    assert sym.min() >= 0 and sym.max() <= 9


def test_planted_strengths():
    n, s = 200_000, 0.05
    bias = S.planted("bias", s, n).symbols
    assert abs((bias == 0).mean() - (0.1 + 0.9 * s)) < 0.005
    markov = S.planted("markov", s, n).symbols
    assert abs((markov[1:] == markov[:-1]).mean() - (0.1 + 0.9 * s)) < 0.005
    per = S.planted("period", s, n, period=7).symbols
    col = per[: n // 7 * 7].reshape(-1, 7)[:, 3]
    assert np.bincount(col, minlength=10).max() / len(col) > 0.1 + 0.9 * s - 0.01
    zero = S.planted("bias", 0.0, n, seed=1).symbols
    assert abs((zero == 0).mean() - 0.1) < 0.005


def test_series_and_controls():
    for name in S.SERIES:
        seq = S.series(name, 1000, base=10)
        assert seq.symbols.min() >= 0 and seq.symbols.max() <= 9 and len(seq) == 1000
    seq = S.get_source("pi", 4, 1000)
    c1, c2 = S.make_control(seq, 1), S.make_control(seq, 2)
    assert len(c1) == 1000 and c1.max() <= 3 and not (c1 == c2).all()
    assert (S.make_control(seq, 1) == c1).all()  # seeded
