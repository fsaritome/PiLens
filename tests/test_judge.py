import numpy as np
import pytest

from pilens import sources as S
from pilens.judge import bh, default_jobs, image_metrics, judge, ridge_cv_permutation
from pilens.lenses import LENSES, load_plugins

load_plugins()


def test_bh_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    q = bh(p)
    # sorted p: .01 .03 .04 .2 -> raw .04 .06 .0533 .2 -> monotone .04 .0533 .0533 .2
    assert np.allclose(q, [0.04, 0.05333333, 0.05333333, 0.2])


def test_ridge_classifier():
    rng = np.random.default_rng(0)
    y = np.array([1] * 30 + [0] * 30)
    X = rng.normal(size=(60, 4))
    acc, p = ridge_cv_permutation(X, y, 200, 0)
    assert p > 0.05
    X[:, 0] += 2.5 * y
    acc, p = ridge_cv_permutation(X, y, 200, 0)
    assert acc > 0.85 and p < 0.01


def test_image_metrics_finite():
    m = image_metrics(np.random.default_rng(0).random((100, 80)))
    assert all(np.isfinite(v) for v in m.values())
    m = image_metrics(np.zeros((50, 50)))
    assert all(np.isfinite(v) for v in m.values())


def test_judge_result_shape():
    seq = S.random_seq(4000, 10, seed=3)
    r = judge("pairs", seq, n_controls=40, k_chunks=8, n_perm=40)
    assert 0 < r.p_maxT <= 1 and 0 < r.p_lens <= 1 and r.clf_acc is not None
    assert {m.name for m in r.metrics} >= {"pair_lag1_z", "img_entropy_bits"}


def test_judge_cf_null():
    seq = S.pi_cf(1500, 10)
    r = judge("pairs", seq, n_controls=30, classifier=False)
    assert r.p_maxT > 0


def _flagged(seq, n_controls, jobs):
    from pilens.judge import judge_many
    rs = judge_many(list(LENSES), seq, n_controls=n_controls, jobs=jobs)
    return {r.lens: r.q_lens for r in rs if r.q_lens < 0.05}


@pytest.mark.slow
@pytest.mark.parametrize("kind", S.PLANTED_KINDS)
def test_validation_flags_planted_structure(kind):
    """At 3% strength and n=20000, at least one lens must survive BH at q<0.05."""
    seq = S.planted(kind, 0.03, 20_000, 10, seed=101)
    hits = _flagged(seq, 300, default_jobs())
    assert hits, f"planted {kind} not detected"


@pytest.mark.slow
def test_validation_random_not_flagged():
    seq = S.random_seq(20_000, 10, seed=4242)
    hits = _flagged(seq, 300, default_jobs())
    assert not hits, f"false positive on fresh random digits: {hits}"
