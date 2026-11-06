import numpy as np
import pytest

from pilens.judge import evaluate
from pilens.lenses import LENSES, get_lens, load_plugins, walk_scaling

load_plugins()
X10 = np.random.default_rng(0).integers(0, 10, 20_000)


@pytest.mark.parametrize("name", sorted(LENSES))
@pytest.mark.parametrize("base", [2, 4, 10, 16])
def test_lens_contract(name, base):
    x = X10 % base
    res = get_lens(name)(x, base)
    assert res.image.ndim in (2, 3) and max(res.image.shape[:2]) <= 1024
    if res.image.dtype != np.uint8:
        assert np.nanmin(res.image) >= 0 and np.nanmax(res.image) <= 1
    assert res.features and all(np.isfinite(v) for v in res.features.values())
    res2 = get_lens(name)(x, base)
    assert res.features == res2.features  # deterministic


def test_unknown_param_rejected():
    with pytest.raises(TypeError):
        get_lens("walk")(X10, 10, nope=1)


def test_plugin_discovered():
    assert LENSES["spiral"].origin != "builtin"


def test_spiral_visits_each_cell_once():
    side = 31
    x = np.arange(side * side) % 7
    img = get_lens("spiral")(x, 7, side=side).image
    # every cell filled exactly once means the image values recover x as a multiset
    vals = np.round(img.ravel() * 6).astype(int)
    assert (np.bincount(vals, minlength=7) == np.bincount(x, minlength=7)).all()


def test_walk_scaling_exponents():
    rng = np.random.default_rng(1)
    ang = 2 * np.pi * rng.integers(0, 4, 2**15) / 4
    pos = np.vstack([[0, 0], np.cumsum(np.stack([np.cos(ang), np.sin(ang)], 1), 0)])
    assert abs(walk_scaling(pos) - 0.5) < 0.08
    straight = np.vstack([[0, 0], np.cumsum(np.ones((2**15, 2)), 0)])
    assert abs(walk_scaling(straight) - 1.0) < 1e-6


def test_walk_self_intersection_base4_lattice():
    feats = get_lens("walk")(np.array([0, 1, 2, 3] * 100), 4).features  # tiny closed squares
    assert feats["self_intersection"] > 0.7
    feats = get_lens("walk")(np.zeros(400, dtype=int), 4).features  # straight line
    assert feats["self_intersection"] == 0.0


def test_evaluate_splits_info_features():
    _, metrics, info = evaluate("grid", X10, 10)
    assert "_fold_best_width" in info and "_fold_best_width" not in metrics
    assert {"img_compress_ratio", "img_entropy_bits", "fft_slope", "fft_peak_log10"} <= set(metrics)
