"""Persist judge results and images under ``data/runs`` and ratings in ``data/ratings.jsonl``.

Layout::

    data/runs/<source>_b<base>_n<n>/
        <lenskey>.json        JudgeResult (+ q_lens from BH over every lens in this run dir)
        <lenskey>_pi.png      the lens applied to the source
        <lenskey>_ctrl.png    the lens applied to one random control
        summary.json          all lenses in the run, with BH q-values

``lenskey`` is the lens name, plus ``__<hash>`` when non-default params were used.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .judge import JudgeResult, bh, to_gray
from .sources import data_dir

# perceptual-ish dark->bright palette (inferno-like anchors)
_ANCHORS = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
], dtype=np.float64)


def colorize(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img.astype(np.uint8)
    g = np.clip(to_gray(img), 0, 1)
    pos = g * (len(_ANCHORS) - 1)
    i = np.minimum(pos.astype(np.int64), len(_ANCHORS) - 2)
    f = (pos - i)[..., None]
    return (_ANCHORS[i] * (1 - f) + _ANCHORS[i + 1] * f).astype(np.uint8)


def save_png(img: np.ndarray, path: Path, min_side: int = 384) -> None:
    rgb = colorize(img)
    h, w = rgb.shape[:2]
    k = max(1, int(np.ceil(min_side / max(h, w))))
    if k > 1:
        rgb = np.kron(rgb, np.ones((k, k, 1), dtype=np.uint8))
    Image.fromarray(rgb).save(path)


def runs_dir() -> Path:
    d = data_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_id(source: str, base: int, n: int) -> str:
    return f"{source}_b{base}_n{n}"


def lens_key(lens: str, params: dict) -> str:
    if not params:
        return lens
    h = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{lens}__{h}"


def _clean(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def save_result(res: JudgeResult, img_pi, img_ctrl, user_params: dict) -> Path:
    d = runs_dir() / run_id(res.source, res.base, res.n)
    d.mkdir(parents=True, exist_ok=True)
    key = lens_key(res.lens, user_params)
    save_png(img_pi, d / f"{key}_pi.png")
    save_png(img_ctrl, d / f"{key}_ctrl.png")
    rec = res.to_dict()
    rec.update(key=key, user_params=user_params, saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    (d / f"{key}.json").write_text(json.dumps(_clean(rec), indent=1))
    update_summary(d)
    return d


def load_run(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.json")):
        if f.name != "summary.json":
            out.append(json.loads(f.read_text()))
    return out


def update_summary(d: Path) -> list[dict]:
    """Recompute BH across all lenses in a run dir (the multiple-testing family)."""
    recs = load_run(d)
    qs = bh([r["p_lens"] for r in recs])
    for r, q in zip(recs, qs):
        r["q_lens"] = float(q)
        (d / f"{r['key']}.json").write_text(json.dumps(r, indent=1))
    summary = [{k: r.get(k) for k in ("key", "lens", "p_maxT", "clf_acc", "clf_p", "p_lens",
                                      "q_lens", "n_controls")} | {"top": top_metric(r)}
               for r in recs]
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return recs


def top_metric(rec: dict) -> dict:
    ms = [m for m in rec["metrics"] if m.get("z") is not None]
    if not ms:
        return {}
    m = max(ms, key=lambda m: abs(m["z"]))
    return {"name": m["name"], "z": m["z"], "p": m["p"]}


def list_runs() -> list[Path]:
    return sorted([p for p in runs_dir().iterdir() if p.is_dir()])


def ratings_path() -> Path:
    return data_dir() / "ratings.jsonl"


def add_rating(run: str, key: str, rating: int, note: str = "", extra: dict | None = None) -> dict:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": run, "key": key,
           "rating": int(rating), "note": note, **(extra or {})}
    with ratings_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load_ratings() -> list[dict]:
    p = ratings_path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
