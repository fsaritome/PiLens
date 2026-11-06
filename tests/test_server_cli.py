import json

from fastapi.testclient import TestClient

from pilens import store
from pilens.cli import main
from pilens.server import app


def test_cli_run_then_gallery_and_rating(capsys):
    main(["run", "--lens", "walk,pairs", "--base", "4", "--n", "3000", "--controls", "30",
          "--perm", "30", "--chunks", "8", "--jobs", "1"])
    out = capsys.readouterr().out
    assert "BH across the 2 lens(es)" in out
    run = "pi_b4_n3000"
    d = store.runs_dir() / run
    assert (d / "walk_pi.png").exists() and (d / "walk_ctrl.png").exists()
    summary = json.loads((d / "summary.json").read_text())
    assert {s["key"] for s in summary} == {"walk", "pairs"} and all("q_lens" in s for s in summary)

    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200 and run in r.text and "walk" in r.text
    r = c.get(f"/run/{run}/walk")
    assert r.status_code == 200 and "random control" in r.text and "nu" in r.text
    assert c.get(f"/img/{run}/walk_pi.png").headers["content-type"] == "image/png"
    r = c.post("/api/rate", json={"run": run, "key": "walk", "rating": 1, "note": "looks like a protein"})
    assert r.json() == {"ok": True, "up": 1, "down": 0}
    assert c.post("/api/rate", json={"run": run, "key": "walk", "rating": 5}).status_code == 400
    assert c.get(f"/run/{run}/..%2F..%2Fx").status_code == 404
    rec = store.load_ratings()[-1]
    assert rec["lens"] == "walk" and rec["base"] == 4 and rec["note"] == "looks like a protein"


def test_cli_lens_params_get_own_key():
    main(["run", "--lens", "grid", "--param", "width=70", "--base", "10", "--n", "3000",
          "--controls", "20", "--no-classifier", "--jobs", "1"])
    keys = {s["key"] for s in json.loads((store.runs_dir() / "pi_b10_n3000" / "summary.json").read_text())}
    assert any(k.startswith("grid__") for k in keys)


def test_chord_3d_page():
    main(["run", "--lens", "chord", "--source", "random", "--base", "10", "--n", "3000", "--controls", "20",
          "--no-classifier", "--jobs", "1"])
    run = next(d.name for d in store.list_runs() if d.name.startswith("random_s12345_b10_n3000"))
    c = TestClient(app)
    r = c.get(f"/run/{run}/chord")
    assert "view the chords in 3D" in r.text
    r = c.get(f"/run/{run}/chord/3d")
    assert r.status_code == 200 and "scatter3d" in r.text and '"ctrl"' in r.text
    assert c.get("/run/nope/chord/3d").status_code == 404


def test_walk3d_page_and_stl():
    main(["run", "--lens", "walk3d", "--source", "random", "--base", "10", "--n", "3000", "--controls", "20",
          "--no-classifier", "--jobs", "1"])
    run = next(d.name for d in store.list_runs() if d.name.startswith("random_s12345_b10_n3000"))
    c = TestClient(app)
    assert "walk and its shell in 3D" in c.get(f"/run/{run}/walk3d").text
    r = c.get(f"/run/{run}/walk3d/3d")
    assert r.status_code == 200 and "mesh3d" in r.text and "percentile" in r.text
    assert c.get("/").status_code == 200  # the cached control volumes must not break the index
    stl = c.get(f"/run/{run}/walk3d/shell.stl")
    assert stl.status_code == 200 and stl.text.startswith("solid") and stl.text.count("facet normal") >= 4
    assert c.get(f"/run/{run}/walk3d/other.stl").status_code == 404
