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
