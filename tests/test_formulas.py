import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pilens import gcf
from pilens.evolve import FamilyProposal, Proposal, history, run_round, summary_for_prompt, validate
from pilens.hunt import PRESETS, run_hunt


def test_known_identities_verify():
    v = gcf.verify((1, 2), (0, 0, 1), (4, 0, 0, 1))  # 4/pi = 1 + 1/(3 + 4/(5 + ...))
    assert v.status == "verified" and v.digits > 100
    v = gcf.verify((6,), (1, -4, 4), (3, 1, 1, 0))  # pi + 3, slow (error ~ N^-3)
    assert v.status == "probable" and "power law" in v.rate


def test_coincidence_rejected():
    # the 4/pi fraction compared with a random constant's nearest Mobius value: must stall
    v = gcf.verify((1, 2), (0, 0, 1), (4, 0, 0, 1), constant="random", seed=3)
    assert v.status == "rejected"
    # a target off by 1e-12 is still caught, because the error stops shrinking
    import gmpy2
    with gmpy2.context(gmpy2.get_context(), precision=600):
        assert gcf.verify((1, 2), (0, 0, 1), (4, 0, 0, 1)).digits > 100


def test_scan_matches_float():
    A = np.array([[1, 2], [6, 0]])
    B = np.array([[0, 0, 1], [1, -4, 4]])
    x, h, term = gcf.scan(A, B, 2000)
    assert abs(x[0] - 4 / np.pi) < 1e-14 and abs(x[1] - (np.pi + 3)) < 1e-9 and not term.any()


def test_formatting_and_canonical():
    assert gcf.mobius_str((4, 0, -2, 1)) == "4/(π - 2)"
    assert gcf.mobius_str((3, 1, 1, 0)) == "π + 3"
    assert gcf.mobius_str((-8, 0, -2, 0)) == "4"
    assert gcf.poly_str((1, -4, 4)) == "4n^2-4n+1"
    assert gcf.gcf_str((1, 2), (0, 0, 1)) == "1 + 1/(3 + 4/(5 + 9/(7 + ...)))"
    assert gcf.canonical((2, 4), (0, 0, 4)) == ((1, 2), (0, 0, 1), 2)


def test_family_enumerate_removes_sign_symmetry():
    A, B = gcf.Family([(-1, 1), (0, 1)], [(-1, 1)]).enumerate()
    # leading nonzero a-coefficient positive: (1,0),(-1,1),(0,1),(1,1); b != 0: 2 values
    assert len(A) == 8 and set(map(tuple, A)) == {(1, 0), (-1, 1), (0, 1), (1, 1)}


def test_hunt_rediscovers_classics_and_control_is_clean():
    h = run_hunt(gcf.Family.from_dict(PRESETS["classic"]), log=lambda *a: None)
    canon = {json.dumps(r["canonical"]) for r in h["distinct_relations"]}
    assert json.dumps([[1, 2], [0, 0, 1]]) in canon  # 4/pi
    assert json.dumps([[6], [1, -4, 4]]) in canon  # pi + 3 (Euler)
    assert h["null"]["count"]["verified"] == 0 and h["null"]["count"]["probable"] == 0
    assert len(h["distinct_relations"]) >= 20


def test_evolve_round_from_proposal_and_validation():
    prop = Proposal(notes="test round", families=[
        FamilyProposal(name="tiny", rationale="smoke test", a=[[0, 2], [1, 2]], b=[[-2, 2], [-2, 2], [0, 1]]),
        FamilyProposal(name="too big", rationale="should be refused", a=[[-30, 30], [-30, 30]], b=[[-30, 30]] * 3),
    ])
    with pytest.raises(ValueError):
        validate(prop.families[1])
    rec = run_round(prop, proposer="test", log=lambda *a: None)
    assert rec["results"][1]["error"] and rec["results"][0]["n_candidates"] > 0
    assert history()[-1]["notes"] == "test round"
    assert "Hunts so far" in summary_for_prompt()


def test_formula_pages():
    from pilens.hunt import list_hunts
    from pilens.server import app

    c = TestClient(app)
    hid = list_hunts()[0]["id"]
    r = c.get("/formulas")
    assert r.status_code == 200 and "random constant" in r.text and "4/π" in r.text and "Distinct identities with" in r.text
    r = c.get(f"/formulas/{hid}")
    assert r.status_code == 200 and "landscape" in r.text
    assert c.get(f"/hunt-img/{hid}/landscape_target.png").headers["content-type"] == "image/png"
    r = c.get(f"/formulas/{hid}/3d")
    assert r.status_code == 200 and "scatter3d" in r.text and '"dims"' in r.text
    r = c.post("/api/rate-formula", json={"run": hid, "key": "2n+1 | n^2", "rating": 1})
    assert r.json()["up"] == 1
    assert c.get("/formulas/..%2F..%2Fx").status_code == 404


def test_equivalence_reduction():
    # (2n+1)(n+1) | n^3 (n+1) is 2n+1 | n^2 dressed with c(n) = n + 1
    assert gcf.family_key((1, 3, 2), (0, 0, 0, 1, 1)) == "2n+1 | n^2"
    # 3n(n+1) | -n^2 (2n-3)(n+1) reduces to the 3n family
    assert gcf.family_key((0, 3, 3), (0, 0, 3, 1, -2)) == gcf.family_key((0, 3), (0, 3, -2))
    # a fraction with no linear factor in a(n) is left alone
    a, b = (-1, 3, 3), (0, 0, 6, -1, -2)
    assert gcf.reduce_equivalence(a, b) == (a, b)


def test_novelty_sequences_and_lookup(tmp_path):
    from pilens import novelty as nv

    # 4/pi = 1 + 1/(3 + 4/(5 + 9/(7 + ...))): a = 2n+1, b = n^2
    P, Q = nv.convergent_sequences((1, 2), (0, 0, 1), 6)
    assert P[:3] == [1, 4, 24] and Q[:4] == [1, 3, 19, 160]
    assert nv.search_terms([0, 1, -2, -4, -6, -8, -10]) == [1, 2, 3, 4, 5]

    class Fake(nv.OEIS):
        def search(self, terms):
            return [{"id": "A000001", "name": "Expansion of 4/Pi", "text": "Expansion of 4/Pi"}]

    r = nv.check_core((1, 2), (0, 0, 1), "pi", Fake(cache=tmp_path / "c.json"))
    assert r["status"] == "known"

    class Empty(nv.OEIS):
        def search(self, terms):
            return []

    assert nv.check_core((1, 2), (0, 0, 1), "pi", Empty(cache=tmp_path / "c.json"))["status"] == "not in OEIS"


def test_export_site(tmp_path):
    from pilens.export import _relink, export_site

    assert _relink('<a href="/formulas/x">', "formulas/index.html") == '<a href="x/index.html">'
    assert _relink("<img src='/hunt-img/h/l.png'>", "formulas/h/index.html") == "<img src='../../hunt-img/h/l.png'>"
    assert _relink('<a href="https://oeis.org/A1">', "index.html") == '<a href="https://oeis.org/A1">'
    n = export_site(tmp_path / "site", log=lambda *a: None)
    assert n > 3 and (tmp_path / "site" / "index.html").exists() and (tmp_path / "site" / "formulas" / "index.html").exists()
    assert 'href="/' not in (tmp_path / "site" / "formulas" / "index.html").read_text(encoding="utf-8")


def test_compare_dynamics():
    import math

    from pilens.compare import _ratio_label, dynamics

    # 4/pi = 1 + 1/(3 + 4/(5 + ...)): converges like (3 + 2 sqrt 2)^-2n, so r -> ln(3 + 2 sqrt 2)
    d = dynamics((1, 2), (0, 0, 1), depth=400)
    assert abs(d["r"] - math.log(3 + 2 * math.sqrt(2))) < 0.02
    assert -0.3 < d["delta"] < -0.1
    assert _ratio_label(0.7, 0.7) == "same rate" and _ratio_label(1.4, 0.7) == "ours = 2× theirs"
    assert _ratio_label(0.7, 1.76) is None


def test_pslq_and_speed_classes():
    from pilens import gcf
    from pilens.compare import classify, same_class

    # a published degree-3/6 formula with a large-coefficient limit, 72/(3 pi - 8)
    a, b = (55, 166, 148, 40), (0, 0, 288, 96, -832, -896, -256)
    rel = gcf.pslq_relation(a, b)
    assert rel is not None and gcf.mobius_str(rel) == "72/(3π - 8)"
    assert gcf.pslq_relation(a, b, "random", 7) is None
    assert same_class(-0.65, 0.69, -0.651, 1.387)  # a 2-fold
    assert not same_class(-0.65, 0.69, -0.20, 1.76)
    assert same_class(-0.99, 0.01, -0.97, 0.03)  # slow ones are one class
    known = [{"delta": -0.2, "r": 1.763}]
    assert classify(-0.205, 3.527, known) is not None and classify(-0.5, 4.0, known) is None
