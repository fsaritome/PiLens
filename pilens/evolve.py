"""Phase 2: a FunSearch-style loop where Claude proposes where to search next.

Each round:

1. Claude (``claude-opus-5-5``) reads a summary of every hunt so far: which families
   were searched, how many identities with pi each produced (verified at high
   precision), what the random-constant control found (should be nothing), and the
   distinct identities found so far, together with any ratings.
2. It proposes 1-3 new families of polynomial continued fractions, each with its
   reasoning, as structured JSON.
3. Each family is hunted (``pilens.hunt.run_hunt``). The judge is mechanical:
   high-precision verification plus the null control. A round's score is the number of
   *new* distinct identities (up to the a -> c*a, b -> c^2*b scaling).

Proposals are data, not code: a family is only coefficient ranges, which are validated
and size-capped before anything runs.

Without API credentials, ``--proposals file.json`` runs a round from a file with the
same shape (``{"families": [...], "notes": "..."}``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from . import gcf
from .hunt import KNOWN, list_hunts, run_hunt
from .sources import data_dir

MODEL = "claude-opus-5-5"
MAX_SIZE = 400_000
MAX_DEG_A, MAX_DEG_B, MAX_COEF = 2, 4, 30


class FamilyProposal(BaseModel):
    name: str = Field(description="short descriptive name")
    rationale: str = Field(description="why this region might hold new identities with pi")
    a: list[list[int]] = Field(description="[lo, hi] for each coefficient of a(n), constant term first")
    b: list[list[int]] = Field(description="[lo, hi] for each coefficient of b(n), constant term first")
    mobius: int = Field(default=6, description="max |coefficient| in the target (p0 + p1 pi)/(q0 + q1 pi), 2..12")


class Proposal(BaseModel):
    families: list[FamilyProposal] = Field(description="1 to 3 families to hunt next")
    notes: str = Field(description="what you learned from the results so far, in 2-4 sentences")


SYSTEM = f"""You are steering a search for continued-fraction formulas for pi, in the spirit of \
the Ramanujan Machine and FunSearch.

Objects: generalised continued fractions a(0) + b(1)/(a(1) + b(2)/(a(2) + ...)) with integer \
polynomial terms a(n), b(n). A *family* is a set of [lo, hi] ranges, one per polynomial \
coefficient (constant term first); every combination is evaluated.

Judge (mechanical, not you): each fraction's limit is matched against (p0 + p1*pi)/(q0 + q1*pi) \
with |p|,|q| <= M (default 6, a family may ask for up to 12) and then re-computed at 600-bit precision to depth 16,000. "verified" = the \
error keeps shrinking to 30+ digits; "probable" = 8-30 digits with a clean convergence law. \
The same search is run against a random constant as a control; any hit there would mean a \
false positive. Fractions that converge too slowly (< 7 digits at depth 2000) are skipped.

Constraints: degree of a(n) <= {MAX_DEG_A}, degree of b(n) <= {MAX_DEG_B}, |coefficients| <= \
{MAX_COEF}, at most {MAX_SIZE:,} combinations per family (product of range widths). The sign \
symmetry a -> -a is removed automatically.

Goal: maximise the number of *new distinct* identities (up to the trivial scaling a -> c*a, \
b -> c^2*b) that are verified. Prefer families that differ structurally from those already \
searched (e.g. other degree pairs, the balance between the leading terms of a and b that \
governs convergence, b(n) with a factor like n^2 or (2n-1)^2), rather than widening ranges \
blindly. Known classical formulas: {'; '.join(KNOWN.values())}."""


def evolve_dir() -> Path:
    d = data_dir() / "evolve"
    d.mkdir(parents=True, exist_ok=True)
    return d


def history() -> list[dict]:
    f = evolve_dir() / "rounds.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary_for_prompt(max_identities: int = 60) -> str:
    hunts = list_hunts()
    lines = ["## Hunts so far"]
    seen: dict[str, dict] = {}
    for h in hunts:
        t, n = h["target"]["count"], h["null"]["count"]
        lines.append(
            f"- {h['family']['name'] or h['id']}: a ranges {h['family']['a']}, b ranges {h['family']['b']}; "
            f"{h['n_candidates']:,} fractions, {h['n_eligible']:,} converged fast enough; "
            f"pi: {t['verified']} verified, {t['probable']} probable "
            f"({len(h['distinct_relations'])} distinct); random-constant control: "
            f"{n['verified'] + n['probable']} hits")
        for r in h["distinct_relations"]:
            seen.setdefault(r["orbit"], r)
    lines.append(f"\n## Distinct identities found so far: {len(seen)} (showing up to {max_identities})")
    for r in list(seen.values())[:max_identities]:
        lines.append(f"- a(n) = {r['a_str']}, b(n) = {r['b_str']}  ->  {r['relation']}  "
                     f"[{r['status']}, {r['rate']}]{'  (classical)' if r['known'] else ''}")
    ratings = data_dir() / "ratings.jsonl"
    if ratings.exists():
        rs = [json.loads(x) for x in ratings.read_text(encoding="utf-8").splitlines() if x.strip()]
        fr = [r for r in rs if r.get("kind") == "formula"]
        if fr:
            lines.append("\n## Human ratings of identities")
            lines += [f"- {r['key']}: {'+1' if r['rating'] > 0 else '-1'} {r.get('note', '')}" for r in fr[-30:]]
    for r in history()[-5:]:
        lines.append(f"\n## Round {r['round']} notes ({r['proposer']}): {r['notes']}")
    return "\n".join(lines)


def propose_api(summary: str) -> Proposal:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": summary + "\n\nPropose the next 1-3 families."}],
        output_format=Proposal,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"no proposal (stop_reason={resp.stop_reason})")
    return resp.parsed_output


def validate(fp: FamilyProposal) -> gcf.Family:
    fam = gcf.Family([tuple(r) for r in fp.a], [tuple(r) for r in fp.b], fp.name)
    if not (1 <= len(fam.a) <= MAX_DEG_A + 1 and 1 <= len(fam.b) <= MAX_DEG_B + 1):
        raise ValueError(f"{fp.name}: degrees out of range")
    for lo, hi in fam.a + fam.b:
        if lo > hi or max(abs(lo), abs(hi)) > MAX_COEF:
            raise ValueError(f"{fp.name}: bad range [{lo}, {hi}]")
    if not 2 <= fp.mobius <= 12:
        raise ValueError(f"{fp.name}: mobius must be 2..12")
    if fam.size() > MAX_SIZE:
        raise ValueError(f"{fp.name}: {fam.size():,} combinations > {MAX_SIZE:,}")
    return fam


def run_round(proposal: Proposal | None = None, proposer: str = MODEL, log=print) -> dict:
    before = {r["orbit"] for h in list_hunts() for r in h["distinct_relations"]}
    if proposal is None:
        log(f"asking {MODEL} for proposals...")
        proposal = propose_api(summary_for_prompt())
    results = []
    for fp in proposal.families[:3]:
        log(f"\n[{fp.name}] {fp.rationale}")
        try:
            fam = validate(fp)
        except ValueError as e:
            log(f"  skipped: {e}")
            results.append({"family": fp.model_dump(), "error": str(e)})
            continue
        h = run_hunt(fam, M=fp.mobius, log=log)
        new = sorted(set(h["orbits"]) - before)  # distinct up to shift and scaling
        before |= set(new)
        results.append({"family": fp.model_dump(), "hunt_id": h["id"], "n_candidates": h["n_candidates"],
                        "verified": h["target"]["count"]["verified"], "probable": h["target"]["count"]["probable"],
                        "null_hits": h["null"]["count"]["verified"] + h["null"]["count"]["probable"],
                        "new_distinct": len(new), "new_orbits": new})
        log(f"  new distinct identities this family: {len(new)}")
    rec = {"round": len(history()) + 1, "proposer": proposer, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "notes": proposal.notes, "results": results,
           "score": sum(r.get("new_distinct", 0) for r in results)}
    with (evolve_dir() / "rounds.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"\nround {rec['round']} score (new distinct identities): {rec['score']}")
    return rec


def load_proposal(path: str) -> Proposal:
    return Proposal.model_validate_json(Path(path).read_text(encoding="utf-8"))
