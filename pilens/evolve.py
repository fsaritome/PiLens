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
MAX_DEG_A, MAX_DEG_B, MAX_COEF = 4, 8, 10**6
MAX_PSLQ = 3000


class FamilyProposal(BaseModel):
    name: str = Field(description="short descriptive name")
    rationale: str = Field(description="why this region might hold new identities with pi")
    a: list[list[int]] = Field(description="[lo, hi] for each coefficient of a(n), constant term first")
    b: list[list[int]] = Field(description="[lo, hi] for each coefficient of b(n), constant term first")
    mobius: int = Field(default=6, description="max |coefficient| in the target (p0 + p1 pi)/(q0 + q1 pi), 2..12")


class Proposal(BaseModel):
    families: list[FamilyProposal] = Field(description="1 to 3 families to hunt next")
    notes: str = Field(description="what you learned from the results so far, in 2-4 sentences")


SYSTEM = f"""You are steering a search for continued-fraction formulas for pi, in the spirit of the Ramanujan Machine and FunSearch.

Objects: generalised continued fractions a(0) + b(1)/(a(1) + b(2)/(a(2) + ...)) with integer polynomial terms a(n), b(n). A *family* is a set of [lo, hi] ranges, one per polynomial coefficient (constant term first); every combination is evaluated.

Judge (mechanical, not you): each fraction's limit is matched against (p0 + p1*pi)/(q0 + q1*pi) with |p|,|q| <= M (M = 2..12) from a table, and fractions that converge fast (12+ digits after 30 terms) and miss the table also get an integer-relation search (PSLQ) for the same form with coefficients up to 10^6 (at most {MAX_PSLQ:,} such fractions per family are checked; larger fast families are sampled). Every match is re-computed at 600-bit precision to depth 16,000. The same search runs against a random constant as a control; a hit there would be a false positive.

GOAL: find a new SPEED CLASS. Each verified identity gets a fingerprint (delta, r) computed on its simplest form at depth 2000: r = convergence rate (the error falls like exp(-r n)) and delta = the irrationality-measure exponent of its convergents. Formulas that are the same up to the known equivalences (index shift, scaling, polynomial equivalence a -> c(n) a, b -> c(n) c(n-1) b, coboundaries, Mobius maps) share delta, and r up to a small integer factor (folds). The Ramanujan Machine group's collection of 149 published pi formulas plus their conservative matrix field fall into the classes listed below; everything this search has found so far falls into three of them. A round scores one point per identity whose (delta, r) matches NO known class (|delta difference| <= 0.015 and r equal or in ratio 2, 3, 4 counts as a match; slow, power-law convergence, r < 0.05, never scores). New cores in known classes are logged but score nothing.

Useful theory: if deg b = 2 deg a with leading coefficients alpha (of a) and beta (of b), the convergents grow like the roots of x^2 - alpha x - beta, and r = ln|lambda_big/lambda_small|, so the leading coefficients choose the speed; the lower coefficients decide whether the limit is related to pi at all. Equal-magnitude roots (alpha^2 + 4 beta = 0) give slow power-law convergence. Known fast pi formulas often come from hypergeometric series: their b(n) factors into linear factors like (2n-1), (3n-1), (4n-1), and a(n) is a fixed polynomial combination; the speed class reflects the series' ratio z (r = ln|lambda ratio| relates to z).

Constraints: degree of a(n) <= {MAX_DEG_A}, degree of b(n) <= {MAX_DEG_B}, |coefficients| <= {MAX_COEF:,}, at most {MAX_SIZE:,} combinations per family (product of range widths). Fix leading coefficients (lo = hi) to target a speed and vary the lower ones. The sign symmetry a -> -a is removed automatically."""


def evolve_dir() -> Path:
    d = data_dir() / "evolve"
    d.mkdir(parents=True, exist_ok=True)
    return d


def history() -> list[dict]:
    f = evolve_dir() / "rounds.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary_for_prompt(max_identities: int = 80, detailed_hunts: int = 30) -> str:
    """What Claude sees each round. Kept bounded: the last hunts in detail, older ones
    as one-liners, identities grouped by core (after removing shifts, scalings and
    polynomial equivalence transforms)."""
    hunts = list_hunts()
    lines = [f"## Hunts so far: {len(hunts)}"]
    old, recent = hunts[:-detailed_hunts], hunts[-detailed_hunts:]
    if old:
        lines.append("Older hunts (ranges, then distinct cores found):")
        for h in old:
            lines.append(f"- a {h['family']['a']} b {h['family']['b']} M={h.get('mobius_M', 6)}: "
                         f"{len(h.get('cores', []))} cores")
    lines.append("Recent hunts:")
    for h in recent:
        t, n = h["target"]["count"], h["null"]["count"]
        lines.append(
            f"- {h['family']['name'] or h['id']}: a {h['family']['a']}, b {h['family']['b']}, "
            f"M={h.get('mobius_M', 6)}; {h['n_candidates']:,} fractions, {h['n_eligible']:,} converged; "
            f"pi: {t['verified']} verified, {t['probable']} probable, {len(h.get('cores', []))} distinct cores; "
            f"control hits: {n['verified'] + n['probable']}")
    cores: dict[str, list] = {}
    for h in hunts:
        if h["constant"] != "pi":
            continue
        for r in h["distinct_relations"]:
            cores.setdefault(r.get("core", r["orbit"]), []).append(r)
    lines.append(f"\n## Distinct identity cores with pi: {len(cores)} (showing up to {max_identities}; "
                 f"'variants' = fractions that reduce to this core)")
    for core, rs in list(cores.items())[:max_identities]:
        r = rs[0]
        lines.append(f"- core {core}: e.g. a(n) = {r['a_str']}, b(n) = {r['b_str']} -> {r['relation']} "
                     f"[{r['status']}, {r['rate']}], variants {len(rs)}{'  (classical)' if r['known'] else ''}")
    ratings = data_dir() / "ratings.jsonl"
    if ratings.exists():
        rs = [json.loads(x) for x in ratings.read_text(encoding="utf-8").splitlines() if x.strip()]
        fr = [r for r in rs if r.get("kind") == "formula"]
        if fr:
            lines.append("\n## Human ratings of identities")
            lines += [f"- {r['key']}: {'+1' if r['rating'] > 0 else '-1'} {r.get('note', '')}" for r in fr[-30:]]
    lines += _class_summary()
    hist = history()
    old_h = [r for r in hist if r.get("scoring") != "speed classes"]
    new_h = [r for r in hist if r.get("scoring") == "speed classes"]
    if old_h:
        lines.append("\n## Earlier round scores (scored by new cores, before the speed-class goal): "
                     + ", ".join(f"r{r['round']}={r['score']}" for r in old_h[-40:]))
    if new_h:
        lines.append("\n## Round scores (new speed classes / new cores): "
                     + ", ".join(f"r{r['round']}={r['score']}/{r.get('new_cores', 0)}" for r in new_h))
        lines.append("Fingerprints of identities found in the speed-class rounds (relation: delta, r):")
        for r in new_h[-10:]:
            for x in r["results"]:
                cs = x.get("classes") or []
                if cs:
                    lines.append(f"- r{r['round']} {x['family']['name']}: " + "; ".join(
                        f"{c['relation']}: {c['delta']:.3f}, {c['r']:.3f}" for c in cs[:6] if c["delta"] is not None))
    for r in hist[-5:]:
        lines.append(f"\n## Round {r['round']} notes ({r['proposer']}): {r['notes']}")
    return "\n".join(lines)


def _class_summary() -> list[str]:
    """The known speed classes (published formulas + PiLens finds), fastest first."""
    from .compare import class_table, known_classes

    known = known_classes()
    if not known:
        return []
    groups = class_table(known)

    def short(c, n=90):
        s = gcf.poly_str(tuple(c))
        return s if len(s) <= n else s[:n] + "…"

    lines = [f"\n## Known speed classes: {len(groups)} (from {len(known)} formulas; 'literature' = published, "
             "'cmf' = Ramanujan Machine matrix field, 'PiLens' = found here). r, delta, count, sources, "
             "degrees, simplest example"]
    for g in groups:
        r = "slow (power law)" if g["r"] is None or g["r"] < 0.05 else f"r={g['r']:.3f}"
        d = "?" if g["delta"] is None else f"{g['delta']:.3f}"
        lines.append(f"- {r}, delta={d}, {g['count']}x, {'+'.join(sorted(g['sources']))}, deg {len(g['a']) - 1}/"
                     f"{len(g['b']) - 1}: a = {short(g['a'])}, b = {short(g['b'])} -> {g['limit'][:60]}")
    return lines


LAST_USAGE: dict = {}
PRICE_IN, PRICE_OUT = 4.0, 20.0  # USD per million tokens, claude-opus-5-5
EFFORT = "high"
MAX_TOKENS = 32000  # 16000 cut off ~15% of answers (stop_reason=max_tokens)


def propose_api(summary: str) -> Proposal:
    import anthropic

    client = anthropic.Anthropic()
    # Streamed: with high effort an answer can be long, and the SDK refuses non-streamed
    # requests whose max_tokens could take more than ~10 minutes.
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": summary + "\n\nPropose the next 1-3 families."}],
        output_format=Proposal,
    ) as stream:
        resp = stream.get_final_message()
    u = resp.usage
    LAST_USAGE.clear()
    LAST_USAGE.update(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                      est_cost_usd=round((u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT) / 1e6, 4))
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


def _known_cores() -> set[str]:
    return {r.get("core", r["orbit"]) for h in list_hunts() if h["constant"] == "pi"
            for r in h["distinct_relations"]}


def _append(rec: dict) -> None:
    with (evolve_dir() / "rounds.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def total_spend() -> float:
    return round(sum((r.get("usage") or {}).get("est_cost_usd", 0) for r in history()), 4)


def run_round(proposal: Proposal | None = None, proposer: str = MODEL, log=print) -> dict:
    from .compare import classify, is_slow, known_classes

    before = _known_cores()
    known = known_classes()
    usage = None
    if proposal is None:
        log(f"asking {MODEL} (effort {EFFORT}) for proposals...")
        proposal = propose_api(summary_for_prompt())
        usage = dict(LAST_USAGE)
        LAST_USAGE.clear()
        log(f"  tokens: {usage['input_tokens']:,} in, {usage['output_tokens']:,} out "
            f"(~${usage['est_cost_usd']:.3f})")
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
        new = sorted(set(h.get("cores", [])) - before)  # up to shift, scaling, polynomial equivalence
        before |= set(new)
        new_classes = []
        for c in h.get("classes", []):
            if is_slow(c["r"]) or c["delta"] is None:
                continue
            if classify(c["delta"], c["r"], known) is None:
                new_classes.append(c)
            known.append({"source": f"PiLens {h['id']}", **{k: c[k] for k in ("a", "b", "delta", "r")},
                          "limit": c["relation"]})
        results.append({"family": fp.model_dump(), "hunt_id": h["id"], "n_candidates": h["n_candidates"],
                        "verified": h["target"]["count"]["verified"], "probable": h["target"]["count"]["probable"],
                        "null_hits": h["null"]["count"]["verified"] + h["null"]["count"]["probable"],
                        "new_distinct": len(new), "new_cores": new, "new_classes": new_classes,
                        "classes": [{k: c[k] for k in ("relation", "delta", "r")} for c in h.get("classes", [])]})
        log(f"  new distinct cores: {len(new)}; NEW SPEED CLASSES: {len(new_classes)}"
            + "".join(f"\n    !! {c['relation']}  delta={c['delta']:.4f} r={c['r']:.4f}  a={c['a']} b={c['b']}"
                      for c in new_classes))
    rec = {"round": len(history()) + 1, "proposer": proposer, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "notes": proposal.notes, "results": results, "usage": usage, "scoring": "speed classes",
           "new_cores": sum(r.get("new_distinct", 0) for r in results),
           "score": sum(len(r.get("new_classes", [])) for r in results)}
    _append(rec)
    log(f"\nround {rec['round']} score (new speed classes): {rec['score']}; new cores: {rec['new_cores']}")
    return rec


def run_budget(budget_usd: float, max_rounds: int = 1000, log=print) -> float:
    """Run API rounds until the estimated spend of this run reaches ``budget_usd``.

    Stops early enough that the next round cannot overshoot (one call is capped at
    MAX_TOKENS output tokens plus input). Returns the estimated spend.
    """
    spent, failures = 0.0, 0
    reserve = round(MAX_TOKENS * PRICE_OUT / 1e6 + 0.15, 2)  # worst case for one call
    for i in range(max_rounds):
        if spent + reserve > budget_usd:
            log(f"budget reached: ~${spent:.2f} of ${budget_usd:.2f}")
            break
        log(f"\n===== run round {i + 1}  (spent so far ~${spent:.2f} of ${budget_usd:.2f}) =====")
        try:
            rec = run_round(log=log)
            spent += (rec.get("usage") or {}).get("est_cost_usd", 0)
            failures = 0
        except Exception as e:  # keep going on transient problems, but account for the cost
            failures += 1
            cost = LAST_USAGE.get("est_cost_usd", reserve) if LAST_USAGE else reserve
            spent += cost
            _append({"round": len(history()) + 1, "proposer": MODEL, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "notes": f"round failed: {type(e).__name__}: {e}", "results": [],
                     "usage": dict(LAST_USAGE) or {"est_cost_usd": cost, "estimated": True}, "score": 0})
            LAST_USAGE.clear()
            log(f"round failed ({failures} in a row): {type(e).__name__}: {e}")
            if failures >= 3:
                log("3 failures in a row, stopping")
                break
            time.sleep(30)
    log(f"\nrun finished: ~${spent:.2f} spent; total recorded over all runs ~${total_spend():.2f}")
    return spent


def load_proposal(path: str) -> Proposal:
    return Proposal.model_validate_json(Path(path).read_text(encoding="utf-8"))
