"""Is a found identity already known? Look its integer sequences up in the OEIS.

Every fraction a(0) + b(1)/(a(1) + b(2)/(a(2) + ...)) produces two integer sequences:
the numerators p(n) and denominators q(n) of its convergents (p(n) = a(n) p(n-1) +
b(n) p(n-2), likewise q). The OEIS (oeis.org) catalogues known integer sequences and
usually cites where they come from. If a core's sequences are in the OEIS, and
especially if the entry mentions the constant, the identity is almost certainly known.
Not finding them is weaker evidence: it means "not catalogued", not "new to mathematics".

We look up each core in its simplest form (after removing shifts, scalings and polynomial
equivalence transforms), so that one lookup covers all its variants.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import gcf
from .hunt import list_hunts
from .sources import data_dir

OEIS_URL = "https://oeis.org/search"
USER_AGENT = "PiLens/0.1 (https://github.com/fsaritome/PiLens)"
N_TERMS = 10
MENTIONS = {
    "pi": re.compile(r"\bpi\b|π", re.I),
    "e": re.compile(r"exp\(1\)|\be\b"),
    "sqrt2": re.compile(r"sqrt\(2\)|2\^\(1/2\)", re.I),
}


def novelty_dir() -> Path:
    d = data_dir() / "novelty"
    d.mkdir(parents=True, exist_ok=True)
    return d


def core_form(a, b, max_shift: int = 12) -> tuple[tuple, tuple]:
    """The simplest representative of a core: reduced, then the smallest valid shift."""
    ra, rb = gcf.reduce_equivalence(tuple(a), tuple(b))
    best = None
    for k in range(-max_shift, max_shift + 1):
        sa, sb = gcf.shift_poly(ra, k), gcf.shift_poly(rb, k)
        if any(gcf.poly_eval(sb, n) == 0 for n in range(1, 60)):
            continue
        ca, cb, _ = gcf.canonical(sa, sb)
        key = (sum(map(abs, ca)) + sum(map(abs, cb)), ca, cb)
        if best is None or key < best:
            best = key
    if best is None:
        return ra, rb
    return best[1], best[2]


def convergent_sequences(a, b, n_terms: int = N_TERMS + 4) -> tuple[list[int], list[int]]:
    """Exact integer numerators and denominators p(0..), q(0..)."""
    p_prev, q_prev = 1, 0
    p, q = gcf.poly_eval(a, 0), 1
    P, Q = [int(p)], [int(q)]
    for n in range(1, n_terms):
        an, bn = gcf.poly_eval(a, n), gcf.poly_eval(b, n)
        p, p_prev = an * p + bn * p_prev, p
        q, q_prev = an * q + bn * q_prev, q
        P.append(int(p))
        Q.append(int(q))
    return P, Q


def search_terms(seq: list[int], n: int = N_TERMS) -> list[int] | None:
    """The terms to look up: drop leading 0/±1 (they match everything), make the first
    nonzero term positive and divide out a common factor of the whole run."""
    s = list(seq)
    while s and abs(s[0]) <= 1:
        s = s[1:]
    s = s[:n]
    if len(s) < 5:
        return None
    if s[0] < 0:
        s = [-v for v in s]
    g = 0
    for v in s:
        g = math.gcd(g, v)
    if g > 1:
        s = [v // g for v in s]
    return s


class OEIS:
    """Tiny client with an on-disk cache and a polite delay between requests."""

    def __init__(self, delay: float = 1.5, cache: Path | None = None):
        self.delay = delay
        self.cache_path = cache or novelty_dir() / "oeis_cache.json"
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        self._last = 0.0

    def search(self, terms: list[int]) -> list[dict]:
        q = ",".join(str(v) for v in terms)
        if q in self.cache:
            return self.cache[q]
        wait = self.delay - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = f"{OEIS_URL}?{urllib.parse.urlencode({'q': q, 'fmt': 'json'})}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8") or "null")
        self._last = time.time()
        if isinstance(data, dict):  # older response format
            data = data.get("results") or []
        hits = [{"id": f"A{int(x['number']):06d}", "name": x.get("name", ""),
                 "text": " ".join([x.get("name", "")] + x.get("comment", []) + x.get("formula", []))[:4000]}
                for x in (data or [])[:10]]
        self.cache[q] = hits
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        return hits


def check_core(a, b, constant: str, oeis: OEIS) -> dict:
    ca, cb = core_form(a, b)
    P, Q = convergent_sequences(ca, cb)
    out = {"a": list(ca), "b": list(cb), "fraction": gcf.gcf_str(ca, cb), "lookups": {}}
    mention = MENTIONS.get(constant)
    found = cites = False
    for name, seq in (("numerators", P), ("denominators", Q)):
        terms = search_terms(seq)
        if terms is None:
            out["lookups"][name] = {"terms": seq[:N_TERMS], "skipped": "too few nontrivial terms"}
            continue
        hits = oeis.search(terms)
        c = [h["id"] for h in hits if mention and mention.search(h["text"])]
        out["lookups"][name] = {"terms": terms, "hits": [{"id": h["id"], "name": h["name"]} for h in hits],
                                "mentions_constant": c}
        found |= bool(hits)
        cites |= bool(c)
    out["status"] = "known" if cites else ("sequence in OEIS" if found else "not in OEIS")
    return out


def collect_cores(constants: tuple[str, ...] | None = None) -> dict[tuple[str, str], dict]:
    """One representative fraction per (constant, core), plus how many variants it has."""
    cores: dict[tuple[str, str], dict] = {}
    for h in list_hunts():
        if constants and h["constant"] not in constants:
            continue
        for r in h["distinct_relations"]:
            k = (h["constant"], r.get("core", r["orbit"]))
            if k not in cores:
                cores[k] = {"rep": r, "variants": 0}
            cores[k]["variants"] += 1
    return cores


def load_results() -> dict:
    f = novelty_dir() / "novelty.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def run_check(constants: tuple[str, ...] | None = None, recheck: bool = False, log=print) -> dict:
    oeis = OEIS()
    results = load_results()
    cores = collect_cores(constants)
    log(f"checking {len(cores)} cores against the OEIS ...")
    for (const, core), info in cores.items():
        key = f"{const}: {core}"
        if key in results and not recheck:
            continue
        r = info["rep"]
        try:
            res = check_core(r["a"], r["b"], const, oeis)
        except Exception as e:  # network trouble: record and move on
            log(f"  {key}: lookup failed ({type(e).__name__}: {e})")
            continue
        res.update(constant=const, core=core, relation=r["relation"], variants=info["variants"],
                   classical=bool(r.get("known")), checked=time.strftime("%Y-%m-%d %H:%M:%S"))
        results[key] = res
        ids = sorted({h["id"] for lk in res["lookups"].values() for h in lk.get("hits", [])})
        log(f"  [{res['status']}] {const}: {res['fraction']} -> {r['relation']}  {' '.join(ids[:4])}")
        (novelty_dir() / "novelty.json").write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")
    counts: dict[str, int] = {}
    for k, v in results.items():
        if k.split(":")[0] in (constants or [k.split(":")[0]]):
            counts[v["status"]] = counts.get(v["status"], 0) + 1
    log("summary: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    return results
