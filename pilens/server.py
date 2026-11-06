"""Gallery: one page per (run, lens) with pi next to a control, the scores, and thumbs up/down.

Ratings go to ``data/ratings.jsonl`` (one JSON object per line) for phase 2.
"""

from __future__ import annotations

import html
import json
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import store

app = FastAPI(title="PiLens gallery")

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--card:#fff;--line:#e4e2dc;--acc:#b4441c;--good:#2f7d32;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececec;--mut:#9a9aa0;--card:#1d1d20;--line:#2d2d31;--acc:#f08a5d;--good:#7bc47f;--bad:#f2867e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 60px}a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}.mut{color:var(--mut)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}th{font-weight:600;color:var(--mut);font-size:13px}
.sig{color:var(--acc);font-weight:600}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}
@media (max-width:700px){.pair{grid-template-columns:1fr}}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
figure img{width:100%;image-rendering:pixelated;border-radius:4px;display:block}figcaption{font-size:13px;color:var(--mut);margin-top:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0;overflow-x:auto}
.rate{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button{font:inherit;padding:7px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.up:hover{border-color:var(--good)}button.down:hover{border-color:var(--bad)}input[type=text]{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);min-width:220px;flex:1}
.nav{display:flex;justify-content:space-between;gap:10px;margin-top:6px}.thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.thumbs a{display:block;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px;font-size:13px;color:var(--fg)}.thumbs img{width:100%;aspect-ratio:1;object-fit:contain;image-rendering:pixelated}
code{font-size:13px}
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><main>{body}</main></body></html>""")


def _fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{nd}f}"
    return html.escape(str(v))


def _rating_counts() -> Counter:
    c: Counter = Counter()
    for r in store.load_ratings():
        c[(r["run"], r["key"], r["rating"])] += 1
    return c


@app.get("/", response_class=HTMLResponse)
def index():
    runs = store.list_runs()
    rc = _rating_counts()
    parts = ["<div class='nav'><span></span><a href='/formulas'>Formula hunt (phase 2) &rarr;</a></div>"
             "<h1>PiLens gallery</h1><p class='mut'>Each run is one source at one base and length. "
             "p_lens is the lens-level empirical p-value against random controls; q is the "
             "Benjamini-Hochberg q-value across all lenses of that run. q &lt; 0.05 is highlighted.</p>"]
    if not runs:
        parts.append("<div class='card'>No runs yet. Try <code>pilens run --lens all --base 10</code>.</div>")
    for d in runs:
        recs = store.load_run(d)
        if not recs:
            continue
        rows = []
        for r in sorted(recs, key=lambda r: r["p_lens"]):
            q = r.get("q_lens")
            top = store.top_metric(r)
            cls = " class='sig'" if q is not None and q < 0.05 else ""
            up, down = rc[(d.name, r["key"], 1)], rc[(d.name, r["key"], -1)]
            rows.append(f"<tr><td><a href='/run/{d.name}/{r['key']}'>{html.escape(r['key'])}</a></td>"
                        f"<td>{_fmt(r['p_maxT'])}</td><td>{_fmt(r.get('clf_acc'), 2)}</td>"
                        f"<td>{_fmt(r['p_lens'])}</td><td{cls}>{_fmt(q)}</td>"
                        f"<td>{html.escape(top.get('name', ''))} ({top.get('z', 0):+.1f})</td>"
                        f"<td>{up}&#128077; {down}&#128078;</td></tr>")
        thumbs = "".join(f"<a href='/run/{d.name}/{r['key']}'><img loading='lazy' src='/img/{d.name}/{r['key']}_pi.png' "
                         f"alt='{html.escape(r['key'])}'>{html.escape(r['key'])}</a>" for r in recs)
        parts.append(f"<h2>{html.escape(d.name)}</h2><div class='card'><table><tr><th>lens</th><th>p_maxT</th>"
                     f"<th>clf acc</th><th>p_lens</th><th>q (BH)</th><th>largest |z| metric</th><th>ratings</th></tr>"
                     f"{''.join(rows)}</table></div><div class='thumbs'>{thumbs}</div>")
    return page("PiLens gallery", "".join(parts))


@app.get("/run/{run}/{key}", response_class=HTMLResponse)
def lens_page(run: str, key: str):
    d = store.runs_dir() / run
    f = d / f"{key}.json"
    if not f.resolve().is_relative_to(store.runs_dir().resolve()) or not f.exists():
        raise HTTPException(404)
    r = json.loads(f.read_text())
    keys = [x["key"] for x in store.load_run(d)]
    i = keys.index(key)
    prev_k, next_k = keys[i - 1], keys[(i + 1) % len(keys)]
    rows = []
    for m in sorted(r["metrics"], key=lambda m: -abs(m["z"] or 0)):
        cls = " class='sig'" if m["p"] is not None and m["p"] < 0.01 else ""
        rows.append(f"<tr><td>{html.escape(m['name'])}</td><td>{_fmt(m['value'], 4)}</td>"
                    f"<td>{_fmt(m['ctrl_mean'], 4)} &plusmn; {_fmt(m['ctrl_sd'], 3)}</td>"
                    f"<td{cls}>{(m['z'] or 0):+.2f}</td><td>{_fmt(m['p'])}</td></tr>")
    q = r.get("q_lens")
    verdict = ("<span class='sig'>differs from random controls after correction</span>" if q is not None and q < 0.05
               else "indistinguishable from random controls")
    rc = _rating_counts()
    clf = (f"classifier accuracy {_fmt(r['clf_acc'], 2)} (chance 0.5), p = {_fmt(r['clf_p'])}"
           if r.get("clf_acc") is not None else "classifier off")
    body = f"""
<div class="nav"><a href="/">&larr; all runs</a><span><a href="/run/{run}/{prev_k}">&lsaquo; {html.escape(prev_k)}</a> &nbsp;
<a href="/run/{run}/{next_k}">{html.escape(next_k)} &rsaquo;</a></span></div>
<h1>{html.escape(key)} <span class="mut">on {html.escape(r['source'])}, base {r['base']}, n = {r['n']:,}</span></h1>
<p class="mut">Verdict: {verdict}. p_maxT = {_fmt(r['p_maxT'])}; {clf}; p_lens = {_fmt(r['p_lens'])};
q (BH over {len(keys)} lens(es) in this run) = {_fmt(q)}; {r['n_controls']} controls.</p>
<div class="pair">
<figure><img src="/img/{run}/{key}_pi.png" alt="lens on {html.escape(r['source'])}"><figcaption>{html.escape(r['source'])}</figcaption></figure>
<figure><img src="/img/{run}/{key}_ctrl.png" alt="lens on a random control"><figcaption>random control (same n, base, null)</figcaption></figure>
</div>
<div class="card rate" id="rate">
<span>Does this lens show anything interesting?</span>
<button class="up" onclick="rate(1)">&#128077; yes</button><button class="down" onclick="rate(-1)">&#128078; no</button>
<input type="text" id="note" placeholder="optional note (what do you see?)">
<span class="mut" id="counts">{rc[(run, key, 1)]}&#128077; {rc[(run, key, -1)]}&#128078;</span>
</div>
<div class="card"><table><tr><th>metric</th><th>{html.escape(r['source'])}</th><th>controls mean &plusmn; sd</th><th>z</th><th>p (two-sided)</th></tr>
{''.join(rows)}</table></div>
<p class="mut">Params: <code>{html.escape(json.dumps(r['params']))}</code>. Saved {html.escape(r.get('saved', ''))},
{r['elapsed_s']:.0f}s.</p>
<script>
async function rate(v){{
  const note=document.getElementById('note').value;
  const res=await fetch('/api/rate',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{run:{json.dumps(run)},key:{json.dumps(key)},rating:v,note:note}})}});
  const j=await res.json();
  document.getElementById('counts').textContent=j.up+'\\u{{1F44D}} '+j.down+'\\u{{1F44E}} saved';
  document.getElementById('note').value='';
}}
</script>"""
    return page(f"{key} · {run}", body)


@app.get("/img/{run}/{name}")
def img(run: str, name: str):
    f = (store.runs_dir() / run / name).resolve()
    if not f.is_relative_to(store.runs_dir().resolve()) or not f.exists() or f.suffix != ".png":
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png")


class Rating(BaseModel):
    run: str
    key: str
    rating: int
    note: str = ""


@app.post("/api/rate")
def rate(r: Rating):
    if r.rating not in (1, -1):
        raise HTTPException(400, "rating must be 1 or -1")
    f = store.runs_dir() / r.run / f"{r.key}.json"
    if not f.resolve().is_relative_to(store.runs_dir().resolve()) or not f.exists():
        raise HTTPException(404)
    rec = json.loads(f.read_text())
    store.add_rating(r.run, r.key, r.rating, r.note,
                     {"lens": rec["lens"], "params": rec["params"], "source": rec["source"],
                      "base": rec["base"], "n": rec["n"], "p_lens": rec["p_lens"], "q_lens": rec.get("q_lens")})
    c = _rating_counts()
    return {"ok": True, "up": c[(r.run, r.key, 1)], "down": c[(r.run, r.key, -1)]}


@app.get("/api/ratings")
def ratings():
    return store.load_ratings()


# ---------------------------------------------------------------------------
# phase 2: formula hunt
# ---------------------------------------------------------------------------

def _hunt_counts(side: dict) -> str:
    c = side["count"]
    return f"{c['verified']} verified, {c['probable']} probable"


@app.get("/formulas", response_class=HTMLResponse)
def formulas_index():
    from .evolve import history
    from .hunt import list_hunts

    hunts = list_hunts()
    rows = []
    for h in hunts:
        nh = h["null"]["count"]["verified"] + h["null"]["count"]["probable"]
        rows.append(f"<tr><td><a href='/formulas/{h['id']}'>{html.escape(h['family']['name'] or h['id'])}</a></td>"
                    f"<td>{h['n_candidates']:,}</td><td>{h['n_eligible']:,}</td>"
                    f"<td>{html.escape(h['constant'])}</td><td>{h['target']['matches']}</td>"
                    f"<td class='sig'>{len(h.get('cores', h.get('orbits', [])))}</td>"
                    f"<td>{h['null']['matches']}</td><td>{nh}</td></tr>")
    by_const: dict[str, dict[str, list]] = {}
    for h in hunts:
        for r in h["distinct_relations"]:
            by_const.setdefault(h["constant"], {}).setdefault(r.get("core", r["orbit"]), []).append(r)
    from .novelty import load_results

    nov = load_results()

    def oeis_cell(const, core):
        n = nov.get(f"{const}: {core}")
        if not n:
            return "<span class='mut'>not checked</span>"
        ids = []
        for lk in n["lookups"].values():
            for h in lk.get("hits", [])[:2]:
                if h["id"] not in ids:
                    ids.append(h["id"])
        links = " ".join(f"<a href='https://oeis.org/{i}'>{i}</a>" for i in ids[:3])
        cls = "sig" if n["status"] == "not in OEIS" else "mut"
        return f"<span class='{cls}'>{html.escape(n['status'])}</span> {links}"

    sections = []
    for const, orbits in by_const.items():
        rows_i = []
        for orbit, rs in sorted(orbits.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            r = rs[0]
            rows_i.append(
                f"<tr><td><code>{html.escape(orbit.split(' | ')[0])}</code></td>"
                f"<td><code>{html.escape(orbit.split(' | ')[1])}</code></td>"
                f"<td><b>{html.escape(r['relation'])}</b> <span class='mut'>(e.g. a={html.escape(r['a_str'])}, "
                f"b={html.escape(r['b_str'])})</span></td><td>{len({x['a_str'] + x['b_str'] for x in rs})}</td>"
                f"<td>{html.escape(r['status'])}</td><td>{html.escape(r['rate'])}</td>"
                f"<td>{'classical' if any(x['known'] for x in rs) else ''}</td>"
                f"<td>{oeis_cell(const, orbit)}</td></tr>")
        name = "&pi;" if const == "pi" else html.escape(const)
        sections.append(f"<h2>Distinct identities with {name}: {len(orbits)}</h2><div class='card'><table>"
                        f"<tr><th>a(n)</th><th>b(n)</th><th>limit</th><th>variants</th><th>status</th>"
                        f"<th>convergence</th><th></th><th>OEIS</th></tr>{''.join(rows_i)}</table></div>")
    ident_html = "".join(sections)
    rounds = "".join(
        f"<tr><td>{r['round']}</td><td>{html.escape(r['proposer'])}</td><td>{r['score']}</td>"
        f"<td style='text-align:left'>{html.escape(r['notes'])}</td></tr>" for r in history())
    body = f"""<div class="nav"><a href="/">&larr; digit lenses (phase 1)</a></div>
<h1>Formula hunt</h1>
<p class="mut">Pi's digits look random, but pi has exact formulas. Each hunt evaluates a family of
continued fractions <code>a(0) + b(1)/(a(1) + b(2)/(a(2) + &hellip;))</code> with polynomial a(n), b(n),
matches each limit against simple expressions in &pi;, and re-checks every match at 600-bit precision.
The same search against a <b>random constant</b> is the control: it must stay at zero.</p>
<h2>Hunts</h2><div class="card"><table><tr><th>family</th><th>fractions</th><th>converged</th>
<th>constant</th><th>matches</th><th>distinct identities</th><th>control matches</th><th>control confirmed</th></tr>
{''.join(rows) or '<tr><td>none yet: <code>pilens hunt</code></td></tr>'}</table></div>
<p class="mut">"Distinct" means different after removing three trivial changes: starting the same fraction
a few terms later (n &rarr; n + k), rescaling (a &rarr; c&middot;a, b &rarr; c&sup2;&middot;b), and polynomial
"dressing" (a &rarr; c(n)&middot;a, b &rarr; c(n)c(n&minus;1)&middot;b). Each row is one such core; "variants" counts
the fractions that reduce to it. Rows can still be equivalent through deeper transformations, and "classical" marks only the few formulas in PiLens's own short list,
so an unmarked row is <i>not</i> a claim of novelty. The OEIS column comes from <code>pilens novelty</code>: it looks up the core's numerator and denominator sequences at oeis.org. "known" means a matching entry mentions the constant; "sequence in OEIS" means the sequence is catalogued but the entry does not mention the constant; "not in OEIS" means neither sequence is catalogued. That makes the identity a candidate, not proof that it is new.</p>
{ident_html}
<h2>Claude rounds</h2><div class="card"><table><tr><th>round</th><th>proposer</th><th>new identities</th>
<th style="text-align:left">notes</th></tr>{rounds or '<tr><td>none yet: <code>pilens evolve</code></td></tr>'}</table></div>"""
    return page("PiLens formula hunt", body)


@app.get("/formulas/{hid}", response_class=HTMLResponse)
def formula_hunt(hid: str):
    from .hunt import hunts_dir

    f = (hunts_dir() / hid / "hunt.json").resolve()
    if not f.is_relative_to(hunts_dir().resolve()) or not f.exists():
        raise HTTPException(404)
    h = json.loads(f.read_text(encoding="utf-8"))
    rc = _rating_counts()

    def hit_rows(hits, limit=200):
        out = []
        for r in hits[:limit]:
            key = f"{r['a_str']} | {r['b_str']}"
            cls = " class='sig'" if r["status"] == "verified" else ""
            jkey = html.escape(json.dumps(key))
            out.append(f"<tr><td><code>{html.escape(r['a_str'])}</code></td><td><code>{html.escape(r['b_str'])}</code></td>"
                       f"<td><b>{html.escape(r['relation'])}</b></td><td{cls}>{r['status']}</td><td>{r['digits']:.0f}</td>"
                       f"<td>{html.escape(r['rate'])}</td><td style='text-align:left'><code>{html.escape(r['gcf'])}</code></td>"
                       f"<td><button onclick='rateF(this,{jkey},1)'>&#128077;</button> "
                       f"<button onclick='rateF(this,{jkey},-1)'>&#128078;</button> "
                       f"<span class='mut'>{rc[(hid, key, 1)]}/{rc[(hid, key, -1)]}</span></td></tr>")
        return "".join(out)

    t, n = h["target"], h["null"]
    body = f"""<div class="nav"><a href="/formulas">&larr; all hunts</a></div>
<h1>{html.escape(h['family']['name'] or hid)}</h1>
<p class="mut">a ranges {h['family']['a']}, b ranges {h['family']['b']} (constant term first) &middot;
{h['n_candidates']:,} fractions &middot; {h['n_eligible']:,} converged to 7+ digits at depth {h['depth']} &middot;
{h['elapsed_s']}s.</p>
<div class="pair">
<figure><img src="/hunt-img/{hid}/landscape_target.png" alt="landscape vs pi"><figcaption>vs &pi;: {t['matches']} matches,
{_hunt_counts(t)}. x = {html.escape(h.get('landscape_axes', ['?', '?'])[0])}, y (up) =
{html.escape(h.get('landscape_axes', ['?', '?'])[1])}. Blue brightness = how well fractions converge there,
gold = an identity was found.</figcaption></figure>
<figure><img src="/hunt-img/{hid}/landscape_null.png" alt="landscape vs random constant"><figcaption>vs a random
constant (control): {n['matches']} matches, {_hunt_counts(n)}.</figcaption></figure>
</div>
<h2>Identities with &pi; ({len(h['distinct_relations'])} distinct, {len(t['hits'])} checked)</h2>
<div class="card"><table><tr><th>a(n)</th><th>b(n)</th><th>limit</th><th>status</th><th>digits</th><th>convergence</th>
<th style="text-align:left">first terms</th><th>rate it</th></tr>{hit_rows(t['hits'])}</table></div>
<h2>Control matches (random constant)</h2>
<div class="card"><table><tr><th>a(n)</th><th>b(n)</th><th>limit</th><th>status</th><th>digits</th><th>convergence</th>
<th style="text-align:left">first terms</th><th></th></tr>{hit_rows(n['hits'], 50) or '<tr><td>none</td></tr>'}</table></div>
<script>
async function rateF(btn,key,v){{
  const r=await fetch('/api/rate-formula',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{run:{json.dumps(hid)},key:key,rating:v}})}});
  const j=await r.json(); btn.parentElement.querySelector('span').textContent=j.up+'/'+j.down;
}}
</script>"""
    return page(f"{hid} · formula hunt", body)


@app.get("/hunt-img/{hid}/{name}")
def hunt_img(hid: str, name: str):
    from .hunt import hunts_dir

    f = (hunts_dir() / hid / name).resolve()
    if not f.is_relative_to(hunts_dir().resolve()) or not f.exists() or f.suffix != ".png":
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png")


@app.post("/api/rate-formula")
def rate_formula(r: Rating):
    from .hunt import hunts_dir

    if r.rating not in (1, -1):
        raise HTTPException(400, "rating must be 1 or -1")
    f = (hunts_dir() / r.run / "hunt.json").resolve()
    if not f.is_relative_to(hunts_dir().resolve()) or not f.exists():
        raise HTTPException(404)
    store.add_rating(r.run, r.key, r.rating, r.note, {"kind": "formula"})
    c = _rating_counts()
    return {"ok": True, "up": c[(r.run, r.key, 1)], "down": c[(r.run, r.key, -1)]}
