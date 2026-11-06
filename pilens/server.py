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
{f'<p><a href="/run/{run}/chord/3d"><b>&rarr; view the chords in 3D</b></a></p>' if key == 'chord' else ''}
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


@app.get("/run/{run}/chord/3d", response_class=HTMLResponse)
def chord_3d(run: str):
    """The chord lens stacked in time: each chord is lifted to its position in the
    sequence, so order (not just counts) becomes visible. Source next to a control."""
    import re

    from .sources import get_source, make_control

    f = (store.runs_dir() / run / "chord.json").resolve()
    if not f.is_relative_to(store.runs_dir().resolve()) or not f.exists():
        raise HTTPException(404)
    r = json.loads(f.read_text())
    src, strength, seed = r["source"], 0.03, 12345
    m = re.match(r"^(planted_[a-z]+)_([0-9.]+)$", src)
    if m:
        src, strength = m.group(1), float(m.group(2))
    m = re.match(r"^random_s(\d+)$", src)
    if m:
        src, seed = "random", int(m.group(1))
    try:
        seq = get_source(src, r["base"], r["n"], seed=seed, strength=strength)
    except ValueError:
        raise HTTPException(404)
    ctrl = make_control(seq, 1)
    k = 2001
    data = {"base": int(r["base"]), "src": [int(v) for v in seq.symbols[:k]],
            "ctrl": [int(v) for v in ctrl[:k]], "name": r["source"]}
    name = html.escape(r["source"])
    body = f"""<div class="nav"><a href="/run/{run}/chord">&larr; back to the flat chord diagram</a></div>
<h1>Chords in 3D <span class="mut">{name}, base {r['base']}</span></h1>
<p class="mut">Same picture as the flat chord diagram: the {r['base']} symbols sit around a ring, and each pair of
consecutive symbols is joined by a curve through the middle. Here every chord is also <b>lifted to its position in
the sequence</b>: the first chord is at the bottom, later ones higher up, so the order becomes visible. Colour =
the symbol the chord starts from. Left: {name}; right: random digits (the control). If {name} hid a pattern in
time, the left tube would show bands, twists or gaps that the right one lacks. Drag to rotate, scroll to zoom.</p>
<div class="card" style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">
<label>chords <select id="nch"><option>100</option><option>300</option><option selected>600</option>
<option>1000</option><option>2000</option></select></label>
<label>height per chord <select id="hz"><option value="1" selected>1</option><option value="0.2">squashed</option>
<option value="3">stretched</option></select></label>
<label><input type="checkbox" id="sync" checked> rotate both together</label></div>
<div class="pair">
<div id="p0" class="card" style="height:600px;padding:0;overflow:hidden;margin:0"></div>
<div id="p1" class="card" style="height:600px;padding:0;overflow:hidden;margin:0"></div>
</div>
<script src="{PLOTLY}"></script>
<script>
const D = {json.dumps(data)};
const S = 18;  // points per chord
// one distinct colour per symbol (stepped colourscale)
const palette = B => {{
  const out = [];
  for (let d = 0; d < B; d++) {{
    const c = `hsl(${{Math.round(360 * d / B)}}, 75%, 58%)`;
    out.push([d / B, c], [(d + 1) / B, c]);
  }}
  return out;
}};
function traces(x, n, hz) {{
  const B = D.base, X = [], Y = [], Z = [], C = [];
  for (let i = 0; i < n; i++) {{
    // fixed point per symbol: height already shows the order, so the flat diagram's slow
    // drift of endpoints across each sector (which would look like a twist) is left out
    const t0 = 2 * Math.PI * (x[i] + 0.5) / B;
    const t1 = 2 * Math.PI * (x[i + 1] + 0.5) / B;
    const ax = Math.cos(t0), ay = Math.sin(t0), bx = Math.cos(t1), by = Math.sin(t1);
    for (let s = 0; s <= S; s++) {{
      const t = s / S, u = 1 - t;
      X.push(u * u * ax + t * t * bx); Y.push(u * u * ay + t * t * by); Z.push((i + t) * hz); C.push(x[i]);
    }}
    X.push(null); Y.push(null); Z.push(null); C.push(x[i]);
  }}
  const lab = [...Array(B).keys()];
  return [{{type: 'scatter3d', mode: 'lines', x: X, y: Y, z: Z, hoverinfo: 'skip',
            line: {{color: C, colorscale: palette(B), cmin: -0.5, cmax: B - 0.5, width: 2}}}},
          {{type: 'scatter3d', mode: 'text', hoverinfo: 'skip',
            x: lab.map(d => 1.15 * Math.cos(2 * Math.PI * (d + 0.5) / B)),
            y: lab.map(d => 1.15 * Math.sin(2 * Math.PI * (d + 0.5) / B)),
            z: lab.map(() => 0), text: lab.map(String),
            textfont: {{size: 14, color: lab.map(d => palette(B)[2 * d][1])}}}}];
}}
let syncing = false;
function draw() {{
  const n = Math.min(+document.getElementById('nch').value, D.src.length - 1);
  const hz = +document.getElementById('hz').value;
  const fg = getComputedStyle(document.body).color;
  const ax = {{visible: false}};
  [[D.src, 'p0', D.name], [D.ctrl, 'p1', 'random control']].forEach(([x, id, title]) => {{
    Plotly.react(id, traces(x, n, hz), {{
      paper_bgcolor: 'rgba(0,0,0,0)', font: {{color: fg}}, showlegend: false,
      margin: {{l: 0, r: 0, t: 30, b: 0}}, title: {{text: title, font: {{size: 14}}}},
      scene: {{xaxis: ax, yaxis: ax, zaxis: ax, aspectmode: 'manual', aspectratio: {{x: 1, y: 1, z: 1.6}},
               camera: {{eye: {{x: 1.6, y: 1.6, z: 0.9}}}}}},
    }}, {{responsive: true, displaylogo: false}});
  }});
  ['p0', 'p1'].forEach((id, k) => {{
    const el = document.getElementById(id);
    el.removeAllListeners && el.removeAllListeners('plotly_relayout');
    el.on('plotly_relayout', ev => {{
      if (syncing || !document.getElementById('sync').checked || !ev['scene.camera']) return;
      syncing = true;
      Plotly.relayout(k ? 'p0' : 'p1', {{'scene.camera': ev['scene.camera']}}).then(() => syncing = false);
    }});
  }});
}}
['nch', 'hz'].forEach(id => document.getElementById(id).onchange = draw);
draw();
</script>"""
    return page(f"chord 3D · {run}", body)


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

    from .compare import load_results as load_comparison

    cmp_ = load_comparison()

    def rm_cell(const, core):
        c = cmp_.get(core) if const == "pi" else None
        if not c:
            return "<span class='mut'>&ndash;</span>"
        if c["exact"]:
            x = c["exact"]
            where = "their unifying family (CMF)" if x["kind"] == "cmf" else "a published formula"
            tip = html.escape(f"{x['limit']}; {x['source']}")
            return f"<span class='mut' title='{tip}'>same as {where}</span>"
        strong = [m for m in c["fingerprint"] if m["rate"]]
        if strong:
            m = strong[0]
            tip = html.escape(f"delta {c['delta']:.4f} vs {m['delta']:.4f}; rate {m['rate']}; e.g. {m['limit']}")
            return f"<span class='mut' title='{tip}'>fingerprint of a known formula</span>"
        return f"<span class='sig'>{html.escape(c['status'])}</span>"

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
                f"<td>{oeis_cell(const, orbit)}</td><td>{rm_cell(const, orbit)}</td></tr>")
        name = "&pi;" if const == "pi" else html.escape(const)
        sections.append(f"<h2>Distinct identities with {name}: {len(orbits)}</h2><div class='card'><table>"
                        f"<tr><th>a(n)</th><th>b(n)</th><th>limit</th><th>variants</th><th>status</th>"
                        f"<th>convergence</th><th></th><th>OEIS</th><th>Ramanujan Machine</th></tr>"
                        f"{''.join(rows_i)}</table></div>")
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
so an unmarked row is <i>not</i> a claim of novelty. The OEIS column comes from <code>pilens novelty</code>: it looks up the core's numerator and denominator sequences at oeis.org. "known" means a matching entry mentions the constant; "sequence in OEIS" means the sequence is catalogued but the entry does not mention the constant; "not in OEIS" means neither sequence is catalogued. That makes the identity a candidate, not proof that it is new.
The Ramanujan Machine column comes from <code>pilens compare</code>, against the 149 &pi; formulas the Ramanujan
Machine group collected from the literature and the 22 representatives of their unifying family
(<a href="https://github.com/RamanujanMachine/euler2ai">euler2ai</a>, arXiv:2502.17533). "same as" means the
fraction reduces to one of theirs; "fingerprint of a known formula" means it has the same irrationality measure
&delta; and convergence speed as a known formula, the signature their paper uses to group equivalent formulas
(hover for details). Result so far: every identity found here falls into three well-known speed classes.</p>
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
{h['elapsed_s']}s. <a href="/formulas/{html.escape(hid)}/3d"><b>&rarr; view in 3D</b></a></p>
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


PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


@app.get("/formulas/{hid}/3d", response_class=HTMLResponse)
def formula_hunt_3d(hid: str):
    """The hits as points in coefficient space: pick any three varying coefficients as
    axes (and one for colour); the shadows on the walls are the flat landscape maps."""
    from .hunt import axis_label, hunts_dir

    f = (hunts_dir() / hid / "hunt.json").resolve()
    if not f.is_relative_to(hunts_dir().resolve()) or not f.exists():
        raise HTTPException(404)
    h = json.loads(f.read_text(encoding="utf-8"))
    fam = h["family"]
    dims = [{"side": s, "k": k, "label": axis_label((s, k)), "lo": r[0], "hi": r[1]}
            for s in ("a", "b") for k, r in enumerate(fam[s]) if r[0] < r[1]]

    def pts(hits):
        out = []
        for r in hits:
            if r["status"] not in ("verified", "probable"):
                continue
            out.append({"a": r["a"], "b": r["b"], "txt": f"a(n) = {r['a_str']}<br>b(n) = {r['b_str']}<br>"
                        f"limit: {r['relation']}<br>{r['status']}, {r['digits']:.0f} digits"})
        return out

    tgt, nul = pts(h["target"]["hits"]), pts(h["null"]["hits"])
    # default axes: the coefficients that vary most among the pi hits
    def spread(d):
        vals = {(p[d["side"]] + [0] * 9)[d["k"]] for p in tgt}
        return len(vals)
    order = sorted(range(len(dims)), key=lambda i: -spread(dims[i]))
    data = {"dims": dims, "target": tgt, "null": nul, "order": order}
    const = "π" if h["constant"] == "pi" else h["constant"]
    body = f"""<div class="nav"><a href="/formulas/{html.escape(hid)}">&larr; back to the flat maps</a></div>
<h1>{html.escape(fam['name'] or hid)}: 3D view</h1>
<p class="mut">Every rule this hunt tried is a point in a box of whole-number settings (the grey frame). The
<b class="sig">coloured dots</b> are the rules whose fraction turned out to equal a formula in {const}; with more
than three settings, the colour shows a fourth one. Bright red crosses would be hits on the random control number
(there should be none). Drag to rotate, scroll to zoom, hover a dot for its formula. The grey shadows on the
walls are what the flat maps show: a 3D trail seen from one side.</p>
<div class="card" style="display:flex;gap:12px;flex-wrap:wrap;align-items:center">
<label>x <select id="ax0"></select></label><label>y <select id="ax1"></select></label>
<label>z <select id="ax2"></select></label><label>colour <select id="ax3"></select></label>
<span class="mut">{len(tgt)} {const} hits, {len(nul)} control hits, {len(dims)} settings vary in this hunt</span></div>
<div id="plot" class="card" style="height:640px;padding:0;overflow:hidden"></div>
<script src="{PLOTLY}"></script>
<script>
const D = {json.dumps(data, ensure_ascii=False)};
const sel = [0,1,2,3].map(i => document.getElementById('ax'+i));
sel.forEach((s, i) => {{
  if (i === 3) s.add(new Option('(none)', -1));
  D.dims.forEach((d, j) => s.add(new Option(d.label, j)));
  s.value = i < 3 ? (D.order[i] ?? D.order.at(-1) ?? 0) : (D.order[3] ?? -1);
  s.onchange = draw;
}});
const coef = (p, d) => (p[d.side][d.k] ?? 0);
function draw() {{
  const ax = sel.map(s => +s.value);
  const css = getComputedStyle(document.body);
  const fg = css.color, line = css.getPropertyValue('--line').trim() || '#888';
  const [dx, dy, dz] = ax.slice(0, 3).map(i => D.dims[i]);
  const dc = ax[3] >= 0 ? D.dims[ax[3]] : null;
  const pts = (P, name, color) => ({{
    type: 'scatter3d', mode: 'markers', name,
    x: P.map(p => coef(p, dx)), y: P.map(p => coef(p, dy)), z: P.map(p => coef(p, dz)),
    text: P.map(p => p.txt), hovertemplate: '%{{text}}<extra></extra>',
    marker: {{size: 6, line: {{width: 0}}, symbol: color === 'gold' ? 'circle' : 'cross',
      color: dc && P.length && color === 'gold' ? P.map(p => coef(p, dc)) : color,
      colorscale: 'YlOrRd', reversescale: true,
      colorbar: dc && color === 'gold' ? {{title: {{text: 'colour = ' + dc.label.replace('coefficient of ', ''), side: 'top', font: {{color: fg, size: 12}}}}, thickness: 12, len: 0.6, tickfont: {{color: fg}}}} : undefined}},
    projection: {{x: {{show: true, opacity: 0.25, scale: 0.6}}, y: {{show: true, opacity: 0.25, scale: 0.6}}, z: {{show: true, opacity: 0.25, scale: 0.6}}}},
  }});
  // wireframe of the search box
  const [x0, x1, y0, y1, z0, z1] = [dx.lo, dx.hi, dy.lo, dy.hi, dz.lo, dz.hi];
  const E = [[[x0,y0,z0],[x1,y0,z0]],[[x0,y1,z0],[x1,y1,z0]],[[x0,y0,z1],[x1,y0,z1]],[[x0,y1,z1],[x1,y1,z1]],
             [[x0,y0,z0],[x0,y1,z0]],[[x1,y0,z0],[x1,y1,z0]],[[x0,y0,z1],[x0,y1,z1]],[[x1,y0,z1],[x1,y1,z1]],
             [[x0,y0,z0],[x0,y0,z1]],[[x1,y0,z0],[x1,y0,z1]],[[x0,y1,z0],[x0,y1,z1]],[[x1,y1,z0],[x1,y1,z1]]];
  const box = {{type: 'scatter3d', mode: 'lines', name: 'search box', hoverinfo: 'skip',
    x: E.flatMap(e => [e[0][0], e[1][0], null]), y: E.flatMap(e => [e[0][1], e[1][1], null]),
    z: E.flatMap(e => [e[0][2], e[1][2], null]), line: {{color: 'rgba(128,128,140,0.5)', width: 2}}}};
  const axis = d => ({{title: {{text: d.label}}, range: [d.lo - 0.5, d.hi + 0.5], color: fg, gridcolor: line, zeroline: false}});
  Plotly.react('plot', [box, pts(D.target, '{const} hits', 'gold'), pts(D.null, 'control hits', '#e0443c')], {{
    paper_bgcolor: 'rgba(0,0,0,0)', font: {{color: fg}}, margin: {{l: 0, r: 0, t: 0, b: 0}},
    legend: {{x: 0, y: 1}}, scene: {{xaxis: axis(dx), yaxis: axis(dy), zaxis: axis(dz), aspectmode: 'cube'}},
  }}, {{responsive: true, displaylogo: false}});
}}
draw();
</script>"""
    return page(f"{hid} · 3D", body)


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
