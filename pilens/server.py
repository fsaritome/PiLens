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
    parts = ["<h1>PiLens gallery</h1><p class='mut'>Each run is one source at one base and length. "
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
