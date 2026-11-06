"""Export the gallery as a static, read-only website (for GitHub Pages).

Renders every page reachable from ``/`` with the FastAPI app in-process, saves pages as
``<path>/index.html`` and images as files, rewrites absolute links to relative ones (so
the site works under any URL prefix and when opened from disk) and hides the rating UI.
"""

from __future__ import annotations

import posixpath
import re
import shutil
import time
from pathlib import Path
from urllib.parse import unquote, urldefrag

LINK = re.compile(r"""(href|src)=(["'])(/(?!/)[^"']*)\2""")
STATIC_CSS = ("#rate,button[onclick^='rateF']{display:none!important}"
              ".snapshot{background:var(--card);border:1px solid var(--line);border-radius:8px;"
              "padding:6px 10px;font-size:13px;color:var(--mut);margin-bottom:10px}")


def _file_for(url_path: str) -> str:
    p = unquote(url_path).strip("/")
    if posixpath.splitext(p)[1]:  # an asset such as .png
        return p
    return posixpath.join(p, "index.html") if p else "index.html"


def _relink(html: str, page_file: str) -> str:
    here = posixpath.dirname(page_file) or "."

    def sub(m):
        url, frag = urldefrag(m.group(3))
        rel = posixpath.relpath(_file_for(url), here)
        return f"{m.group(1)}={m.group(2)}{rel}{'#' + frag if frag else ''}{m.group(2)}"

    return LINK.sub(sub, html)


def export_site(out: Path, log=print) -> int:
    from fastapi.testclient import TestClient

    from .server import app

    client = TestClient(app)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    banner = (f"<div class='snapshot'>Read-only snapshot of the PiLens gallery, exported {stamp}. "
              f"Code and how to run it yourself: <a href='https://github.com/fsaritome/PiLens'>"
              f"github.com/fsaritome/PiLens</a>.</div>")
    todo, seen, n = ["/"], {"/"}, 0
    while todo:
        url = todo.pop()
        r = client.get(url)
        if r.status_code != 200:
            log(f"  skip {url}: HTTP {r.status_code}")
            continue
        f = out / _file_for(url)
        f.parent.mkdir(parents=True, exist_ok=True)
        if "text/html" in r.headers.get("content-type", ""):
            html = r.text
            for m in LINK.finditer(html):
                u = urldefrag(m.group(3))[0]
                if u.startswith("/api/") or u in seen:
                    continue
                seen.add(u)
                todo.append(u)
            html = html.replace("</style>", STATIC_CSS + "</style>", 1).replace("<main>", "<main>" + banner, 1)
            f.write_text(_relink(html, _file_for(url)), encoding="utf-8")
        else:
            f.write_bytes(r.content)
        n += 1
    (out / ".nojekyll").write_text("", encoding="utf-8")
    log(f"exported {n} files to {out}")
    return n
