"""Command line: ``pilens run | validate | digits | lenses | serve``."""

from __future__ import annotations

import argparse
import ast
import sys
import time

from . import sources
from .judge import bh, default_jobs, judge
from .lenses import LENSES, load_plugins


def _parse_params(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def _lens_list(spec: str) -> list[str]:
    load_plugins()
    if spec == "all":
        return list(LENSES)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for n in names:
        if n not in LENSES:
            sys.exit(f"unknown lens {n!r}; known: {', '.join(LENSES)}")
    return names


def cmd_run(a) -> None:
    from .store import save_result, update_summary

    lenses = _lens_list(a.lens)
    params = _parse_params(a.param)
    t = time.time()
    seq = sources.get_source(a.source, a.base, a.n, seed=a.source_seed, strength=a.strength)
    print(f"source {seq.name} base {seq.base} n={len(seq)} (null: {seq.null}) "
          f"ready in {time.time() - t:.1f}s; {a.controls} controls, jobs={a.jobs}")
    d = None
    for name in lenses:
        lp = params if len(lenses) == 1 else {}
        res, ip, ic = judge(name, seq, n_controls=a.controls, params=lp, seed=a.seed,
                            classifier=not a.no_classifier, k_chunks=a.chunks, n_perm=a.perm,
                            jobs=a.jobs, return_images=True)
        d = save_result(res, ip, ic, lp)
        print(f"  {res.summary()}  [{res.elapsed_s:.0f}s]")
    recs = update_summary(d)
    print(f"\nBH across the {len(recs)} lens(es) in {d}:")
    for r in sorted(recs, key=lambda r: r["p_lens"]):
        flag = "  <-- q<0.05" if r["q_lens"] < 0.05 else ""
        print(f"  {r['key']:<14} p_lens={r['p_lens']:.4f}  q={r['q_lens']:.4f}{flag}")


def cmd_validate(a) -> None:
    lenses = _lens_list(a.lens)
    seqs = [sources.planted(k, a.strength, a.n, a.base, seed=a.seed + i)
            for i, k in enumerate(sources.PLANTED_KINDS)]
    seqs.append(sources.random_seq(a.n, a.base, seed=a.seed + 99))
    print(f"validation: n={a.n} base={a.base} strength={a.strength} controls={a.controls}")
    for s in seqs:
        rs = [judge(l, s, n_controls=a.controls, classifier=not a.no_classifier, jobs=a.jobs)
              for l in lenses]
        qs = bh([r.p_lens for r in rs])
        hits = [f"{r.lens}(q={q:.3f})" for r, q in zip(rs, qs) if q < 0.05]
        print(f"  {s.name:<24} flagged by: {', '.join(hits) or '-'}")


def cmd_digits(a) -> None:
    t = time.time()
    s = sources.get_source(a.source, a.base, a.n)
    head = "".join("0123456789abcdefghijklmnopqrstuvwxyz"[d] for d in s.symbols[:60])
    print(f"{s.name} base {a.base}: {len(s)} symbols in {time.time() - t:.1f}s: {head}...")


def cmd_lenses(a) -> None:
    load_plugins()
    for name, l in LENSES.items():
        print(f"{name:<11} {l.description}\n            params: {l.defaults}  [{l.origin}]")


def cmd_serve(a) -> None:
    import uvicorn

    from .server import app

    print(f"PiLens gallery on http://{a.host}:{a.port}/")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pilens", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, n_default):
        sp.add_argument("--lens", default="all", help="lens name, comma list, or 'all'")
        sp.add_argument("--base", type=int, default=10)
        sp.add_argument("--n", type=int, default=n_default)
        sp.add_argument("--controls", type=int, default=500)
        sp.add_argument("--no-classifier", action="store_true")
        sp.add_argument("--jobs", type=int, default=default_jobs())
        sp.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run", help="judge lens(es) on a source and save to the gallery")
    common(r, 100_000)
    r.add_argument("--source", default="pi",
                   help="pi | e | sqrt2 | pi_cf | leibniz | nilakantha | bbp | random | planted_<bias|period|markov>")
    r.add_argument("--param", action="append", help="lens parameter k=v (single lens only)")
    r.add_argument("--chunks", type=int, default=32, help="chunks per class for the classifier")
    r.add_argument("--perm", type=int, default=None, help="classifier permutations (default = controls)")
    r.add_argument("--strength", type=float, default=0.03, help="for planted sources")
    r.add_argument("--source-seed", type=int, default=12345, help="for random/planted sources")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("validate", help="check the judge flags planted structure but not random")
    common(v, 20_000)
    v.set_defaults(controls=300)
    v.add_argument("--strength", type=float, default=0.03)
    v.set_defaults(fn=cmd_validate)

    dg = sub.add_parser("digits", help="compute and cache digits")
    dg.add_argument("--source", default="pi")
    dg.add_argument("--base", type=int, default=10)
    dg.add_argument("--n", type=int, default=1_000_000)
    dg.set_defaults(fn=cmd_digits)

    ls = sub.add_parser("lenses", help="list lenses and their parameters")
    ls.set_defaults(fn=cmd_lenses)

    sv = sub.add_parser("serve", help="web gallery with thumbs up/down ratings")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
