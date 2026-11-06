"""Judge calibration: power vs planted strength, and false-positive rate on fresh random digits.

    python scripts/calibrate.py [--n 20000] [--controls 300] [--seeds 20]

Prints markdown tables (used in NOTES.md).
"""

import argparse

from pilens import sources as S
from pilens.judge import default_jobs, judge_many
from pilens.lenses import LENSES, load_plugins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--base", type=int, default=10)
    ap.add_argument("--controls", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--strengths", default="0.005,0.01,0.02,0.03,0.05")
    a = ap.parse_args()
    load_plugins()
    lenses = list(LENSES)
    jobs = default_jobs()
    strengths = [float(s) for s in a.strengths.split(",")]

    print(f"## Power (n={a.n}, base {a.base}, {a.controls} controls, {len(lenses)} lenses, BH q<0.05)\n")
    print("| planted | strength | lenses flagged (q) |")
    print("|---|---|---|")
    for kind in S.PLANTED_KINDS:
        for s in strengths:
            seq = S.planted(kind, s, a.n, a.base, seed=500)
            rs = judge_many(lenses, seq, n_controls=a.controls, jobs=jobs)
            hits = ", ".join(f"{r.lens} ({r.q_lens:.3f})" for r in rs if r.q_lens < 0.05) or "none"
            print(f"| {kind} | {s:g} | {hits} |", flush=True)

    print(f"\n## False positives on {a.seeds} fresh random sequences\n")
    any_hit = 0
    for seed in range(a.seeds):
        seq = S.random_seq(a.n, a.base, seed=90_000 + seed)
        rs = judge_many(lenses, seq, n_controls=a.controls, jobs=jobs, seed=seed + 1)
        hits = [r.lens for r in rs if r.q_lens < 0.05]
        any_hit += bool(hits)
        print(f"seed {seed}: min q = {min(r.q_lens for r in rs):.3f} {hits or ''}", flush=True)
    print(f"\nsequences with any lens flagged: {any_hit}/{a.seeds}")


if __name__ == "__main__":
    main()
