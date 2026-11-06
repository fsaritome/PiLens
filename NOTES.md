# PiLens notes: results so far (phase 1, 2026-11-06)

All numbers below come from `pilens run ... --controls 500` (300 for `pi_cf` and `bbp`)
and `scripts/calibrate.py`. Each cell is the lens p-value and, in brackets, the BH
q-value across the 7 lenses of that run. A lens is **flagged** when q < 0.05.

## 1. The judge works (validation)

Planted structure, n = 20,000 decimal digits, 300 controls, 7 lenses:

| planted | 0.5% | 1% | 2% | 3% | 5% |
|---|---|---|---|---|---|
| bias (extra zeros) | – | – | walk, grid, pairs, spiral | + glyph | + recurrence |
| period 7 | – | – | pairs | grid, pairs, spiral | + walk, glyph |
| Markov (repeat previous) | – | – | grid, chord, recurrence, pairs, spiral | same | same |

* At 3% strength every kind is caught by at least 3 lenses. `pytest -m slow` checks the same thing with other seeds.
* At 1% and n = 20,000 nothing is caught. With n = 100,000, Markov at 1% is caught
  by 5 of 7 lenses (grid, chord, recurrence, pairs, spiral; q = 0.01). Power grows
  with n, as it should.
* **False positives:** on 20 fresh random sequences, 1 had a lens flagged (glyph,
  q = 0.047). That is 5%, the family-wise rate expected at q < 0.05 under the null.
* Different lenses see different things. The walk sees bias (drift) but not Markov
  repeats. Glyph tiles are weak everywhere. The grid's period scan and the pair
  transforms are the most sensitive general-purpose detectors.

## 2. Pi (and friends)

| run | walk | grid | chord | recurrence | pairs | glyph | spiral |
|---|---|---|---|---|---|---|---|
| pi, base 10, n=10^5 | .62 (.87) | .27 (.75) | .32 (.75) | 1 (1) | .50 (.87) | .17 (.75) | 1 (1) |
| pi, base 10, n=10^6 | 1 (1) | .52 (1) | .28 (.99) | 1 (1) | 1 (1) | .21 (.99)\* | 1 (1)\* |
| pi, base 4, n=10^5 | 1 (1) | 1 (1) | .024 (.17) | 1 (1) | .83 (1) | 1 (1) | 1 (1) |
| pi, base 4, n=10^6 | 1 (1) | .50 (1) | 1 (1) | 1 (1) | .65 (1) | .65 (1)\* | 1 (1)\* |
| pi, base 16, n=10^5 | 1 (1) | .23 (.54) | .50 (.59) | .50 (.59) | .35 (.59) | .19 (.54) | .16 (.54) |
| e, base 10, n=10^5 | .74 (.92) | .54 (.92) | 1 (1) | .77 (.92) | .37 (.92) | .61 (.92) | .79 (.92) |
| sqrt2, base 10, n=10^5 | .70 (.81) | .40 (.69) | 1 (1) | .55 (.77) | .19 (.45) | .06 (.21) | .02 (.14) |
| pi CF terms, n=2·10^4 (CF null) | .68 (1) | 1 (1) | .94 (1) | 1 (1) | 1 (1) | .45 (1) | .91 (1) |
| **positive:** BBP series terms | .007 (.01) | .007 (.01) | .007 (.01) | .007 (.01) | .007 (.01) | .007 (.01) | .007 (.01) |
| **positive:** planted Markov 1%, n=10^5 | .45 (.45) | .004 (.01) | .004 (.01) | .004 (.01) | .004 (.01) | .30 (.34) | .004 (.01) |

\* glyph uses only the first 128² digits and spiral the first 255² digits, so their
n = 10^6 rows look at the same digits as the n = 10^5 rows.

**Verdict: no lens finds anything in pi, e or sqrt(2).** The smallest q for any of
them is 0.14. Single p-values around 0.02 (pi base-4 chord, sqrt(2) spiral) are what
the ~56 lens tests on these sequences give you by chance, and the correction removes them. Pi's continued
fraction is indistinguishable from the CF of a random real. The positive controls
light up immediately: the BBP series terms (|z| in the hundreds to tens of thousands)
and a planted 1% Markov dependence. So the null result for pi isn't because the
judge is blind.

## 3. Walks as chains

Scaling exponent ν (R ∝ N^ν) and self-intersection (the fraction of steps landing on an
already visited cell), pi against controls (mean ± sd):

| walk | ν (pi) | ν (controls) | self-intersection (pi) | (controls) |
|---|---|---|---|---|
| base 4, n=10^5 | 0.501 | 0.498 ± 0.020 | 0.756 | 0.763 ± 0.021 |
| base 4, n=10^6 | 0.494 | 0.499 ± 0.011 | 0.796 | 0.797 ± 0.015 |
| base 10, n=10^5 | 0.526 | 0.498 ± 0.019 | 0.734 | 0.761 ± 0.023 |
| base 10, n=10^6 | 0.491 | 0.499 ± 0.011 | 0.789 | 0.796 ± 0.015 |

Pi's walks are freely jointed chains: ν = 1/2, and about 3 out of 4 steps land
somewhere already visited. Unfolded proteins (self-avoiding in 3-D) have
ν ≈ 0.59; a 2-D self-avoiding walk has 0.75. The resemblance to an amino-acid chain
is real, but it's the resemblance any random walk has, not something specific to pi.

## 4. Ideas for next steps

* Phase 2: a Claude loop writes lenses into `lenses_ext/`, judged on *fresh* digits
  (designed on digits 1..10^6, tested on 10^6..2·10^6), with the BH family growing
  with every lens tried. Rate pictures in the gallery to steer it.
* Look where structure is known to live: generalised continued fractions, and
  partial sums or digit-extraction residues of BBP-type series. A lens that
  "rediscovers" BBP structure would be a good sanity target for phase 2.
* More controls (2,000+) if the number of lenses grows past ~12, since the smallest
  p is limited by 1/(N+1).

---

# Phase 2: formula hunt (2026-11-06)

Phase 1 showed that pi's digits look random under every lens. Phase 2 searches where pi
has structure: continued fractions a(0) + b(1)/(a(1) + b(2)/(a(2) + …)) with polynomial
a(n), b(n), whose limit is (p0 + p1·π)/(q0 + q1·π). Every match is re-computed at 600-bit
precision to depth 16,000, and the same search is run against a random constant as the
control.

## Totals

| | |
|---|---|
| continued fractions scanned (π) | 3,067,769 (1,479,498 in rounds 0–4) |
| distinct identities with π (up to shift and scaling) | **45**: 21 by round 4 (19 verified to ~175 digits, 2 probable at 13–16 digits), +24 in the API rounds 5–7 (all verified) |
| matches against the random constant | 2 raw coincidences, **0 confirmed** |
| same "linear b" family against e (contrast) | 0 identities with π, **9 distinct with e** (50 fractions) |

## Rounds

The proposals for rounds 1–4 are in `proposals/`. They were written by Claude Opus 5.5
in the build session, because this machine has no API key. With a key, `pilens evolve`
asks `claude-opus-5-5` the same question through the API.

| round | idea | new distinct identities |
|---|---|---|
| 0 | classic preset: a linear, b quadratic, small coefficients | 11 (baseline) |
| 1 | the convergence boundary a1² + 4·b2 ≈ 1; Euler-type constant a; quadratic a with quartic b | 2 |
| 2 | each known pattern with much wider lower-order coefficients | 0 |
| 3 | the same patterns with a larger target table (\|coef\| ≤ 12) | **8** |
| 4 | the 3n pattern even wider; linear b; a1 = 1 | 0 |
| 5 | *(Claude via API)* other integer-root pairs of x² − a1·x − b2: (1,3), (2,3), (1,4) | 0 |
| 6 | *(API)* keep root ratio 2 and −1; half-integer shifts (6n \| −8n²); degree-lifted quadratic/quartic versions | **23** |
| 7 | *(API)* half-integer trick for the 4/π family; both tricks combined; b(0) ≠ 0 | 1 |

Rounds 5–7 ran through `pilens evolve --rounds 3` with `claude-opus-5-5`. Claude's own
reasoning is in `data/evolve/rounds.jsonl`. Its key step was a hypothesis that it tested
and rejected: in round 5 it guessed that fast π identities need integer roots of
x² − a1·x − b2, and three root pairs gave zero hits. In round 6 it narrowed this down to
the *root ratio* (2, i.e. hypergeometric argument 1/2, the 3n family, and −1, the
arctan(1) / 4/π family), and searching those regimes one degree up found 23 new ones.

Caveat on the 24 from rounds 6–7: these quadratic/quartic fractions mostly factor, e.g.
a = (2n+1)(n+1), b = n³(n+1). That is the shape an equivalence transform with c_n = n + k
produces from the linear families, so they are very likely the known families in a new
form, not new mathematics. PiLens doesn't detect that equivalence yet (see Next ideas).

What the loop learned:
* Identities come in a few parametric families. The dominant one is
  a(n) = 3n + i, b(n) = −2n² + j·n + k, with limits like 2/π, 4/π, 2/(π+2), 6/(3π−8).
  Most of them lie on a straight line in the (a0, b1) plane: see the landscape on the
  "r3 3n-type" page of the gallery. The next is 2n+1 | n² (4/π), and the slow Euler-type
  fractions with constant a and b = 4n² + ….
* Widening the coefficients found nothing new (round 2). Widening the *targets* did
  (round 3): the missing identities had limits with larger coefficients, like
  12/(3π − 4).
* Fractions with linear b(n) never give π (round 4), but the same 218k fractions give 9
  distinct identities for e. This fits theory: linear-b fractions converge to ratios of
  Bessel-type functions, which is e's world, while π needs quadratic b(n).
* The two quadratic-a / quartic-b identities have the same limits as members of the 3n
  family (2/(π+2), −2/(π+4)), so they are very likely the same fractions in disguise
  (an equivalence transform with c_n ∝ n).

**Novelty:** none claimed. The 2n+1 | n² and 3n | −2n² families are in the literature
(classical continued fractions and the Ramanujan Machine papers). PiLens only removes
shift and scaling duplicates and has no database of known formulas beyond three classics.

## Next ideas

* Run `pilens evolve` with an API key and let Claude steer for 10+ rounds.
* Allow a(n) of degree 2 with b(n) of degree 4 over wider ranges, with Mobius ≤ 12; that
  is where Apéry-style formulas live.
* Add an equivalence test beyond shift and scaling (compare limits under general Mobius
  maps), so "distinct" means distinct up to every known transformation.
* Look for the *line*: the 3n family's identities lie on a line in coefficient space, so a
  conjectured closed form for the whole line would be a real, checkable statement.
