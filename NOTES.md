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
