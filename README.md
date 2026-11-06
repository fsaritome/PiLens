# PiLens

*Look at the digits of pi through many "lenses", and check honestly whether any picture
shows something that random digits don't.*

## The idea

Written out, pi is an endless string of digits. The hunch behind this project: arrange
that string the right way (as a walk, a coloured grid, a chord diagram, a tiling of
symbols instead of digits) and some visual structure might show up that a list of
digits hides.

The catch is that the human eye finds patterns in pure noise. So every picture here
comes with a **judge**: the same lens is applied to hundreds of random sequences of the
same length and base, and pi only "has structure" under a lens if its picture is
measurably different from those controls, after correcting for how many lenses we tried.

## Expected outcome

**The judge will almost certainly find nothing in the digits of pi**, and that is the
expected result, not a failure.

* Pi is conjectured to be *normal* in every base: every block of k digits appears with
  frequency B^-k. This hasn't been proved, but every statistical test run on trillions
  of digits has passed. A normal sequence has no structure that a fixed, finite lens
  could see.
* The lenses here are fixed, simple functions of the digits. A lens that found
  structure in pi's digits would amount to a new statistical test that pi fails,
  which would be a major result.
* Pi *does* have structure, but in other representations: series and generalized
  continued fractions with regular patterns (Leibniz, Nilakantha, Brouncker, the
  Ramanujan Machine's conjectures), and the BBP formula, which computes the n-th
  hexadecimal digit without the earlier ones. The simple continued fraction of pi
  (7, 15, 1, 292, ...) has no known pattern and looks like a typical real number's
  (Gauss–Kuzmin statistics). PiLens includes some of these as sources, so you can see
  what a picture of real structure looks like.

So what PiLens gives you is: (1) a lot of pictures of pi, (2) an honest, calibrated
answer to "is this pattern real?", and (3) the setup for phase 2, where an AI writes
new lenses and the judge plus your ratings decide which ones are worth keeping.

## Method

```
sources  ──►  lens(seq, base, **params) ──►  image + feature vector
                       │
controls ──►  same lens (≥200×)        ──►  judge: z-scores, empirical p, classifier, BH
                       │
                    gallery (pi vs control, scores, 👍/👎 → data/ratings.jsonl)
```

**Sources** ([pilens/sources.py](pilens/sources.py), cached in `data/cache/`)

| source | what | null used for the controls |
|---|---|---|
| `pi`, `e`, `sqrt2` | digits in any base 2–36 (gmpy2/MPFR; 10^6 decimal digits of pi in under 1 s) | i.i.d. uniform digits |
| `pi_cf` | simple continued-fraction terms of pi (exact via interval arithmetic), symbol = min(a, B) − 1 | CF terms of a uniformly random real |
| `leibniz`, `nilakantha`, `bbp` | k-th series term, symbolised by the mantissa of its magnitude | uniform (these are obvious positives) |
| `random` | seeded random digits | uniform |
| `planted_bias/period/markov` | random digits with weak structure at an adjustable strength: extra zeros, a weak period-7 pattern, or "repeat the previous digit" | uniform |

**Lenses** ([pilens/lenses.py](pilens/lenses.py)): `walk` (turtle walk, angle 2πd/B),
`grid` (colour grid with row width, plus a scan over row widths 2–64 for periodicity),
`chord` (Krzywinski-style chord diagram of consecutive digits), `recurrence`
(recurrence plot plus lag profile), `pairs` (pair/difference transforms for lags 1–8),
`glyph` (each digit becomes a small black-and-white tile; connected-component
statistics), and `spiral` (an Ulam-style square spiral, shipped as an example plug-in in
[lenses_ext/](lenses_ext/)).

**Judge** ([pilens/judge.py](pilens/judge.py)). For a lens and a sequence:

1. Metrics = the lens's own features + generic image metrics: compressed size (zlib),
   pixel entropy, 2-D FFT power spectrum (radial slope, peak-to-median ratio, flatness).
2. The same lens runs on N random controls (default 500) from the source's null.
   Each metric gets a z-score and an empirical two-sided p-value.
3. **One p-value per lens (max-T):** T = max |z| over the lens's metrics, ranked among
   the controls' own T. This corrects exactly for the many metrics inside one lens.
4. **Classifier:** pi is cut into 32 chunks and a control into 32 chunks; a ridge
   (regularised LDA) linear classifier tries to tell them apart from the chunk metrics,
   with 5-fold CV and a label-permutation p-value.
5. Lens p = min(1, 2·min(p_maxT, p_clf)); then **Benjamini–Hochberg** q-values across all
   lenses in the run. "Flagged" means q < 0.05.

The smallest p you can get is about 1/(N+1), so with m lenses you need N > 2m/0.05
controls for any lens to be flagged at all. The default of 500 covers about 12 lenses.

**Validation.** `pytest -m slow` (and `pilens validate`) checks that the judge flags
each planted kind (bias, period, Markov) at 3% strength with n = 20,000, and doesn't
flag a fresh random sequence. [scripts/calibrate.py](scripts/calibrate.py) measures
power against strength and the false-positive rate over many random seeds. The results
are in [NOTES.md](NOTES.md).

### Walks, chains and proteins

A base-4 walk of pi is a random walk on the square lattice, and it really does look like
a polymer chain. Polymer physics gives the yardstick, the scaling exponent ν in
R ∝ N^ν (end-to-end size against chain length):

| chain | ν |
|---|---|
| freely jointed chain / random walk (any dimension) | 1/2 |
| self-avoiding walk, 2-D | 3/4 |
| self-avoiding walk, 3-D (≈ unfolded protein) | 0.588 |
| compact globule (≈ folded protein), 3-D | 1/3 (in 2-D: 1/2) |
| biased / ballistic walk | 1 |

The `walk` lens measures ν from non-overlapping windows of length 2^k, the
self-intersection rate (the fraction of steps landing on a lattice cell that was already
visited), the radius of gyration and the asphericity. Pi's walks give ν ≈ 0.5 and heavy
self-intersection, just like random controls: in polymer terms, a freely jointed chain
rather than a protein. The 0.59 for unfolded proteins is a 3-D value. In 2-D the
self-avoiding value is 0.75.

## Precedents

* **Walking on real numbers.** F. J. Aragón Artacho, D. H. Bailey, J. M. Borwein,
  P. B. Borwein, *The Mathematical Intelligencer* 35 (2013). Base-4 walks of pi (up to about
  10^11 base-4 digits), compared with pseudo-random walks and other constants, using walk
  statistics such as expected distance and fractal dimension. This is the closest
  ancestor of the `walk` lens and of the "compare with controls" discipline.
* **Martin Krzywinski's pi art** (Circos chord diagrams and posters of pi's digits for
  Pi Day): the inspiration for the `chord` and `grid` lenses.
* **The Ramanujan Machine.** G. Raayoni et al., *Nature* 590 (2021). Automated search
  for continued-fraction formulas for constants like pi and e. It finds structure in a
  *representation* (generalised continued fractions), which is where structure is known
  to live.
* **FunSearch.** B. Romera-Paredes et al., *Nature* 625 (online Dec 2023, print 2024). An LLM writes programs, an
  automatic evaluator scores them, and the best ones are fed back. Phase 2 of PiLens
  follows this pattern, with lenses as the programs and the judge plus human ratings as
  the evaluator.
* **BBP formula.** D. Bailey, P. Borwein, S. Plouffe (1997): hexadecimal digits of pi
  without the preceding ones. Bailey and Crandall linked BBP-type formulas to the
  normality question.

## Usage

```bash
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

```bash
pilens lenses                                   # list lenses and their parameters
pilens digits --base 10 --n 1000000             # compute + cache digits
pilens run --lens walk --base 4 --n 100000      # judge one lens, save to the gallery
pilens run --lens all --base 10 --n 100000      # every lens, BH across them
pilens run --lens grid --param width=70 --base 10
pilens run --lens all --source planted_markov --strength 0.01   # see what a hit looks like
pilens validate                                 # planted vs random sanity check
pilens serve                                    # gallery on http://127.0.0.1:8765/
pytest                                          # all tests (the slow validation takes about a minute)
```

`run` writes `data/runs/<source>_b<base>_n<n>/` (JSON + PNGs + `summary.json`). BH is
recomputed over every lens in that directory, so the correction always matches the
number of lenses tried for that run. The gallery has a page per lens with pi next to a
control, every metric's z and p, and 👍/👎 buttons with an optional note. Ratings are
appended to `data/ratings.jsonl` together with the lens, its parameters and its scores,
for phase 2 to use.

## Writing a lens (and phase 2)

A lens is a pure function. Drop a file into `lenses_ext/` and it is picked up by the
CLI, the judge and the gallery:

```python
import numpy as np
from pilens.lenses import LensResult, lens

@lens("my_lens", "one-line description")
def my_lens(x: np.ndarray, base: int, *, width=128):   # every param has a default
    img = ...                                            # HxW floats in [0,1], or HxWx3 uint8
    return LensResult(img, {"some_feature": float(...)}) # '_name' = informational, not tested
```

The rules are in the docstring of [pilens/lenses.py](pilens/lenses.py). Phase 2 (not built
yet) will have a Claude loop write such files, run `pilens run --lens <new>`, and use
`data/ratings.jsonl` plus the judge's scores to decide what to try next. The judge's
multiple-testing correction has to grow with the number of lenses tried, and generated
lenses must be judged on fresh digits (e.g. digits 10^6 to 2·10^6) rather than the
digits they were designed on.

## Layout

```
pilens/sources.py   digits, CF, series, controls, planted structure
pilens/lenses.py    lens interface, registry, plug-in loader, built-in lenses
pilens/judge.py     metrics, controls, max-T, classifier, BH
pilens/store.py     run/ratings persistence, PNG rendering
pilens/server.py    FastAPI gallery
pilens/cli.py       `pilens` command
lenses_ext/         plug-in lenses (example: spiral.py)
scripts/            calibration
tests/              pytest (-m slow = judge validation)
data/               cache, runs, ratings (git-ignored)
```
