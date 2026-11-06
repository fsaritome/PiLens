# Plug-in lenses

Every `*.py` file here (except names starting with `_`) is imported by
`pilens.lenses.load_plugins()`. Any `@lens(...)` it defines becomes available to
`pilens run`, the judge and the gallery. `spiral.py` is the example.

Phase 2 will write generated lenses into this folder. The contract is in the docstring
of `pilens/lenses.py`: a pure, deterministic function `(x, base, **params) -> LensResult`,
every parameter has a default, no I/O.

Set `$PILENS_LENS_DIR` to load lenses from somewhere else.
