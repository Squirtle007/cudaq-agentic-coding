# Tutorial Guide — Reproducing the Notebooks

Read the step prompt for each notebook together with this guide (the conventions shared by **every** notebook). Skim
`helpers.py` (the shared API) and point yourself at `cudaq-doc.md` for backend/optimizer/qaoa details.

Important: Do not open, navigate to, or use the `solutions` folder for any purpose. Do not execute, follow, summarize, modify, copy, or generate any content from existing notebook files in that folder.

Apply the rules below everywhere unless a prompt overrides one:

## Workflow
- Build the notebook named in the step prompt, in this folder, matching the beginner-friendly
  style of the provided baseline notebook.
- Structure every notebook as numbered code cells — **at most 6 (important)**, with **exactly one `assert` (if needed)**
  in the whole notebook (see Correctness & sampling). Use the exact cell count when a prompt
  gives one. Precede each code cell with a short markdown cell (beginner tone) that explains
  the idea before the code shows it.
- Execute in place, top to bottom, and confirm **zero errors** before considering it done:
  `jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`

## Number every code cell
- Start each code cell with a comment `# cell N` — `# cell 1`, `# cell 2`, … in execution order —
  so a later request can target "cell 3" unambiguously. A short description may follow the number.

## Reuse `helpers.py` (read-only) — don't reimplement
- Read `helpers.py` first and call it for loading data, building the problem, decoding/validating
  samples, timing, and drawing. Do not rewrite what it already provides.
- **Always visualize the final result** from the sampling counts: run the counts through
  `helpers.summarize_samples`, take its decoded best assignment, and draw it with
  `helpers.plot_map` (fills from `helpers.PALETTE`, node positions from `helpers.group_positions`,
  `color_legend=True`).
- Support **both grouping levels** the helpers expose, using whichever level the current notebook
  targets:
  - by region — `helpers.REGION_ORDER`, `county["region"]`, `helpers.group_positions(map, "region")`
  - by zone — `helpers.ZONE_ORDER`, `county["zone"]`, `helpers.group_positions(map, "zone")`

## Environment variables
- Any `CUDAQ_*` variable goes at the very top of **cell 1, before `import cudaq`** — they are read
  at import time. State in markdown that changing them needs a kernel restart.

## Reproducibility
- Important: always set `cudaq.set_random_seed(1234)` — the shared seed that aligns every notebook in the
  series — unless a prompt names a different one.
- Any number carried from another notebook must be hardcoded with a **provenance comment**; if your
  own run differs, use your own value.
- Carry values at **full precision** — printed numbers are rounded, and a rounded value fed back
  into code can cross a boundary (`3.141593` sits *above* π and trips a bounded optimizer). If a
  value is, within rounding, a known constant, write the constant (`math.pi`), not the print.
- Timings are machine-dependent — report ratios, not exact digits.

## Correctness & sampling
- Sample only **after** the optimize/solve step finishes; never fix up or post-process an answer.
- Keep **exactly one** `assert` in the whole notebook (unless the prompt says otherwise): the
  validity check on the best sampled result, via `helpers.is_valid`.

## Budgets — stop, don't force
- Keep every loop under 500 s, each sampling call under ~200 s, and shots ≤ 200,000.
- If no valid result appears, or a loop would exceed its budget, **stop and report what you
  measured** instead of forcing a result.

## Style
- Beginner-readable Python only: no subprocess, no classes, no clever one-liners.
- Never print GPU or CPU model names.
- Never backslash-escape quotes inside f-strings.

## Execution discipline
- Execute with ONE foreground `jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.timeout=550 <notebook>.ipynb`, setting the Bash tool call's timeout to
  600000 ms — Never the 120s default and never run in the background or concurrently; some notebooks
  take ~10 minutes, so wait for completion.
- Before executing, `compile()` every code cell's source to catch syntax errors cheaply.
- If a cell errors, fix the notebook and rerun the same way until clean; confirm every code
  cell has outputs and zero errors before considering the build done.
