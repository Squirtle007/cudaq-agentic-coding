# Coloring a Map with QAOA — a CUDA-Q GPU-Optimization Tutorial

This text is visible. <!-- This text is completely hidden on the rendered page -->

In this hands-on series you build five Jupyter notebooks that solve one problem — coloring a map
so that no two neighboring areas share a color — with the Quantum Approximate Optimization
Algorithm (QAOA), scaling from a 16-qubit exact state-vector simulation to 36-qubit tensor
networks, all on a single NVIDIA GPU. The series follows one pattern: a plain baseline, then
**one clearly named improvement per step, measured honestly**, with correctness verified after
every change.

| Notebook | Optimizer | Backend | The improvement it teaches |
|---|---|---|---|
| 00 (provided) | COBYLA | `qpp-cpu` vs `nvidia` fp32 | the baseline: full QAOA workflow, CPU-vs-GPU timing |
| 01 (you build) | **Nelder–Mead** | `nvidia` fp32 | optimizer swap + gate-fusion settings + FP32-emulation benchmark |
| 02 (you build) | Nelder–Mead | `tensornet` fp32 | scaling to 36 qubits; a circuit that encodes the constraints |
| 03 (you build) | Nelder–Mead | `tensornet` fp32 | contraction-path reuse: several-times-cheaper evaluations |
| 04 (you build) | Nelder–Mead | `tensornet-mps` fp32 | a fast approximate solver (χ=16), judged by verified answers |

The example map is Taiwan: its 19 main-island counties, grouped into 4 regions (00–01) or 9 zones
(02–04). Nothing in the code is specific to Taiwan — swap in any nodes and edges and everything
still works.

## What's provided

| File | What it is |
|---|---|
| `00_qaoa_taiwan_regions_baseline.ipynb` | the finished baseline notebook (COBYLA) — **read and run it first**; every later notebook is based on it |
| `helpers.py` | shared beginner-friendly functions: map loading/drawing, QUBO building, spin conversion, decoding/validation, timing |
| `data/taiwan_map_xy.json` | a simplified schematic of the 19 counties (polygons, centroids, region and zone tags, adjacency lists) |
| `cudaq-doc.md` | a local CUDA-Q reference — point your coding agent at it with `@cudaq-doc.md` |

Map-data provenance, in one sentence: the schematic was derived once from geoBoundaries TWN ADM1
(www.geoboundaries.org, ODbL 1.0 license, OSM-derived), polygons aggressively simplified for
illustration, offshore counties omitted, and all adjacency computed from the real county borders
and verified — the notebooks never touch the network.

## Setup

```
pip install networkx==3.6.1
```

Tested with Python 3.12, CUDA-Q 0.15.0 (preinstalled source build), numpy 2.5.0 and
matplotlib 3.11.0 (preinstalled). One NVIDIA GPU is required. All timings you will see are
machine-dependent and can vary noticeably from run to run — treat ratios, not digits, as the
message.

## How the tutorial works

1. Open and run `00_qaoa_taiwan_regions_baseline.ipynb` top to bottom, reading as you go — it is
   the baseline every other notebook improves on.
2. For each step below, copy the prompt into your coding agent, let it build and execute the
   notebook, check the expected outcome, and move on. Each prompt is self-contained.

---

## Step 1 — Swap the optimizer, set the GPU switches, benchmark FP32 emulation

**Goal:** based on notebook 00, make the tutorial's first improvements: replace COBYLA with
Nelder–Mead (a big energy win for the same budget), set two gate-fusion environment variables,
and benchmark `CUDAQ_ALLOW_FP32_EMULATED` fairly.

```text
Based on the `00` notebook, replace the COBYLA optimizer with Nelder–Mead. Configure
`CUDAQ_FUSION_MAX_QUBITS=3` and `CUDAQ_FUSION_NUM_HOST_THREADS=4`, then benchmark execution time
with `CUDAQ_ALLOW_FP32_EMULATED` disabled and enabled. Refer to @cudaq-doc.md for adopting the
optimization techniques (optimizers, gate fusion, FP32 emulation).

Concretely: build and execute a Jupyter notebook named 01_statevector_gpu_optimization.ipynb in
this folder. It is based on 00_qaoa_taiwan_regions_baseline.ipynb (present, executed — read it
and match its beginner-friendly style): the same 16-qubit problem (taiwan = helpers.load_map();
NODES = helpers.REGION_ORDER; EDGES = taiwan["region_edges"]; A=4.0, B=1.0; build_qubo ->
qubo_to_spin -> terms_to_lists), the same p=1 QAOA kernel (h init; rz / cx-rz-cx cost; rx mixer;
only 1- and 2-qubit gates), seed cudaq.set_random_seed(1234), target
cudaq.set_target("nvidia", option="fp32").

Structure (at most 6 code cells, exactly ONE assert in the whole notebook):
1. FIRST code cell, at the very top BEFORE `import cudaq`:
   os.environ["CUDAQ_FUSION_MAX_QUBITS"] = "3"
   os.environ["CUDAQ_FUSION_NUM_HOST_THREADS"] = "4"
   os.environ["CUDAQ_ALLOW_FP32_EMULATED"] = "1"
   Markdown explains each intuitively (gate fusion = merging small neighboring gates so the GPU
   makes fewer passes; host threads = CPU helpers preparing the circuit; FP32 emulation = routing
   single-precision math through fast tensor units — optional note: it only has an effect on
   Blackwell-generation GPUs, and is simply ignored on older ones so the numbers will match) and
   notes that CUDA-Q reads these at import time (restart the kernel to change them). Then rebuild
   the problem.
2. The QAOA kernel and an energy(params) function, as in notebook 00.
3. Optimize with cudaq.optimizers.NelderMead, max_iterations = 50, initial_parameters =
   [0.1, 0.1], timed with time.perf_counter. In the same cell print the head-to-head against the
   baseline: notebook 00's COBYLA reached best energy 17.305686 (its cell 3, same budget) —
   print how much lower Nelder-Mead lands (expect a large improvement, roughly down to ~10.5).
4. Sample 20,000 shots ONLY after the optimization; helpers.summarize_samples; THE one assert:
   the best sampled bitstring passes helpers.is_valid. Also print the valid-rate comparison with
   notebook 00's 0.0360%.
5. Draw the coloring with helpers.plot_map (counties filled by their region's helpers.PALETTE
   color, region graph overlaid via helpers.group_positions(taiwan, "region"), unit_labels and
   color_legend on).
6. Benchmarking section, its own markdown + cell: time 200 energy evaluations in one plain loop
   — that is the Enabled side, live. The Disabled side cannot be toggled after import, so
   hardcode it with a provenance comment: the identical 200-evaluation loop with
   CUDAQ_ALLOW_FP32_EMULATED="0" (same fusion settings) measured 0.390 s in a separate fresh
   run. Print both numbers and the Disabled/Enabled ratio with the honest framing that across
   repeated runs this ratio bounces around 1.0 (roughly 0.8x-1.25x) — a tie within run-to-run
   noise at this problem size; the fair-benchmark method is the takeaway. Then print a small
   timing table against notebook 00's hardcoded GPU numbers (0.0019 s per evaluation, 0.25 s
   optimization time; provenance comments; values from the reference run — if your notebook 00's
   summary shows slightly different numbers, use yours) with time rows and a speed ratio only,
   plus a one-line verdict: the optimizer swap is the real win; the GPU switches tie at 16 qubits.

Rules: beginner-readable Python only (no subprocess, no classes, no clever one-liners); never
print GPU or CPU model names; sampling only after optimization; every loop must stay under 500 s.
Execute the notebook in place top to bottom (jupyter nbconvert --to notebook --execute --inplace)
and confirm zero errors. If the sample contains no valid coloring, or a loop would exceed its
budget, stop and report what you measured instead of forcing a result.
```

**Expected outcome:** `01_statevector_gpu_optimization.ipynb`, executed: Nelder–Mead lands far
below COBYLA's 17.31 on the same budget, the valid-sample rate jumps by orders of magnitude, the
FP32 benchmark reads as an honest tie, and the one assert passes on a validly colored 4-region
map.

---

## Step 2 — Scale to 36 qubits with tensor networks and an XY-mixer circuit

**Goal:** based on your notebooks 00–01, move to the 9-zone map (36 qubits) on the exact
`tensornet` backend — default settings, Nelder–Mead — with a circuit that builds the
one-color-per-zone rule into the quantum state (plain QAOA would almost never sample a valid
coloring at this size).

```text
Build and execute a Jupyter notebook named 02_exact_tensornet_taiwan_counties.ipynb in this
folder. It is based on your notebooks 00/01 (read them for style; keep Nelder-Mead as the
optimizer — the lesson of notebook 01). Refer to @cudaq-doc.md for the tensornet backend.

The problem is now the 9-zone map: NODES = helpers.ZONE_ORDER (9 zones covering all 19
counties), EDGES = helpers.load_map()["zone_edges"] (14 borders), 4 colors -> 36 qubits,
penalties A=8.0, B=1.0 (max degree 5). Target cudaq.set_target("tensornet", option="fp32") —
deliberately NO environment variables in this notebook (library defaults; notebook 03 changes
exactly that). Seed 1234.

Markdown must include (beginner tone): why zones (finer maps make exact tensor networks explode
on a single mid-range GPU — measured, so we work at 9 zones); the memory math — a state vector
for the full 19-county problem (76 qubits) needs 2^76 x 8 bytes ~ 6e23 bytes, and even 36 qubits
need 2^36 x 8 bytes = 550 GB, more than any single GPU today — hence tensor networks; and one
sentence that a tensor network's cost depends on circuit structure, with one energy evaluation
costing tens of seconds here (measure, then size budgets; notebook 03 makes evaluations 3-9x
cheaper, run-depending).

Structure (at most 6 code cells, exactly ONE assert):
1. Setup: zones printed with their member counties, the 36-qubit QUBO built via helpers, term
   counts printed.
2. ONE kernel, xy_kernel — do NOT build or run the plain rx-mixer circuit: per zone, prepare a
   W state on its 4 qubits — x(q[base]) then for k in 0..2:
   ry.ctrl(W_ANGLES[k], q[base+k], q[base+k+1]) followed by x.ctrl(q[base+k+1], q[base+k]),
   with W_ANGLES = [2.0943951023931953, 1.9106332362490186, 1.5707963267948966] (angles that
   split one "1" equally over 4 qubits) — then the same rz / cx-rz-cx cost layer as notebook 00,
   then an XY ring mixer instead of rx: for every zone's ring pairs (0,1),(1,2),(2,3),(3,0),
   apply exp(-i*beta*XX) as h,h / cx / rz(2*beta) / cx / h,h and exp(-i*beta*YY) as
   rx(+pi/2),rx(+pi/2) / cx / rz(2*beta) / cx / rx(-pi/2),rx(-pi/2). Markdown, about two
   sentences of motivation: plain QAOA with the standard rx mixer would almost never sample a
   valid coloring at this size — the valid patterns are about 1 in 30 million of all 2^36
   (exactly 2,304) — so the rule is built into the circuit instead: the W start means every
   zone has exactly one color from the beginning, and the XY mixer only SWAPS that single "1"
   around the zone, so every sample automatically obeys the one-color rule — only
   neighbors-differ remains. Only 1- and 2-qubit gates anywhere.
3. Optimize: Nelder-Mead, max_iterations = 5 (each evaluation costs tens of seconds — say so),
   start [0.1, 0.1], timed; print best energy and angles.
4. Define SHOTS = 2000 and sample SHOTS shots at the best XY angles (sampling only after
   optimization; the call stays well under 200 s), helpers.summarize_samples, then THE one
   assert: best sampled bitstring passes helpers.is_valid. Markdown: about 0.9% of
   one-color-per-zone patterns are valid (2,304 of 4^9), so roughly eighteen valid colorings
   are expected.
5. Map + summary: every county filled with its ZONE's palette color (county["zone"] from the
   JSON), the 9-zone graph overlaid (helpers.group_positions(taiwan, "zone")), color legend on.
   Then a timing-only summary (XY seconds per evaluation, seconds per shot) and print
   theta* (gamma and beta to 6 decimals) with a note that notebook 03 reuses this exact problem
   and these angles.

Rules: beginner-readable Python; no environment variables; no subprocess; never print GPU/CPU
model names; loops under 500 s, sampling calls under ~200 s, shots <= 200,000. Execute in place
top to bottom, zero errors. If the XY sample contains no valid coloring, or a loop would blow
its budget, stop and report rather than forcing it.
```

**Expected outcome:** `02_exact_tensornet_taiwan_counties.ipynb`, executed: the XY circuit
yields on the order of twenty valid colorings in 2,000 shots; the final cell shows all 19
counties colored by zone and prints θ* for Step 3.

---

## Step 3 — Reuse contraction paths: several-times-cheaper evaluations

**Goal:** based on your notebook 02 — identical problem, circuit, and angles — set
`CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=TRUE` and `CUDAQ_TENSORNET_NUM_HYPER_SAMPLES=32`,
measure the repeated-evaluation speedup, and spend the savings on a real optimization run.

```text
Build and execute a Jupyter notebook named 03_tensornet_path_optimization.ipynb in this folder.
It is based on your 02_exact_tensornet_taiwan_counties.ipynb (read it; reuse its 9-zone problem,
its xy_kernel exactly, A=8.0/B=1.0, Nelder-Mead, seed 1234, tensornet fp32). Refer to
@cudaq-doc.md for the tensornet path-reuse technique (CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE,
CUDAQ_TENSORNET_NUM_HYPER_SAMPLES).

Structure (at most 6 code cells, exactly ONE assert):
1. FIRST code cell, BEFORE `import cudaq`:
   os.environ["CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE"] = "TRUE"
   os.environ["CUDAQ_TENSORNET_NUM_HYPER_SAMPLES"] = "32"
   Markdown, beginner tone: before contracting, the simulator PLANS the combination order (like
   choosing the order of a chain of matrix multiplications); by default it replans for every
   Hamiltonian term on every evaluation; PATH_REUSE caches and reuses the plans (the circuit
   SHAPE never changes between evaluations — only angle values do), and NUM_HYPER_SAMPLES makes
   the planner search more candidates. Set-before-import + restart-kernel note. Then rebuild the
   same 36-qubit zone problem. Define THETA_2 = the gamma and beta your notebook 02 printed in
   its final cell, as a hardcoded list with a provenance comment — if your beta rounds to
   3.141593 or above, nudge it just inside the optimizer's default +/- pi bounds (e.g. 3.141592)
   and say so in a comment.
2. The same xy_kernel and energy(params) as notebook 02.
3. Time ONE energy evaluation at THETA_2, then ONE repeated evaluation at slightly shifted
   angles (e.g. gamma + 0.003), printing each time separately. Compare against your notebook
   02's measured seconds-per-evaluation — its optimize cell prints it — hardcoded with a
   provenance comment, and print the speedup on repeated evaluations. Markdown: with reuse ON even the
   first call can come out cheap (the planning is amortized across the Hamiltonian's terms);
   judge settings by the REPEATED-evaluation cost — that is what the optimization loop pays.
4. Spend the savings: Nelder-Mead from initial_parameters = THETA_2 with
   max_iterations = min(15, int(400 / repeat_seconds)) — the cap keeps the loop safely
   under 500 s since repeat costs vary run to run. Print best energy, angles, elapsed seconds.
5. Define SHOTS = 2000 and sample SHOTS shots at the new best angles (after optimization only),
   summarize_samples, THE one assert: best sampled bitstring passes helpers.is_valid.
6. Map (counties colored by zone, zone graph overlaid) + a comparison table with exactly these
   rows: seconds per evaluation (your notebook 02's number vs this notebook's repeat) and
   evaluations afforded (5 vs this budget), then the speedup line and a one-line verdict. Print
   theta* (6 decimals) AND the exact best energy value with a note that notebook 04 will check
   its approximation against them.

Rules: beginner-readable Python; no subprocess; no GPU/CPU model names; loops under 500 s;
sampling under ~200 s. Execute in place, zero errors. If the sample has no valid coloring or a
loop would exceed budget, stop and report honestly.
```

**Expected outcome:** `03_tensornet_path_optimization.ipynb`, executed: repeated evaluations
several times cheaper than your notebook 02's (typically 3–9×, run-depending), a bigger
optimization within the same budget, a valid zone coloring, and θ* plus its exact energy printed
for Step 4.

---

## Step 4 — A fast approximate solver: MPS at χ=16, judged by verified answers

**Goal:** based on your notebook 03, run the identical workload on the approximate
`tensornet-mps` backend with `CUDAQ_MPS_MAX_BOND=16` and `CUDAQ_MPS_ABS_CUTOFF=1e-4` — the
sweep-chosen trade-off, its looser cutoff picked for extra speed — disclose the (intentionally)
untrustworthy energy readout, and verify every answer by decoding sampled bitstrings.

```text
Build and execute a Jupyter notebook named 04_mps_accuracy_performance.ipynb in this folder. It
is based on your 03_tensornet_path_optimization.ipynb (read it; same 9-zone problem, same
xy_kernel, A=8.0/B=1.0, Nelder-Mead, seed 1234). Refer to @cudaq-doc.md for the tensornet-mps
backend and its settings (CUDAQ_MPS_MAX_BOND, CUDAQ_MPS_ABS_CUTOFF).

Structure (at most 6 code cells, exactly ONE assert):
1. FIRST code cell, BEFORE `import cudaq`:
   os.environ["CUDAQ_MPS_MAX_BOND"] = "16"      (the sweep-chosen sweet spot; default is 64)
   os.environ["CUDAQ_MPS_ABS_CUTOFF"] = "1e-4"  (loosened from the 1e-5 default — the sweep's
   extra-speed pick)
   Then cudaq.set_target("tensornet-mps", option="fp32") and rebuild the 36-qubit zone problem.
   Markdown, beginner tone: an MPS stores the state as a chain of small tensors whose
   neighbor-connections have width chi (the bond dimension); memory is about n * 2 * chi^2 * 8
   bytes — for 36 qubits at chi=16 about 0.15 MB versus 550 GB for the full vector — but when
   the true state needs more width than chi, the simulator TRUNCATES. Include this pre-measured
   sweep summary table (settings tested at notebook 03's angles, 800 sampling shots each;
   CUDAQ_MPS_ABS_CUTOFF at 1e-5 and 1e-4; exact reference energy ~3.50): bond 2 -> energy
   readout ~158, 0 valid colorings; bond 8 -> readout ~113-124, 4 valid (0.50%); bond 16 ->
   readout 0.24 (at 1e-5) / 14.1 (at 1e-4), 9 and 6 valid (1.12% / 0.75%); bond 32 -> readout
   ~0.02 / -1.1, 9 and 7 valid; bond 64 -> readout -1.8 / -2.1, 8 and 13 valid; cost per
   evaluation rises steadily with bond. Conclusion in markdown: solution quality (valid
   colorings from sampling) collapses at chi=2, recovers by chi=8, and plateaus from chi=16 on —
   so chi=16 with the looser 1e-4 cutoff is the chosen trade-off, picked for the extra speed
   (9.3 vs 9.9 s per evaluation in the sweep) at a still-robust ~0.75% valid rate (roughly
   fifteen valid colorings in a 2,000-shot sample). ONE honest disclosure sentence with raw
   numbers: "the chi=16 energy estimate reads ~14.1 in the sweep while the exact value is
   3.50 — an intentional trade-off; we judge by the sampled colorings, and the energy readout
   should not be trusted at this compression." Hardcode THETA_3 and EXACT_E = the angles and
   exact energy printed by YOUR notebook 03's final cell (provenance comments).
2. The same xy_kernel and energy(params).
3. One timed evaluation at THETA_3: print the chi=16 energy readout next to EXACT_E and repeat
   the disclosure in one line (no accuracy gate — the trade-off is intentional and judged by
   sampled colorings).
4. End to end: Nelder-Mead from THETA_3, max_iterations = min(15, int(400 / measured seconds
   per evaluation)), timed; print the best readout with a reminder it comes from the imperfect
   meter.
5. Define SHOTS = 2000 and sample SHOTS shots ONCE at the optimized angles (sampling only
   after optimization), timed; helpers.summarize_samples; then the EXPLICIT validity step and
   THE one assert: decode the best sampled bitstring and verify helpers.is_valid. One markdown
   sentence: valid counts vary a little from run to run on this backend — that is normal.
6. Map (counties by zone) + a timing comparison table with ONLY time rows — seconds per
   evaluation (your notebook 03's repeated-evaluation number vs the chi=16 number) and the
   speed ratio — then a plain verdict: chi=16 delivers the same sampled solution quality as the
   exact simulator at lower cost per evaluation (state the ratio you measure), with a tiny
   memory footprint, while its energy readout stays untrustworthy at this compression — a deal
   worth taking only because every answer is verified exactly after sampling.

Rules: beginner-readable Python; no subprocess; no GPU/CPU model names; loops under 500 s; each
sampling call well under 200 s; shots <= 200,000 total. Execute in place, zero errors. If no
valid coloring appears in the sample, stop and report honestly.
```

**Expected outcome:** `04_mps_accuracy_performance.ipynb`, executed: the χ=16 readout is far off
the exact value (disclosed, intentional), the sample lands near the ~1% valid rate, the best
bitstring passes the explicit validity check, and the timing table shows a speed ratio near or
above 1× — state what you measure.

---

*That's the series: one problem, five notebooks, one named improvement at a time — and every
answer verified, every claim measured on your own machine.*
