# CUDA-Q Agentic Coding Bootcamp

In this hands-on bootcamp you drive a coding agent that writes, runs, and verifies
GPU-accelerated quantum programs with CUDA-Q!🚀

## What's provided

| File | What it is |
|---|---|
| `cudaq-doc.md` | a local CUDA-Q reference — kernels, backends, optimizers, performance switches; point your coding agent at it with `@cudaq-doc.md` |
| `tutorial-guide.md` | the build conventions every agent-built notebook follows — cell numbering, helpers reuse, reproducibility, budgets, execution discipline; point your coding agent at it with `@tutorial-guide.md` |
| `cudaq_basics.ipynb` | a hands-on CUDA-Q fundamentals notebook — states, gates, entanglement, noise, a small variational loop; you run this one yourself (Warm-up 0) |
| `00_notebook.ipynb` | the prebuilt QAOA baseline notebook (COBYLA) — **read and run it before Step 1**; every later notebook is based on it |
| `helpers.py` | shared beginner-friendly functions: map loading/drawing, QUBO building, spin conversion, decoding/validation, timing |
| `data/taiwan_map_xy.json` | a simplified schematic of Taiwan's 19 main-island counties (polygons, centroids, region and zone tags, adjacency lists) |

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.12, CUDA-Q 0.15.0 (preinstalled source build). One NVIDIA GPU is required.
Follow the detailed [installation guide](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html) and the system [prerequisites](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility).

## How the bootcamp works

1. **Warm-up 0:** run `cudaq_basics.ipynb` yourself, top to bottom, to meet CUDA-Q hands-on.
2. **Warm-ups 1–2:** meet your coding agent — have it read the two reference docs, then have it
   build and run its first GPU program. Copy each prompt below into the agent as-is.
3. **Main project:** open and run `00_notebook.ipynb`, the QAOA baseline every later notebook
   improves on. Then, for each of Steps 1–4, copy the prompt into your coding agent, let it
   build and execute the notebook, check the expected outcome, and move on. Each prompt is
   self-contained.

---

## Warm-up 0 — Meet CUDA-Q, no agent yet

**Goal:** get a feel for what the agent will be writing for you. Open `cudaq_basics.ipynb` and run it top to bottom: it serves as the CUDA-Q fundamental, walking from single-qubit states and gates through entanglement and noise to a small variational optimization.

---

## Warm-up 1 — Have the agent read the references

**Goal:** the two markdown files in this folder are the agent's textbook (`cudaq-doc.md`) and
rulebook (`tutorial-guide.md`) for everything that follows. Before asking for any code, have the
agent read both and tell you what it found — you confirm it starts from the right knowledge, and
it keeps that knowledge in context.

```text
Read @cudaq-doc.md and @tutorial-guide.md, then give me a short plain-language summary of what
each one covers. No code or files yet — just the wrap-up.
```

---

## Warm-up 2 — First GPU program: a 20-qubit GHZ state

**Goal:** run the full agentic loop once — prompt in, working GPU program out — on the
"hello, world" of entanglement. A GHZ state puts all qubits into a single superposition of
all-zeros and all-ones, so every measurement comes back as one of exactly those two strings.

Example prompt -- just copy it to your terminal:
```text
Write a simple script ghz.py that uses CUDA-Q to prepare a 20-qubit GHZ state on the GPU
("nvidia" target), sample 1000 shots, and print the counts. Refer to @cudaq-doc.md if needed,
then run it and show me the output.
```

**Expected outcome:** `ghz.py` runs on the GPU and the counts show only two results — all
zeros and all ones, about half each.

---

## Agentic coding with CUDA-Q — Advancing QAOA simulation

This is the core session of the bootcamp: a self-contained, natural-language prompt goes in, and a working, verified program comes out—powering a full quantum-application build. You will co-work with agents: they read the prompt, write and execute each notebook, and check every result before you move on.

Across five Jupyter notebooks, you will solve a single problem—coloring a Taiwan map so that no two neighboring areas share a color—using the Quantum Approximate Optimization Algorithm (QAOA). Along the way, you scale from a 16-qubit exact state-vector simulation to 36-qubit tensor-network simulations with CUDA-Q’s built-in optimization techniques, all on a single NVIDIA GPU.

| Notebook | Optimizer | Backend | The improvement it teaches |
|---|---|---|---|
| 00 (provided) | COBYLA | `qpp-cpu` vs `nvidia` `fp32` | the baseline: full QAOA workflow, CPU-vs-GPU timing |
| 01 (you build) | Nelder–Mead | `nvidia` `fp32` | optimizer swap + gate-fusion settings + per-iteration GPU timing |
| 02 (you build) | Nelder–Mead | `tensornet` `fp32` | scaling to 36 qubits using tensor-network method |
| 03 (you build) | Nelder–Mead | `tensornet` `fp32` | contraction-path reuse: speedup with a simple setup |
| 04 (you build) | Nelder–Mead | `tensornet-mps` `fp32` | a fast approximate solver (χ=16), trading accuracy for speed |

Notebook 00 is pre-built as a baseline: the full QAOA workflow — turn
the map into a math problem (QUBO → spin Hamiltonian), build the circuit, optimize, sample,
verify, and draw the answer back on the map. Each subsequent notebook retains this skeleton and adds incremental optimizations, leveraging CUDA-Q’s built-in features to advance the quantum simulation.

---

## Step 1 — Swap the optimizer, set the GPU switches, compare against the baseline

**Goal:** make the cheapest big improvement first: on the same 50-evaluation budget, swapping
the classical optimizer from COBYLA to Nelder–Mead drops the energy far lower and increases the
valid-coloring rate.

Example prompt:
```text
Build and execute `01_notebook.ipynb` here, applying tutorial-guide.md conventions, from
`00_notebook.ipynb` (read it): reuse its 16-qubit region problem and p=1 kernel exactly
(A=4.0/B=1.0, only 1-/2-qubit gates), seed 1234, cudaq.set_target("nvidia", option="fp32").

Structure (5 code cells):
1. FIRST, BEFORE `import cudaq`:
   os.environ["CUDAQ_FUSION_MAX_QUBITS"] = "3"
   os.environ["CUDAQ_FUSION_NUM_HOST_THREADS"] = "4"
   Rebuild problem.
2. The QAOA kernel and energy(params) from 00.
3. Optimize: cudaq.optimizers.NelderMead, num_iterations=50, initial_parameters=[0.1,0.1],
   timed. Each energy() call prints its GPU milliseconds alongside the energy; print best
   energy, total time. Nelder-Mead lands far below 00's COBYLA on the same 50-evaluation
   budget.
4. Sample 20,000 shots after optimization; summarize_samples; the single is_valid assert.
   Valid rate far above 00's.
5. Draw coloring (region level, unit_labels on). Compare best energy and valid rate vs 00.
```

**Expected outcome:** `01_notebook.ipynb` runs end to end: the new optimizer clearly beats
notebook 00 on the same budget, and the map comes out validly colored.

---

## Step 2 — Scale to 36 qubits with tensor networks and an XY-mixer circuit

**Goal:** scale from 16 to 36 qubits — hit a memory wall: a 36-qubit state vector
needs 2^36 × 8 bytes ≈ 550 GB, more than most single GPUs, so we switch to the
tensor-network method. With the CUDA-Q `tensornet` backend, we can push further by expanding 
the coloring problem from four regions to nine zones on the map, bringing it closer to a real-world scenario.

Here, we build QAOA with an [XY mixer](https://arxiv.org/abs/1904.09314) to improve optimization. The mixer is the trick: plain rx mixing wanders over all 2^36 bitstrings, and almost none of them are
valid colorings. The XY circuit instead starts each zone in a W state — its single excitation
shared equally across the zone's four color qubits — and mixes by swapping that excitation
around the zone, never changing the counts of 1.
```text
Build and execute 02_notebook.ipynb here from notebooks 00/01 (read them; keep Nelder-Mead).

Problem: 9 zones (helpers.ZONE_ORDER, load_map()["zone_edges"]), 4 colors -> 36 qubits,
A=8.0/B=1.0. cudaq.set_target("tensornet", option="fp32"), deliberately NO env vars;
cudaq.set_random_seed(1234). Markdown: why tensornet — 36 qubits = 2^36x8 = 550 GB, beyond
most single GPU.

Structure (<=6 code cells, exactly ONE assert):
1. QUBO via helpers; print term counts.
2. ONE kernel, xy_kernel (not plain rx), 1-/2-qubit gates only:
   - W-state per zone: x(q[base]); for k in 0..2:
     ry.ctrl(W_ANGLES[k], q[base+k], q[base+k+1]) then x.ctrl(q[base+k+1], q[base+k]);
     W_ANGLES = [2.0943951023931953, 1.9106332362490186, 1.5707963267948966].
   - 00's cost layer: rz for Z terms, cx-rz-cx for ZZ.
   - XY ring mixer over ring pairs (0,1),(1,2),(2,3),(3,0):
     XX: h,h / cx / rz(2*beta) / cx / h,h
     YY: rx(+pi/2),rx(+pi/2) / cx / rz(2*beta) / cx / rx(-pi/2),rx(-pi/2).
3. Optimize: Nelder-Mead, max_iterations=5, start [0.1,0.1], timed; print best energy,
   angles, seconds/evaluation (total/5). No energy calls outside these 5.
4. SHOTS=2000; sample best angles ONLY after optimization; helpers.summarize_samples; THE
   assert: helpers.is_valid(best).
5. Map: counties by zone PALETTE color, zone graph (helpers.group_positions(taiwan,"zone")),
   color legend; per-eval + per-shot seconds; theta* (6 decimals).
```

**Expected outcome:** `02_notebook.ipynb` runs end to end at 36 qubits: valid colorings show
up in the sample, and the final map shows all 19 counties colored by zone.

---

## Step 3 — Reuse contraction paths: several-times-cheaper evaluations

**Goal:** stop paying for planning twice. A tensor-network evaluation has two parts — first
*plan* the contraction path (like choosing the order of a chain of matrix multiplications),
then *contract* — and since the QAOA loop never changes the circuit's shape, only the angle
values, replanning on every evaluation is pure waste. Two variables enable that:
`CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE` caches the contraction path, and
`CUDAQ_TENSORNET_NUM_HYPER_SAMPLES` lets the planner search for a better path.

```text
Build and execute 03_notebook.ipynb here from 02_notebook.ipynb (read it; reuse
problem/xy_kernel exactly — A=8.0/B=1.0, Nelder-Mead, seed 1234, tensornet fp32).

Structure (<=6 code cells, exactly ONE assert):
1. FIRST cell, BEFORE `import cudaq`:
   os.environ["CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE"] = "TRUE"
   os.environ["CUDAQ_TENSORNET_NUM_HYPER_SAMPLES"] = "32"
   Rebuild problem.
2. Same xy_kernel and energy(params).
3. Retrieve the elapsed time from 02, or rerun its optimization if no timing result is available,
   SAME size: start [0.1,0.1], max_iterations=5; print best energy, angles,
   elapsed vs 02's total (hardcoded, provenance) — print speedup.
4. SHOTS=2000; sample best angles (post-optimization); summarize_samples; THE assert:
   helpers.is_valid(best).
5. Map (as 02) + table: seconds/evaluation and optimize seconds (02 vs this); speedup+verdict;
   theta* (6 decimals) AND exact energy (04 needs them).

Executing takes ~5 min. Run nbconvert as ONE foreground Bash call with explicit
timeout=600000 ms (NEVER the 120 s default — it backgrounds and dies); after it exits,
confirm every cell has outputs and zero errors BEFORE reporting executed=true.
```

**Expected outcome:** `03_notebook.ipynb` runs the same optimization with contraction path reuse and finishes noticeably faster, and the map is again validly colored.

---

## Step 4 — A fast approximate solver: Matrix product state

**Goal:** By leveraging matrix product states (MPS) with a lower bond dimension, we can dramatically reduce the memory footprint while retaining exact verification of every sampled result. With a manageable accuracy trade-off, the CUDA-Q's `tensornet-mps` backend can unlock problem sizes that other approaches cannot.

```text
Build and execute 04_notebook.ipynb here from 03_notebook.ipynb (read it; same problem and
xy_kernel, A=8.0/B=1.0, Nelder-Mead, seed 1234).

Structure (5 code cells, exactly ONE assert):
1. FIRST cell, BEFORE `import cudaq`:
   os.environ["CUDAQ_MPS_MAX_BOND"]="16"      (default 64)
   os.environ["CUDAQ_MPS_ABS_CUTOFF"]="1e-4"  (default 1e-5)
   cudaq.set_target("tensornet-mps", option="fp32"); rebuild the zone problem.
   Markdown, ONE honest disclosure: the MPS energy readout is untrustworthy at this
   compression; judge by sampled colorings, not the readout.
2. Same xy_kernel and energy(params).
3. Nelder-Mead from [0.1,0.1] FRESH (never 03's final angles).
   max_iterations=5; timed; print seconds/evaluation and best readout.
4. SHOTS=2000; sample ONCE at optimized angles (post-optimization), timed;
   summarize_samples; THE assert: helpers.is_valid(best).
5. Map (counties by zone) + timing table, ONLY time rows: seconds/evaluation (03's vs
   chi=16's), ratio; verdict: sampled quality matches the exact backend at
   at-or-below cost, memory far smaller, readout untrustworthy — answers verified exactly
   after sampling.
```

**Expected outcome:** `04_notebook.ipynb` with the `tensornet-mps` backend still delivers a
validly colored map at a similar speed while using a tiny fraction of the memory.

---
