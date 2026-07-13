# CUDA-Q Agentic Coding Bootcamp

In this hands-on bootcamp you drive a coding agent that writes, runs, and verifies
GPU-accelerated quantum programs with CUDA-Q!🚀

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.12, CUDA-Q 0.15.0 (preinstalled source build). One NVIDIA GPU is required.
Follow the detailed [installation guide](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html) and the system [prerequisites](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility).

## How the bootcamp works

1. **Warm-up 0:** run `cudaq_basics.ipynb` yourself, top to bottom, to learn CUDA-Q hands-on.
2. **Warm-ups 1–2:** meet your coding agent — have it read the two references to streamline your learning experience.
3. **Agentic coding with CUDA-Q:** start from the QAOA baseline in `00_notebook.ipynb`. Then, for Steps 1-4, copy or
   customize each prompt into your coding agent and let it iteratively build, optimize, and scale the CUDA-Q program.

<br>

---

## Warm-up 0 — Meet CUDA-Q, no agent yet

**Goal:** get a feel for what the agent will be writing for you. Open `cudaq_basics.ipynb` and run it top to bottom: it serves as the CUDA-Q fundamental, walking from single-qubit states and gates through entanglement and noise to a variational optimization.

<br>

---

## Warm-up 1 — Equip the agent with CUDA-Q knowledge

**Goal:** the two markdown files in this folder are the agent's textbook (`cudaq-doc.md`) and
rulebook (`tutorial-guide.md`) for everything that follows. Before asking for any code, have the
agent read both and tell you what it found:
```text
Read @cudaq-doc.md and @tutorial-guide.md, then give me a short summary of what
each one covers. Just the wrap-up. No code or files yet.
```

<br>

---

## Warm-up 2 — Simple GHZ state

**Goal:** run the agentic flow — prompt in, working GPU program out — on the
"hello, world" of entanglement.  A GHZ state puts all qubits into a single superposition of
all-zeros and all-ones, so every measurement comes back as one of exactly those two strings.

Just copy the example prompt to your terminal:
```text
Write a simple script ghz.py using CUDA-Q to prepare a 20-qubit GHZ state on the GPU
("nvidia" target), sample 1000 shots, and print the counts. Refer to @cudaq-doc.md if needed,
then run it and show me the output.
```

**Expected outcome:** `ghz.py` runs on the GPU and the counts show only two results — all
zeros and all ones, about half each.

<br>

---

## 🖥️ Agentic coding with CUDA-Q — Advancing QAOA simulation 

This is the core session of the bootcamp: a self-contained, natural-language prompt goes in, and a working, verified program comes out—powering a full quantum-application build.You will co-work with agents, solve a single problem:

> Coloring a Taiwan map so that **no two neighboring areas share a color** using the Quantum Approximate Optimization Algorithm (QAOA)

Along the way, you scale from a 16-qubit exact state-vector simulation to 36-qubit tensor-network and matrix product state simulation with CUDA-Q’s built-in optimization techniques, advancing the quantum simulation at scale.

![](images/learning_path.png)

| Notebook | Optimizer | Backend | The improvement it teaches |
|---|---|---|---|
| 00 (provided) | COBYLA | `qpp-cpu` vs `nvidia` `fp32` | the pre-built baseline: full QAOA workflow, CPU-vs-GPU timing |
| 01 (you build) | Nelder–Mead | `nvidia` `fp32` | optimizer swap + gate-fusion settings + per-iteration GPU timing |
| 02 (you build) | Nelder–Mead | `tensornet` `fp32` | scaling to 36 qubits using tensor-network method |
| 03 (you build) | Nelder–Mead | `tensornet` `fp32` | contraction-path reuse: speedup with a simple setup |
| 04 (you build) | Nelder–Mead | `tensornet-mps` `fp32` | matrix product state approximation, trading accuracy for speed |

Other ingredients:
| File | What it is |
|---|---|
| `cudaq-doc.md` | agent-ready CUDA-Q reference for quantum programming - point your agent at it with `@cudaq-doc.md` |
| `tutorial-guide.md` | conventions that every agent-built notebook follows to ensure reproducibility |
| `helpers.py` | shared useful functions: map visualization, QUBO building, decoding/validation, etc |
| `data/taiwan_map_xy.json` | simplified schematic of Taiwan's 19 main-island counties |

<br>

---

## Step 1 — Swap the optimizer, tune the GPU, compare against the baseline

**Goal:** make the cheapest big improvement first: on the same 50-evaluation budget, swapping
the classical optimizer from COBYLA to Nelder–Mead drops the energy far lower and increases the
valid-coloring rate.

Example prompt:
```text
Build and execute `01_notebook.ipynb` here, applying tutorial-guide.md conventions, from
`00_notebook.ipynb` (read it): reuse its 16-qubit region problem and p=1 kernel exactly
(A=4.0/B=1.0, only 1-/2-qubit gates), target "nvidia" single-GPU backend, and FP32 precision.

Structure:
1. FIRST, BEFORE `import cudaq`:
   os.environ["CUDAQ_FUSION_MAX_QUBITS"] = "3"
   os.environ["CUDAQ_FUSION_NUM_HOST_THREADS"] = "4"
   Rebuild problem.
2. The QAOA kernel and energy(params) from 00.
3. Optimize: cudaq.optimizers.NelderMead, num_iterations=50, initial_parameters=[0.1,0.1],
   timed. Each energy() call prints its GPU milliseconds alongside the energy; print best
   energy, total time. Nelder-Mead lands far below 00's COBYLA on the same 50-evaluation
   budget.
4. Sample 20,000 shots after optimization; summarize_samples; perform a single assert helpers.is_valid(best) check.
   Report a valid rate well above 00’s.
5. Draw coloring (region level, unit_labels on). Compare best energy and valid rate vs 00.
```

**Expected outcome:** `01_notebook.ipynb` with the new optimizer clearly beats
notebook 00 on the same budget, and the map comes out validly colored.

<br>

---

## Step 2 — Scale to 36 qubits with tensor networks

**Goal:** scale from 16 to 36 qubits — hit a memory wall: a 36-qubit state vector
needs 2^36 × 8 bytes ≈ 550 GB, more than most single GPUs, so we switch to the
tensor-network method. With the CUDA-Q `tensornet` backend, we can push the coloring problem 
from 4 regions to 9 zones on the map, bringing it closer to a real-world scenario.

Here, we build QAOA with an [XY mixer](https://arxiv.org/abs/1904.09314) to improve optimization. 
The mixer is the trick: plain rx mixing wanders over all 2^36 bitstrings, and almost none of them are
valid colorings. The XY circuit instead starts each zone in a W state — its single excitation
shared equally across the zone's four color qubits — and mixes by swapping that excitation
around the zone, never changing the counts of 1:
```text
Build and execute 02_notebook.ipynb here from notebooks 00/01 (read them; keep Nelder-Mead).

Problem: 9 zones * 4 colors -> 36 qubits, using the zones listed in helpers.ZONE_ORDER and the adjacency specified in load_map()["zone_edges"].
A=8.0/B=1.0. Target "tensornet" (tensor-network) backend, and FP32 precision.
Markdown: why tensornet — 36 qubits = 2^36x8 = 550 GB, beyond most single GPU.

Structure:
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
   assert: helpers.is_valid(best) to verify the bitstring is a valid map coloring.
5. Map: counties by zone PALETTE color, zone graph (helpers.group_positions(taiwan,"zone")),
   color legend; per-eval + per-shot seconds; theta*.
```

**Expected outcome:** `02_notebook.ipynb` Successfully completed a 36-qubit simulation and obtained 
valid colorings through sampling, with the map showing all 19 counties colored by zone.

<br>

---

## Step 3 — Reuse and optimize contraction paths

**Goal:** stop paying for planning twice. A tensor-network evaluation has two parts — first
*plan* the contraction path (the order of a chain of matrix multiplications),
then *contract* — and since the QAOA loop never changes the circuit's shape, only the angle
values, replanning on every evaluation is pure waste. So we can optimize:
`CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE` caches the contraction path, and
`CUDAQ_TENSORNET_NUM_HYPER_SAMPLES` lets the planner search for a better path.

```text
Build and execute 03_notebook.ipynb here from 02_notebook.ipynb (read it; reuse
problem/xy_kernel exactly — A=8.0/B=1.0, Nelder-Mead, tensornet fp32).

Structure:
1. FIRST cell, BEFORE `import cudaq`:
   os.environ["CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE"] = "TRUE"
   os.environ["CUDAQ_TENSORNET_NUM_HYPER_SAMPLES"] = "32"
   Rebuild problem.
2. Same xy_kernel and energy(params).
3. Retrieve the elapsed time from 02, or rerun its optimization if no timing result is available,
   SAME size: start [0.1,0.1], max_iterations=5; print best energy, angles,
   elapsed vs 02's total (hardcoded, provenance) — print speedup.
4. SHOTS=2000; sample best angles (post-optimization); summarize_samples; THE assert:
   helpers.is_valid(best), verifying that the sampled bitstring is a valid map coloring.
5. Map (as 02) + table: seconds/evaluation and optimize seconds (02 vs this); speedup+verdict;
   theta* AND exact energy (04 needs them).
```

**Expected outcome:** `03_notebook.ipynb` runs the same optimization with contraction path reuse and 
finishes noticeably faster, and the map is again validly colored.

<br>

---

## Step 4 — A fast approximate solver: Matrix product state

**Goal:** By leveraging matrix product states (MPS) with a lower bond dimension, we can dramatically reduce the memory footprint while retaining exact verification of every sampled result. With a manageable accuracy trade-off, the CUDA-Q's `tensornet-mps` backend can unlock problem sizes that other approaches cannot.

```text
Build and execute 04_notebook.ipynb here from 03_notebook.ipynb (read it; same problem and
xy_kernel, A=8.0/B=1.0, Nelder-Mead).

Structure:
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
   summarize_samples; THE assert: helpers.is_valid(best) to verify validity..
5. Map (counties by zone) + timing table, ONLY time rows: seconds/evaluation (03's vs
   chi=16's), ratio; verdict: sampled quality matches the exact backend at
   at-or-below cost, memory far smaller, readout untrustworthy — answers verified exactly
   after sampling.
```

**Expected outcome:** `04_notebook.ipynb` with the `tensornet-mps` backend still delivers a
validly colored map at a similar speed while using a tiny fraction of the memory.

<br>

---
