# Reproduction Criteria — Notebooks 01–04

Minimal pass/fail bars for an agent reproducing the tutorial notebooks. Each notebook must meet
the **shared floor** plus its own bars. Judge direction and magnitude — timings, sampled
bitstrings, and exact digits vary run to run and must NOT be required to match.

## Shared floor (applies to every notebook)

- [ ] Executes end-to-end (top to bottom) with zero errors.
- [ ] Sampling happens only AFTER the optimization finishes — never during, no post-processing.
- [ ] Exactly one validity assert: the best sampled bitstring passes `helpers.is_valid`.
- [ ] The final result is drawn on the map from the sampled counts.

## 01_notebook — Best optimizer + GPU tuning (statevector)

**Objective:** on the same problem and budget, a better optimizer plus GPU tuning env vars beat
the baseline.

- [ ] Nelder–Mead replaces COBYLA on the same 50-evaluation budget.
- [ ] Gate-fusion env vars are set before `import cudaq`:
      `CUDAQ_FUSION_MAX_QUBITS=3` and `CUDAQ_FUSION_NUM_HOST_THREADS=4`.
- [ ] Per-iteration GPU time is displayed for every evaluation, at millisecond scale.
- [ ] Final energy is at least ~20% lower than the COBYLA baseline.
- [ ] Valid-sample rate improves by at least ~5x over the baseline.

## 02_notebook — Tensor network for scaling

**Objective:** run a problem too big for any state vector, on the `tensornet` backend.

- [ ] The 36-qubit zone problem runs on `tensornet` with default settings (no env vars).
- [ ] The circuit encodes the one-color constraint (W-state start + XY mixer).
- [ ] The notebook completes on one GPU — a size no state vector could hold — with per-evaluation
      cost measured and reported (tens of seconds is expected, not a failure).
- [ ] The 2,000-shot sample contains valid colorings and the best one passes the validity check.

## 03_notebook — Path reuse for cheaper evaluations

**Objective:** same math, cheaper evaluations — only the tensornet env vars change.

- [ ] Problem and circuit are identical to 02_notebook; the only change is the two tensornet
      env vars set before `import cudaq`: `CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=TRUE`
      (path reuse ON) and `CUDAQ_TENSORNET_NUM_HYPER_SAMPLES` raised (e.g. `32`).
- [ ] The same-size optimization (02's start and iteration budget) finishes in clearly less
      wall-clock time than 02_notebook's measured total — taken from 02's output, or re-measured
      by rerunning it if 02 carries no timing — with the speedup reported.

## 04_notebook — MPS: accuracy traded for speed

**Objective:** an approximate backend buys speed; correctness is judged by verified samples, not
the energy readout.

- [ ] Same workload runs on `tensornet-mps` with reduced bond dimension and loosened cutoff via
      env vars set before `import cudaq`: `CUDAQ_MPS_MAX_BOND` (e.g. `16`) and
      `CUDAQ_MPS_ABS_CUTOFF` (e.g. `1e-4`).
- [ ] Per-evaluation and sampling cost come out at-or-below the exact TN's (any measurable
      speedup passes).
- [ ] The energy readout is allowed to disagree with the exact value — do not fail on it.
- [ ] Accuracy is judged by sampling instead: the valid rate stays in the same ballpark as the
      exact backend (not collapsed toward zero), and the best sample passes the exact validity
      check.