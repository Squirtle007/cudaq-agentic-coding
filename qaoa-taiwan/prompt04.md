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
