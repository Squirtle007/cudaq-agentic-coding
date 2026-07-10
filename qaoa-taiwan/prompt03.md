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
