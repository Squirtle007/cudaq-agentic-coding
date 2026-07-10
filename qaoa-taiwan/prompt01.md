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
