Build and execute 02_notebook.ipynb here from notebooks 00/01 (read them; keep Nelder-Mead).

Problem: 9 zones (helpers.ZONE_ORDER, load_map()["zone_edges"]), 4 colors -> 36 qubits,
A=8.0/B=1.0. cudaq.set_target("tensornet", option="fp32"), deliberately NO env vars;
cudaq.set_random_seed(1234). Markdown: why tensornet — 36 qubits = 2^36x8 = 550 GB, beyond
most single GPU; evaluations cost tens of seconds.

Structure (<=6 code cells, exactly ONE assert):
1. QUBO via helpers; print term counts.
2. ONE kernel, xy_kernel (NOT plain rx): per-zone W state — x(q[base]), for k in
   0..2: ry.ctrl(W_ANGLES[k], q[base+k], q[base+k+1]) then x.ctrl(q[base+k+1], q[base+k]),
   W_ANGLES = [2.0943951023931953, 1.9106332362490186, 1.5707963267948966] — then 00's
   rz/cx-rz-cx cost layer, then XY ring mixer: ring pairs (0,1),(1,2),(2,3),(3,0), XX as
   h,h/cx/rz(2*beta)/cx/h,h; YY as rx(+pi/2),rx(+pi/2)/cx/rz(2*beta)/cx/rx(-pi/2),rx(-pi/2).
   Only 1-/2-qubit gates.
3. Optimize: Nelder-Mead, max_iterations=5, start [0.1,0.1], timed; print best energy,
   angles, seconds/evaluation (total/5). No energy calls outside these 5.
4. SHOTS=2000; sample best angles ONLY after optimization; helpers.summarize_samples; THE
   assert: helpers.is_valid(best).
5. Map: counties by zone PALETTE color, zone graph (helpers.group_positions(taiwan,"zone")),
   color legend; per-eval + per-shot seconds; theta* (6 decimals).
