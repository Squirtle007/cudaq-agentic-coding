"""A simple Quantum Fourier Transform (QFT) implementation in Qiskit.

The QFT converts computational-basis states into Fourier-basis states.

Circuit structure:
  1. Process target qubits from highest to lowest index.
  2. Apply a Hadamard gate to each target qubit.
  3. Apply controlled phase rotations from each lower-index control qubit,
     using the angle pi / 2^(target - control).
  4. Swap the qubits to reverse their order.
"""

from math import pi

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

num_qubits = 3

# Prepare |001>, then apply the QFT.
circuit = QuantumCircuit(num_qubits)
circuit.x(0)

for target in range(num_qubits - 1, -1, -1):
    circuit.h(target)

    for control in range(target - 1, -1, -1):
        angle = pi / 2 ** (target - control)
        circuit.cp(angle, control, target)

# Reverse the qubit order.
for qubit in range(num_qubits // 2):
    circuit.swap(qubit, num_qubits - qubit - 1)

print(circuit.draw())

print("\nQFT of |001>:")
state = Statevector(circuit)

for index in range(2**num_qubits):
    # Round to 3 decimals; the + 0j turns a stray "-0.000" back into "0.000".
    amplitude = np.round(state[index], 3) + 0j

    label = format(index, f"0{num_qubits}b")
    print(f"|{label}>: {amplitude:.3f}")
