"""A simple Grover search implementation in Qiskit.

Grover's algorithm finds a marked item in an unstructured set.

Circuit structure:
  1. Put every qubit into superposition with a Hadamard gate.
  2. Repeat twice: an oracle that flips the phase of |111>, then a diffuser
     that reflects every amplitude about their average.
  3. Two iterations to amplify the probability of |111>.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

num_qubits = 3
iterations = 2

circuit = QuantumCircuit(num_qubits)
circuit.h(range(num_qubits))

for _ in range(iterations):
    # Oracle: a controlled-controlled-Z flips the phase of |111>.
    circuit.ccz(0, 1, 2)

    # Diffuser: reflect the amplitudes about their average.
    circuit.h(range(num_qubits))
    circuit.x(range(num_qubits))
    circuit.ccz(0, 1, 2)
    circuit.x(range(num_qubits))
    circuit.h(range(num_qubits))

probabilities = Statevector(circuit).probabilities()

# The probability of |111> state.
print(f"|{int(2**num_qubits-1):03b}>: {probabilities[int(2**num_qubits-1)]:.3f}")
