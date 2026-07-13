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

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def qft(num_qubits: int) -> QuantumCircuit:
    """Create a Quantum Fourier Transform circuit."""
    circuit = QuantumCircuit(num_qubits, name="QFT")

    # Apply Hadamard gates and controlled phase rotations.
    for target in reversed(range(num_qubits)):
        circuit.h(target)

        for control in reversed(range(target)):
            distance = target - control
            angle = pi / (2**distance)
            circuit.cp(angle, control, target)

    # Reverse the qubit order.
    for qubit in range(num_qubits // 2):
        circuit.swap(qubit, num_qubits - qubit - 1)

    return circuit


def inverse_qft(num_qubits: int) -> QuantumCircuit:
    """Create an inverse Quantum Fourier Transform circuit."""
    return qft(num_qubits).inverse()


# Create and display a three-qubit QFT.
num_qubits = 3
qft_circuit = qft(num_qubits)
print(qft_circuit.draw())

# Prepare |001> and apply the QFT.
circuit = QuantumCircuit(num_qubits)
circuit.x(0)
circuit.compose(qft_circuit, inplace=True)

print("\nQFT of |001>:")
qft_state = Statevector.from_instruction(circuit)

for basis_state, amplitude in qft_state.to_dict().items():
    print(f"|{basis_state}>: {amplitude:.3f}")

# Apply the inverse QFT to recover |001>.
circuit.compose(inverse_qft(num_qubits), inplace=True)

print("\nAfter QFT followed by inverse QFT:")
recovered_state = Statevector.from_instruction(circuit)
print(recovered_state.probabilities_dict())