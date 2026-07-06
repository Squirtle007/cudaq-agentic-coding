# CUDA-Q Reference for AI Agents (Python)

> Combined from the NVIDIA CUDA-Q documentation (https://nvidia.github.io/cuda-quantum/latest/): the "CUDA-Q by Example" tutorials, the Dynamics Simulation guide, the GPU simulator backend pages, the Amazon Braket backend page, and the Python API reference. Python only; verified against the latest published docs.

## Contents

- **[Part 1: CUDA-Q by Example](#part-1-cuda-q-by-example)**
  - [1. Introduction](#1-introduction)
  - [2. Building Kernels](#2-building-kernels)
  - [3. Quantum Operations](#3-quantum-operations)
  - [4. Measuring Kernels](#4-measuring-kernels)
  - [5. Visualizing Kernels](#5-visualizing-kernels)
  - [6. Executing Kernels](#6-executing-kernels)
  - [7. Computing Expectation Values](#7-computing-expectation-values)
  - [8. Multi-GPU Workflows](#8-multi-gpu-workflows)
  - [9. Optimizers & Gradients](#9-optimizers-gradients)
  - [10. Noisy Simulations](#10-noisy-simulations)
  - [11. Pre-Trajectory Sampling with Batch Execution (PTSBE)](#11-pre-trajectory-sampling-with-batch-execution-ptsbe)
  - [12. Detector Error Models](#12-detector-error-models)
  - [13. Constructing Operators](#13-constructing-operators)
  - [14. Performance Optimizations](#14-performance-optimizations)
  - [15. Using Quantum Hardware Providers](#15-using-quantum-hardware-providers)
  - [16. When to Use sample vs. run](#16-when-to-use-sample-vs-run)
- **[Part 2: Dynamics Simulation](#part-2-dynamics-simulation)**
  - [Quick Start](#quick-start)
  - [Operator](#operator)
  - [Time-Dependent Dynamics](#time-dependent-dynamics)
  - [Super-operator Representation](#super-operator-representation)
  - [Numerical Integrators](#numerical-integrators)
  - [Batch Simulation](#batch-simulation)
  - [Multi-GPU Multi-Node Execution](#multi-gpu-multi-node-execution)
  - [Examples](#examples)
- **[Part 3: GPU Simulator Backends and Performance Options](#part-3-gpu-simulator-backends-and-performance-options)**
  - [State Vector Simulators](#state-vector-simulators)
  - [Tensor Network Simulators](#tensor-network-simulators)
- **[Part 4: Amazon Braket Backend](#part-4-amazon-braket-backend)**
  - [Setting Credentials](#setting-credentials)
  - [Submitting](#submitting)
- **[Part 5: Python API Reference](#part-5-python-api-reference)**
  - [Program Construction](#program-construction)
  - [Kernel Execution](#kernel-execution)
  - [Quantum Error Correction](#quantum-error-correction)
  - [Backend Configuration](#backend-configuration)
  - [Dynamics](#dynamics)
  - [Operators](#operators)
  - [Data Types](#data-types)
  - [MPI Submodule](#mpi-submodule)
  - [PTSBE Submodule](#ptsbe-submodule)

---

## Part 1: CUDA-Q by Example

> Compiled from the [CUDA-Q by Example](https://nvidia.github.io/cuda-quantum/latest/using/examples/examples.html) section of the NVIDIA CUDA-Q documentation. Python only; code blocks are taken verbatim from the official example sources in the [CUDA-Q repository](https://github.com/NVIDIA/cuda-quantum).

### 1. Introduction

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/introduction.html

Welcome to CUDA-Q! On this page we will illustrate CUDA-Q with several examples.

Quantum programs are constructed through CUDA-Q's `Kernel` API; invoking a kernel's methods builds a program that is then executed by calling, for example, `cudaq.sample`:

```python
import cudaq

# We begin by defining the `Kernel` that we will construct our
# program with.
@cudaq.kernel
def kernel():
    '''
    This is our first CUDA-Q kernel.
    '''
    # Next, we can allocate a single qubit to the kernel via `qubit()`.
    qubit = cudaq.qubit()

    # Now we can begin adding instructions to apply to this qubit!
    # Here we'll just add every non-parameterized
    # single qubit gate that is supported by CUDA-Q.
    h(qubit)
    x(qubit)
    y(qubit)
    z(qubit)
    t(qubit)
    s(qubit)

    # Next, we add a measurement to the kernel so that we can sample
    # the measurement results on our simulator!
    mz(qubit)

# Finally, we can execute this kernel on the state vector simulator
# by calling `cudaq.sample`. This will execute the provided kernel
# `shots_count` number of times and return the sampled distribution
# as a `cudaq.SampleResult` dictionary.
result = cudaq.sample(kernel)

# Now let's take a look at the `SampleResult` we've gotten back!
print(result)
```

### 2. Building Kernels

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html

This section will cover the most basic CUDA-Q construct, a quantum kernel.
Topics include, building kernels, initializing states, and applying gate operations.

#### Defining Kernels

Kernels are the building blocks of quantum algorithms in CUDA-Q. A kernel is specified by using the following syntax. `cudaq.qubit` builds a register consisting of a single qubit, while `cudaq.qvector` builds a register of $N$ qubits.

```python
import cudaq

@cudaq.kernel
def kernel():
    A = cudaq.qubit()
    B = cudaq.qvector(3)
    C = cudaq.qvector(5)
```

Inputs to kernels are defined by specifying a parameter in the kernel definition along with the appropriate type. The kernel below takes an integer to define a register of N qubits.

```python
N = 2

@cudaq.kernel
def kernel(N: int):
    register = cudaq.qvector(N)
```

#### Initializing states

It is often helpful to define an initial state for a kernel. There are a few ways to do this in CUDA-Q. Note, method 5 is particularly useful for cases where the state of one kernel is passed into a second kernel to prepare its initial state.

1. Passing complex vectors as parameters

```python
# Passing complex vectors as parameters
c = [.707 + 0j, 0 - .707j]

@cudaq.kernel
def kernel(vec: list[complex]):
    q = cudaq.qvector(vec)
```

2. Capturing complex vectors

```python
# Capturing complex vectors
c = [0.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)
```

3. Precision-agnostic API

```python
# Precision-Agnostic API
import numpy as np

c = np.array([0.70710678 + 0j, 0., 0., 0.70710678], dtype=cudaq.complex())

@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)
```

4. Define as CUDA-Q amplitudes

```python
# Define as CUDA-Q amplitudes
c = cudaq.amplitudes([0.70710678 + 0j, 0., 0., 0.70710678])

@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)
```

5. Pass in a state from another kernel

```python
# Pass in a state from another kernel
c = [0.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel_initial():
    q = cudaq.qvector(c)

state_to_pass = cudaq.get_state(kernel_initial)

@cudaq.kernel
def kernel(state: cudaq.State):
    q = cudaq.qvector(state)

kernel(state_to_pass)
```

#### Applying Gates

After a kernel is constructed, gates can be applied to start building out a quantum circuit.
All the predefined gates in CUDA-Q can be found [here](https://nvidia.github.io/cuda-quantum/api/default_ops).

Gates can be applied to all qubits in a register.

```python
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    h(register)
```

Or, to individual qubits in a register.

```python
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    h(register[0])  # first qubit
    h(register[-1])  # last qubit
```

#### Controlled Operations

Controlled operations are available for any gate and can be used by adding `.ctrl` to the end of any gate, followed by specification of the control qubit and the target qubit.

```python
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    x.ctrl(register[0],
           register[1])  # CNOT gate applied with qubit 0 as control
```

#### Multi-Controlled Operations

It is valid for more than one qubit to be used for multi-controlled gates. The control qubits are specified as a list.

```python
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    x.ctrl([register[0], register[1]],
           register[2])  # X applied to qubit two controlled by qubit 0 and 1
```

You can also call a controlled kernel within a kernel.

```python
@cudaq.kernel
def x_kernel(qubit: cudaq.qubit):
    x(qubit)

# A kernel that will call `x_kernel` as a controlled operation.
@cudaq.kernel
def kernel():

    control_vector = cudaq.qvector(2)
    target = cudaq.qubit()

    x(control_vector)
    x(target)
    x(control_vector[1])
    cudaq.control(x_kernel, control_vector, target)

# The above is equivalent to:
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(3)
    x(qvector)
    x(qvector[1])
    x.ctrl([qvector[0], qvector[1]], qvector[2])
    mz(qvector)

results = cudaq.sample(kernel)
print(results)
```

#### Adjoint Operations

The adjoint of a gate can be applied by appending the gate with the `adj` designation.

```python
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    t.adj(register[0])
```

#### Custom Operations

Custom gate operations can be specified using `cudaq.register_operation`. A one-dimensional `Numpy` array specifies the unitary matrix to be applied. The entries of the array read from top to bottom through the rows.

```python
import numpy as np

cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

@cudaq.kernel
def kernel():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    custom_x(qubits[0])
    custom_x.ctrl(qubits[0], qubits[1])
```

#### Building Kernels with Kernels

A kernel can call another kernel as a subroutine; below, `kernel_A` is called within `kernel_B` to perform CNOT operations.

```python
@cudaq.kernel
def kernel_A(qubit_0: cudaq.qubit, qubit_1: cudaq.qubit):
    x.ctrl(qubit_0, qubit_1)

@cudaq.kernel
def kernel_B():
    reg = cudaq.qvector(10)
    for i in range(5):
        kernel_A(reg[i], reg[i + 1])
```

#### Parameterized Kernels

It is often useful to define parameterized circuit kernels which can be used for applications like VQE.

```python
@cudaq.kernel
def kernel(thetas: list[float]):
    qubits = cudaq.qvector(2)
    rx(thetas[0], qubits[0])
    ry(thetas[1], qubits[1])

thetas = [.024, .543]
kernel(thetas)
```

### 3. Quantum Operations

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/quantum_operations.html

The fundamental unit of classical information storage, processing and
transmission is the bit. Analogously, we define its quantum counterpart,
a quantum bit or simply the qubit.

Classical bits are transistor elements whose states can be altered to
perform computations. Similarly qubits too have physical realizations
within superconducting materials, ion-traps and photonic systems. We
shall not concern ourselves with specific qubit architectures but rather
think of them as systems which obey the laws of quantum mechanics and
the mathematical language physicists have developed to describe the
theory: linear algebra.

#### Quantum States

Information storage scales linearly if bits have a single state. Access
to multiple states, namely a 0 and a 1 allows for information encoding
to scale logarithmically. Similarly we define a qubit to have the states
$\ket{0}$ and $\ket{1}$ in Dirac notation where:

$$\ket{0} = \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix}$$

$$\ket{1} = \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix}$$

Rather than just the two states each classical bit can be in,
quantum theory allows one to explore linear combinations of states,
also called superpositions:

$$\ket{\psi} = \alpha\ket{0} + \beta\ket{1}$$

where $\alpha$ and $\beta$ $\in \mathbb{C}$. It is
important to note that this is still the state of one qubit;
although we have two kets, they both represent a
superposition state of one qubit.

If we have two classical bits, the possible states we could encode
information in would be 00, 01, 10 and 11. Correspondingly, multiple
qubits can be combined and the possible combinations of their states
used to process information.

A two qubit system has four computational basis states:
$\ket{00}, \ket{01}, \ket{10}, \ket{11}$.

More generally, the quantum state of a $n$ qubit system is written
as a sum of $2^n$ possible basis states where the coefficients
track the probability of the system collapsing into that state if a
measurement is applied.

For $n = 500$, storing the $2^{500} \approx 10^{150}$ amplitudes is classically infeasible, yet nature requires only 500 qubits — hence the value of offloading large statevector evolution onto a quantum computer.

#### Quantum Gates

We can manipulate the state of a qubit via quantum gates.
For example, the Pauli X gate allows us to flip the state of the qubit:

$$X \ket{0} = \ket{1}$$

$$\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix}$$

```python
import cudaq

@cudaq.kernel
def kernel():
    # A single qubit initialized to the ground / zero state.
    qubit = cudaq.qubit()

    # Apply the Pauli x gate to the qubit.
    x(qubit)

    # Measurement operator.
    mz(qubit)

# Sample the qubit for 1000 shots to gather statistics.
result = cudaq.sample(kernel)
print(result.most_probable())
```

```text
{ 1:1000 }
```

The Hadamard gate allows us to put the qubit in an equal superposition
state:

$$H \ket{0} =  \tfrac{1}{\sqrt{2}} \ket{0} + \tfrac{1}{\sqrt{2}} \ket{1}  \equiv \ket{+}$$

$$\tfrac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} = \tfrac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} + \tfrac{1}{\sqrt{2}} \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix}.$$

The probability of finding the qubit in the $\ket{0}$ or $\ket{1}$ state is hence
$\lvert \tfrac{1}{\sqrt{2}} \rvert ^2 = \tfrac{1}{2}$. Lets verify
this with some code:

```python
import cudaq

@cudaq.kernel
def kernel():
    # A single qubit initialized to the ground/ zero state.
    qubit = cudaq.qubit()

    # Apply Hadamard gate to single qubit to put it in equal superposition.
    h(qubit)

    # Measurement operator.
    mz(qubit)

result = cudaq.sample(kernel)
print("Measured |0> with probability " +
      str(result["0"] / sum(result.values())))
print("Measured |1> with probability " +
      str(result["1"] / sum(result.values())))
```

```text
{ 0:502 1:498 }
```

For a qubit in a superposition state, quantum gates
act linearly:

$$X (\alpha\ket{0} + \beta\ket{1}) = \alpha\ket{1} + \beta\ket{0}$$

As we evolve quantum states via quantum gates, the normalization
condition requires that the sum of modulus squared of amplitudes must
equal 1 at all times:

$$\ket{\psi} = \alpha\ket{0} + \beta\ket{1},          |\alpha|^2 + |\beta|^2 = 1.$$

This is to adhere to the conservation of probabilities which translates
to a constraint on types of quantum gates we can define.
For a general quantum state $\ket{\psi}$, upholding the
normalization condition requires quantum gates to be unitary, that is
$U^{\dagger}U = U^{*^{T}}U = \mathbb{I}$.

Just like the single-qubit gates above, we can define
multi-qubit gates to act on multiple-qubits.
The controlled-NOT or CNOT gate, for example, acts on 2 qubits: the control qubit and
the target qubit. Its effect is to flip the target if the control is in
the excited $\ket{1}$ state.

```python
import cudaq

@cudaq.kernel
def kernel():
    # 2 qubits both initialized to the ground/ zero state.
    qvector = cudaq.qvector(2)

    x(qvector[0])

    # Controlled-not gate operation.
    x.ctrl(qvector[0], qvector[1])

    mz(qvector[0])
    mz(qvector[1])

result = cudaq.sample(kernel)
print(result)
```

```text
{ 11:1000 }
```

The CNOT gate in matrix notation is represented as

$$CNOT \equiv \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

and one can check that $CNOT^\dagger CNOT = \mathbb{I}$.
Its effect on the computational basis states is:

$$CNOT\ket{00} = \ket{00}$$

$$CNOT\ket{01} = \ket{01}$$

$$CNOT\ket{10} = \ket{11}$$

$$CNOT\ket{11} = \ket{10}$$

For a full list of gates supported in CUDA-Q see [default ops](https://nvidia.github.io/cuda-quantum/latest/api/default_ops.html).

#### Measurements

Quantum theory is probabilistic and hence requires statistical inference
to derive observations. Prior to measurement, the state of a qubit is
all possible combinations of $\alpha$ and $\beta$ and upon
measurement, wavefunction collapse yields either a classical 0 or 1.

The mathematical theory devised to explain quantum phenomena tells us
that the probability of observing the qubit in the state
$\ket{0}$ or $\ket{1}$, yielding a classical 0 or 1, is
$\lvert \alpha \rvert ^2$ or $\lvert \beta \rvert ^2$, respectively.

As we see in the example of the Hadamard gate above,
the result 0 or 1 each is yielded roughly 50% of the times as predicted
by the postulate stated above thus proving the theory.

Classically, we cannot encode information within states such as 00 + 11
but quantum mechanics allows us to write linear superpositions

$$\ket{\psi} = \alpha_{00}\ket{00} + \alpha_{01}\ket{01} + \alpha_{10}\ket{10} + \alpha_{11}\ket{11}$$

where the probability of measuring $x = 00, 01, 10, 11$ occurs
with probability $\lvert \alpha_{x} \rvert ^2$ with the
normalization condition that
$\sum_{x \in \{ 0,1 \}^2} \lvert \alpha_{x} \rvert ^2 = 1$.

### 4. Measuring Kernels

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/measuring_kernels.html

```python
import cudaq
```

Kernel measurement can be specified in the Z, X, or Y basis using `mz`, `mx`, and `my`. Measurement occurs in the Z basis by default.

```python
@cudaq.kernel
def kernel():
    qubits = cudaq.qvector(2)
    mz(qubits)
```

Specific qubits or registers can be measured rather than the entire kernel.

```python
@cudaq.kernel
def kernel():
    qubits_a = cudaq.qvector(2)
    qubit_b = cudaq.qubit()
    mz(qubits_a)
    mx(qubit_b)
```

#### Measurement Handles

In CUDA-Q, `mz`, `mx`, and `my` return a *measurement
handle* — `cudaq::measure_handle` in C++ (with the alias
`cudaq::measure_result`), and the `measure_handle` type in
Python — rather than a classical value. Measuring a single qubit returns one
handle; measuring a `qvector` returns a vector of handles. A handle
records a measurement event and defers reading its classical value, so the
same measurement can drive mid-circuit conditional logic and
quantum-error-correction declarations (see [dem from kernel](https://nvidia.github.io/cuda-quantum/latest/using/examples/dem_from_kernel.html)).

A handle is *discriminated* into its classical bit by using it in a boolean
context inside the kernel — for example the `if (b0)` test in the
mid-circuit example below. To read a whole vector of handles at once,
discriminate it in bulk with `to_bools` (yielding a
`list[bool]` / `std::vector<bool>`) or `to_integer`
(packing the bits little-endian into an integer). The C++ mid-circuit example
below returns `to_bools(mz(q))`; a Python kernel typed to return
`list[bool]` discriminates a returned handle vector automatically.

A handle cannot cross the host-device boundary without being discriminated:
convert it to a boolean, `list[bool]` / `std::vector<bool>`, or
integer inside the kernel before returning it.

#### Mid-circuit Measurement and Conditional Logic

Operations in a kernel can depend on earlier measurement results. The example below applies a Hadamard on qubit 0, measures it into `b0`, then resets qubit 0 (flipping it to 1 for later use), and finally applies a Hadamard on qubit 1 if `b0` is 1.

The results show qubit 0 is one, indicating the reset worked, and qubit 1 has a 75/25 distribution, demonstrating the mid-circuit measurement worked as expected.

```python
@cudaq.kernel
def kernel() -> list[bool]:
    q = cudaq.qvector(2)

    h(q[0])
    b0 = mz(q[0])
    reset(q[0])
    x(q[0])

    if b0:
        h(q[1])

    return mz(q)

from collections import Counter

results = cudaq.run(kernel, shots_count=1000)
# Convert results to bitstrings and count
bitstring_counts = Counter(
    ''.join('1' if bit else '0' for bit in result) for result in results)

print(f"Bitstring counts: {dict(bitstring_counts)}")
```

Output

```python
Bitstring counts: {'11': 247, '10': 753}
```

### 5. Visualizing Kernels

> Source: https://nvidia.github.io/cuda-quantum/latest/examples/python/visualization.html

#### Qubit Visualization

What are the possible states a qubit can be in and how can we build up a visual cue to help us make sense of quantum states and their evolution?

The states $\ket{0}$ and $\ket{1}$ can be placed on a one-dimensional line (the z-axis below); equal superpositions such as $\ket{+}$ and $\ket{-}$ require the 2D $xy$-plane; capturing all remaining states calls for a 3D extension. 

In general, a quantum state can be written in the form $\ket{\psi} = \cos(\frac{\theta}{2})\ket{0}+e^{i\varphi}\sin(\frac{\theta}{2})\ket{1}$ where $\theta$ is a real number between $0$ and $\pi$ and $\varphi$ is a real value between $0$ and $2\pi$.  For example, the minus state, $\ket{-} = \frac{1}{\sqrt{2}}\ket{0}- \frac{1}{\sqrt{2}}\ket{1}$, can be rewritten as
$$\ket{-}  = \cos(\frac{\theta}{2})\ket{0}+e^{i\varphi}\sin(\frac{\theta}{2})\ket{1}\text{ with }\theta = \frac{\pi}{2}\text{ and }\varphi = \pi.$$ 
This can be visualized in the image below as a unit vector pointing in the direction of the negative $x$-axis.

Using spherical coordinates, it is possible to depict all the possible states of a single qubit on a sphere. This is called a Bloch sphere. 

![figure](https://nvidia.github.io/cuda-quantum/latest/_images/Bloch_sphere.png)

Let us try to showcase the functionality to render such a 3D representation with CUDA-Q. 
First, let us define a single-qubit kernel that returns a different state each time. This kernel uses random rotations.

Note: CUDA-Q uses the [QuTiP](https://qutip.org) library to render Bloch spheres. The following code will throw an error if QuTiP is not installed.

```python
# Install `qutip` and `matplotlib` in the current Python kernel if missing
# (restart the kernel after installing).

import sys

try:
    import matplotlib.pyplot as plt
    import qutip

except ImportError:
    print("Tools not found, please install and restart your kernel after this is done.")
    #!{sys.executable} -m pip install qutip\>5 matplotlib\>=3.5
```

```python
import cudaq
import numpy as np

## Retry the subsequent cells by setting the target to density matrix simulator.
# cudaq.set_target("density-matrix-cpu")

@cudaq.kernel
def kernel(angles: np.ndarray):
    qubit = cudaq.qubit()
    rz(angles[0], qubit)
    rx(angles[1], qubit)
    rz(angles[2], qubit)
```

Next, we instantiate a random number generator, so we can get random outputs. We then create 4 random single-qubit states by using `cudaq.add_to_bloch_sphere()` on the output state obtained from the random kernel.

```python
rng = np.random.default_rng(seed=11)
blochSphereList = []
for _ in range(4):
    angleList = rng.random(3) * 2 * np.pi
    sph = cudaq.add_to_bloch_sphere(cudaq.get_state(kernel, angleList))
    blochSphereList.append(sph)
```

We can display the spheres with `cudaq.show()`; set `nrows`/`ncols` to arrange multiple spheres in a grid (the grid must have at least as many slots as spheres, else it throws an error):

```python
cudaq.show(blochSphereList[0])                      # a single sphere
cudaq.show(blochSphereList[:2], nrows=1, ncols=2)   # two in a row
cudaq.show(blochSphereList[:], nrows=2, ncols=2)    # all four in a 2x2 grid
```

*(figure outputs omitted)*

Multiple states can be added to one Bloch sphere by passing a `qutip.Bloch` object to `cudaq.add_to_bloch_sphere()`:

```python
import qutip

rng = np.random.default_rng(seed=47)
blochSphere = qutip.Bloch()
for _ in range(10):
    angleList = rng.random(3) * 2 * np.pi
    sph = cudaq.add_to_bloch_sphere(cudaq.get_state(kernel, angleList), blochSphere)
```

Display the resulting sphere (10 random vectors) with `blochSphere.show()`. *(figure output omitted)*

Unfortunately, there is no such handy visualization for multi-qubit states. In particular, a multi-qubit state cannot be visualized as multiple Bloch spheres due to the nature of entanglement that makes quantum computing so powerful.

#### Kernel Visualization

A CUDA-Q kernel can be visualized using the `cudaq.draw` API which returns a string representing the drawing of the execution path, in the specified format. ASCII (default) and LaTeX formats are supported.

```python
@cudaq.kernel
def kernel_to_draw():
    q = cudaq.qvector(4)
    h(q)
    x.ctrl(q[0], q[1])
    y.ctrl([q[0], q[1]], q[2])
    z(q[2])
    
    swap(q[0], q[1])
    swap(q[0], q[3])
    swap(q[1], q[2])

    r1(3.14159, q[0])
    tdg(q[1])
    s(q[2])
```

```python
print(cudaq.draw(kernel_to_draw))
```

Output:

```text
     ╭───╮                  ╭───────────╮       
q0 : ┤ h ├──●────●────╳───╳─┤ r1(3.142) ├───────
     ├───┤╭─┴─╮  │    │   │ ╰───────────╯╭─────╮
q1 : ┤ h ├┤ x ├──●────╳───┼───────╳──────┤ tdg ├
     ├───┤╰───╯╭─┴─╮╭───╮ │       │      ╰┬───┬╯
q2 : ┤ h ├─────┤ y ├┤ z ├─┼───────╳───────┤ s ├─
     ├───┤     ╰───╯╰───╯ │               ╰───╯ 
q3 : ┤ h ├────────────────╳─────────────────────
     ╰───╯
```

```python
print(cudaq.draw('latex', kernel_to_draw))
```

Output:

```text
\documentclass{minimal}
\usepackage{quantikz}
\begin{document}
\begin{quantikz}
... (LaTeX/quantikz source continues)
```

Copy this output string into any LaTeX editor and export it to PDF.

![figure](https://nvidia.github.io/cuda-quantum/latest/_images/circuit_pdf.png)

### 6. Executing Kernels

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html

In CUDA-Q, there are 4 ways in which one can execute quantum kernels:

1. `sample`: yields measurement counts
2. `run`: yields individual return values from multiple executions
3. `observe`: yields expectation values
4. `get_state`: yields the quantum statevector of the computation

Asynchronous execution lets the program continue while a long-running task completes in the background. Since kernel execution is the most intensive task, each execution function can be parallelized given access to multiple quantum processing units (multi-QPU) using: `sample_async`, `run_async`, `observe_async` and `get_state_async`.

Since multi-QPU platforms are not yet feasible, we emulate each QPU with a GPU.

#### Sample

Quantum states collapse upon measurement and hence need to be sampled many times to gather statistics. The CUDA-Q `sample` call enables this:

```python
import cudaq
import numpy as np

qubit_count = 2

# Define the simulation target.
cudaq.set_target("qpp-cpu")

# Define a quantum kernel function.
@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)

    # 2-qubit GHZ state.
    h(qvector[0])
    for i in range(1, qubit_count):
        x.ctrl(qvector[0], qvector[i])

    # If we do not specify measurements, all qubits are measured in
    # the Z-basis by default or we can manually specify it also
    mz(qvector)

print(cudaq.draw(kernel, qubit_count))

result = cudaq.sample(kernel, qubit_count, shots_count=1000)

print(result)
```

```python
     ╭───╮     
q0 : ┤ h ├──●──
     ╰───╯╭─┴─╮
q1 : ─────┤ x ├
          ╰───╯

{ 11:506 00:494 }
```

Note that there is a subtle difference between how `sample` is executed with the target device set to a simulator or with the target device set to a QPU. In simulation mode, the quantum state is built once and then sampled $s$ times where $s$ equals the `shots_count`. In hardware execution mode, the quantum state collapses upon measurement and hence needs to be rebuilt over and over again.

There are a number of helpful tools that can be found in the [API docs](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api) to process the `cudaq.SampleResult` object produced by `sample`.

##### Sample Asynchronous

`sample` also supports asynchronous execution for the [sample_async arguments it accepts](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.sample_async). One can parallelize over various kernels, variational parameters or even distribute shots counts over multiple QPUs.

```python
result_async = cudaq.sample_async(kernel, qubit_count, shots_count=1000)

print(result_async.get())
```

```python
{ 00:498 11:502 }
```

#### Run

The `run` API executes a quantum kernel multiple times and returns each individual result. Unlike `sample`, which collects measurement statistics as counts, `run` preserves each individual return value from every execution. This is useful when you need to analyze the distribution of returned values rather than just aggregated measurement counts.

Key points about `run`:

- Requires a kernel that returns a non-void value
- Returns a list containing all individual execution results
- Supports scalar types (bool, int, float) and custom data classes as return types

```python
# Define a quantum kernel that returns an integer
@cudaq.kernel
def simple_ghz(num_qubits: int) -> int:
    # Allocate qubits
    qubits = cudaq.qvector(num_qubits)

    # Create GHZ state
    h(qubits[0])
    for i in range(1, num_qubits):
        x.ctrl(qubits[0], qubits[i])

    # Measure and return total number of qubits in state |1⟩
    res = 0
    for i in range(num_qubits):
        if mz(qubits[i]):
            res += 1

    return res

# Execute the kernel 20 times
num_qubits = 3
results = cudaq.run(simple_ghz, num_qubits, shots_count=20)

print(f"Executed {len(results)} shots")
print(f"Results: {results}")
print(f"Possible values: Either 0 or {num_qubits} due to GHZ state properties")

# Count occurrences of each result
value_counts = {}
for value in results:
    value_counts[value] = value_counts.get(value, 0) + 1

print("\nCounts of each result:")
for value, count in value_counts.items():
    print(f"{value}: {count} times")
```

```python
Executed 20 shots
Results: [0, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 0, 3, 0, 3, 3, 0, 3, 3, 3]
Possible values: Either 0 or 3 due to GHZ state properties

Counts of each result:
0: 8 times
3: 12 times
```

##### Return Custom Data Types

The `run` API also supports returning custom data types using Python's data classes. This allows returning multiple values from your quantum computation in a structured way.

```python
from dataclasses import dataclass

# Define a custom `dataclass` to return from our quantum kernel
@dataclass(slots=True)
class MeasurementResult:
    first_qubit: bool
    last_qubit: bool
    total_ones: int

@cudaq.kernel
def bell_pair_with_data() -> MeasurementResult:
    # Create a bell pair
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])

    # Measure both qubits
    first_result = mz(qubits[0])
    last_result = mz(qubits[1])

    # Return custom data structure with results
    total = 0
    if first_result:
        total = 1
    if last_result:
        total = total + 1

    return MeasurementResult(first_result, last_result, total)

# Run the kernel 10 times and get all results
results = cudaq.run(bell_pair_with_data, shots_count=10)

# Analyze the results
print("Individual measurement results:")
for i, res in enumerate(results):
    print(
        f"Shot {i}: {{{res.first_qubit}, {res.last_qubit}}}\ttotal ones={res.total_ones}"
    )

# Verify the Bell state correlations
correlated_count = sum(
    1 for res in results if res.first_qubit == res.last_qubit)
print(
    f"\nCorrelated measurements: {correlated_count}/{len(results)} ({correlated_count/len(results)*100:.1f}%)"
)
```

```python
Individual measurement results:
Shot 0: {True, True}	total ones=2
Shot 1: {False, False}	total ones=0
Shot 2: {False, False}	total ones=0
Shot 3: {False, False}	total ones=0
Shot 4: {True, True}	total ones=2
Shot 5: {True, True}	total ones=2
Shot 6: {True, True}	total ones=2
Shot 7: {False, False}	total ones=0
Shot 8: {False, False}	total ones=0
Shot 9: {True, True}	total ones=2

Correlated measurements: 10/10 (100.0%)
```

##### Run Asynchronous

Similar to `sample_async`, `run` also supports asynchronous execution for the [run_async arguments it accepts](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.run_async).

```python
# Example of `run_async` with a simple integer return type
# Define a quantum kernel that returns an integer
@cudaq.kernel
def simple_count(angle: float) -> int:
    q = cudaq.qubit()
    rx(angle, q)
    return int(mz(q))

# Execute asynchronously with different parameters
futures = []
angles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

for i, angle in enumerate(angles):
    futures.append(cudaq.run_async(simple_count, angle, shots_count=10))

# Process results as they complete
for i, future in enumerate(futures):
    results = future.get()
    ones_count = sum(results)
    print(f"Angle {angles[i]:.1f}: {ones_count}/10 ones measured")
```

```python
Angle 0.0: 0/10 ones measured
Angle 0.2: 0/10 ones measured
Angle 0.4: 0/10 ones measured
Angle 0.6: 0/10 ones measured
Angle 0.8: 1/10 ones measured
Angle 1.0: 2/10 ones measured
Angle 1.2: 3/10 ones measured
Angle 1.4: 5/10 ones measured
```

> **Note:** Currently, `run` and `run_async` are supported on simulator targets and select hardware platforms.

#### Observe

The `observe` function allows us to calculate expectation values. We must supply a spin operator in the form of a Hamiltonian, $H$, from which we would like to calculate $\langle\psi|H|\psi\rangle$.

```python
from cudaq import spin

# Define a Hamiltonian in terms of Pauli Spin operators.
hamiltonian = spin.z(0) + spin.y(1) + spin.x(0) * spin.z(0)

@cudaq.kernel
def kernel1(n_qubits: int):
    qubits = cudaq.qvector(n_qubits)
    h(qubits[0])
    for i in range(1, n_qubits):
        x.ctrl(qubits[0], qubits[i])

# Compute the expectation value given the state prepared by the kernel.
result = cudaq.observe(kernel1, hamiltonian, qubit_count).expectation()

print('<H> =', result)
```

```python
<H> = 0.0
```

##### Observe Asynchronous

`observe` can be a time intensive task. We can parallelize the execution of `observe` via the arguments it accepts.

```python
# Define a quantum kernel function.
@cudaq.kernel
def kernel1(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)

    # 2-qubit GHZ state.
    h(qvector[0])
    for i in range(1, qubit_count):
        x.ctrl(qvector[0], qvector[i])

# Measuring the expectation value of 2 different Hamiltonians in parallel
hamiltonian_1 = spin.x(0) + spin.y(1) + spin.z(0) * spin.y(1)

# Asynchronous execution on multiple `qpus` via `nvidia` `gpus`.
result_1 = cudaq.observe_async(kernel1, hamiltonian_1, qubit_count, qpu_id=0)

# Retrieve results
print(result_1.get().expectation())
```

```python
1.1102230246251565e-16
```

Above we parallelized the `observe` call over the `hamiltonian` parameter; however, we can parallelize over any of the arguments it accepts by just iterating over the `qpu_id`.

#### Get State

The `get_state` function gives us access to the quantum statevector of the computation. Remember, that this is only feasible in simulation mode.

```python
# Compute the statevector of the kernel
result = cudaq.get_state(kernel, qubit_count)

print(np.array(result.dump()))
```

```python
[0.+0.j 0.+0.j 0.+0.j 1.+0.j]
```

The statevector generated by the `get_state` command follows Big-endian convention for associating numbers with their binary representations, which places the least significant bit on the left. That is, for the example of a 2-bit system, we have the following translation between integers and bits:

$$
\begin{matrix} 
\text{Integer} & \text{Binary representation}\\
& \text{least signinificant bit on left}\\
0 = \textcolor{red}{0}*2^0 + \textcolor{blue}{0} *2^1 & \textcolor{red}{0}\textcolor{blue}{0} \\
1 = \textcolor{red}{1}*2^0 + \textcolor{blue}{0} *2^1 & \textcolor{red}{1}\textcolor{blue}{0} \\
2 = \textcolor{red}{0}*2^0 + \textcolor{blue}{1} *2^1 & \textcolor{red}{0}\textcolor{blue}{1} \\
3 = \textcolor{red}{1}*2^0 + \textcolor{blue}{1} *2^1 & \textcolor{red}{1}\textcolor{blue}{1}
\end{matrix}
$$

##### Get State Asynchronous

Similar to `observe_async` above, `get_state` also supports asynchronous execution for the [arguments it accepts](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.get_state_async).

```python
import numpy as np

@cudaq.kernel
def bell_state():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])

# Get state asynchronously
state_future = cudaq.get_state_async(bell_state)

# Do other work while waiting for state computation...
print("Computing state asynchronously...")

# Get the state when ready
state = state_future.get()
print("Bell state vector:")
print(np.array(state.dump()))
```

```python
Computing state asynchronously...
Bell state vector:
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```

### 7. Computing Expectation Values

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/expectation_values.html

CUDA-Q provides generic library functions enabling one to compute expectation values
of quantum spin operators with respect to a parameterized CUDA-Q kernel.

#### Parallelizing across Multiple Processors

One typical use case of multi-processor platforms is to distribute the
expectation value computations of a multi-term Hamiltonian across multiple virtual QPUs.

The following shows an example using the `nvidia-mqpu` platform:

```python
import cudaq
from cudaq import spin

cudaq.set_target("nvidia", option="mqpu")
target = cudaq.get_target()
num_qpus = target.num_qpus()
print("Number of QPUs:", num_qpus)

# Define spin ansatz.
@cudaq.kernel
def kernel(angle: float):
    qvector = cudaq.qvector(2)
    x(qvector[0])
    ry(angle, qvector[1])
    x.ctrl(qvector[1], qvector[0])

# Define spin Hamiltonian.
hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

exp_val = cudaq.observe(kernel,
                        hamiltonian,
                        0.59,
                        execution=cudaq.parallel.thread).expectation()
print("Expectation value: ", exp_val)
```

The Hamiltonian above contains four non-identity terms, so four circuits must be executed to compute its expectation value for the state prepared by the ansatz. On the `nvidia-mqpu` platform these circuits are distributed across all available QPUs and the final expectation value is assembled from all QPU results.

### 8. Multi-GPU Workflows

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/multi_gpu_workflows.html

CUDA-Q backends enable seamless switching between CPUs, GPUs and QPUs, including workflows that mix architectures. This section covers accelerating any circuit simulation with a GPU and scaling large simulations with multi-GPU multi-node capabilities.

#### From CPU to GPU

The code below defines a kernel that creates a GHZ state using $N$ qubits.

```python
import cudaq

@cudaq.kernel
def ghz_state(qubit_count: int):
    qubits = cudaq.qvector(qubit_count)
    h(qubits[0])
    for i in range(1, qubit_count):
        cx(qubits[0], qubits[i])
    mz(qubits)

def sample_ghz_state(qubit_count, target):
    """A function that will sample a variable sized GHZ state."""
    cudaq.set_target(target)
    result = cudaq.sample(ghz_state, qubit_count, shots_count=1000)
    return result
```

You can run a state vector simulation using your CPU with the `qpp-cpu` backend. This is helpful for debugging code and testing small circuits.

```python
cpu_result = sample_ghz_state(qubit_count=2, target="qpp-cpu")
cpu_result.dump()
```

```text
{ 00:475 11:525 }
```

As the number of qubits increases to even modest size, the CPU simulation will become impractically slow.  By switching to the `nvidia` backend, you can accelerate the same code on a single GPU and achieve a speedup of up to **425x**.  If you have a GPU available, this the default backend to ensure maximum productivity.

```python
if cudaq.num_available_gpus() > 0:
    gpu_result = sample_ghz_state(qubit_count=25, target="nvidia")
    gpu_result.dump()
```

```text
{ 0000000000000000000000000:510 1111111111111111111111111:490 }
```

#### Pooling the memory of multiple GPUs (`mgpu`)

The state vector has $2^N$ complex elements (8 bytes each) — roughly 8 GB at 30 qubits — so a few more qubits quickly exceed a single GPU's memory. The `mgpu` option solves this by pooling the memory of multiple GPUs across multiple nodes to perform a single state vector simulation.

![figure](https://nvidia.github.io/cuda-quantum/latest/_images/mgpu.png)

If you have multiple GPUs, you can use the following command to run the simulation across $n$ GPUs.

`mpiexec -np n python3 program.py --target nvidia --target-option mgpu`

This executes in an MPI context, providing the memory to simulate much larger state vectors.

You can also set `cudaq.set_target('nvidia', option='mgpu')` within the file to select the target.

#### Parallel execution over multiple QPUs (`mqpu`)

##### Batching Hamiltonian Terms

Multiple GPUs can also come in handy for cases where applications might benefit from multiple QPUs running in parallel.  The `mqpu` backend uses multiple GPUs to simulate QPUs so you can accelerate quantum applications with parallelization.

![figure](https://nvidia.github.io/cuda-quantum/latest/_images/mqpu.png)

The most simple example is Hamiltonian Batching. In this case, an expectation value of a large Hamiltonian is distributed across multiple simulated QPUs, where each QPUs evaluates a subset of the Hamiltonian terms.

![figure](https://nvidia.github.io/cuda-quantum/latest/_images/hsplit.png)

The code below evaluates the expectation value of a random 100000 term Hamiltonian. A standard `observe` call will run the program on a single GPU.  Adding the argument `execution=cudaq.parallel.thread` or `execution=cudaq.parallel.mpi` will automatically distribute the Hamiltonian terms across multiple GPUs on a single node or multiple GPUs on multiple nodes, respectively.

The code is executed with `mpiexec -np n python3 program.py --target nvidia --target-option mqpu` where $n$ is the number of GPUs available.

```python
import cudaq
from cudaq import spin

if cudaq.num_available_gpus() == 0:
    print("This example requires a GPU to run. No GPU detected.")
    exit(0)

cudaq.set_target("nvidia", option="mqpu")
cudaq.mpi.initialize()

qubit_count = 15
term_count = 100000

@cudaq.kernel
def kernel(n_qubits: int):

    qubits = cudaq.qvector(n_qubits)

    h(qubits[0])
    for i in range(1, n_qubits):
        x.ctrl(qubits[0], qubits[i])

# Create a random Hamiltonian
hamiltonian = cudaq.SpinOperator.random(qubit_count, term_count)

# The observe calls allows calculation of the the expectation value of the Hamiltonian with respect to a specified kernel.

# Single node, single GPU.
result = cudaq.observe(kernel, hamiltonian, qubit_count)
result.expectation()

# If multiple GPUs/ QPUs are available, the computation can parallelize with the addition of an argument in the observe call.

# Single node, multi-GPU.
result = cudaq.observe(kernel,
                       hamiltonian,
                       qubit_count,
                       execution=cudaq.parallel.thread)
result.expectation()

# Multi-node, multi-GPU.
result = cudaq.observe(kernel,
                       hamiltonian,
                       qubit_count,
                       execution=cudaq.parallel.mpi)
result.expectation()

cudaq.mpi.finalize()
```

##### Circuit Batching

A second way to leverage the `mqpu` backend is to batch circuit evaluations across multiple simulated QPUs.

![figure](https://nvidia.github.io/cuda-quantum/latest/_images/circsplit.png)

A common use is evaluating a parameterized circuit many times with different parameters; the code below prepares 10000 parameter sets for a 5-qubit circuit.

```python
import cudaq
from cudaq import spin
import numpy as np

if cudaq.num_available_gpus() == 0:
    print("This example requires a GPU to run. No GPU detected.")
    exit(0)

np.random.seed(1)
cudaq.set_target("nvidia")

qubit_count = 5
sample_count = 10000
h = spin.z(0)
parameter_count = qubit_count

# prepare 10000 different input parameter sets.
parameters = np.random.default_rng(13).uniform(low=0,
                                               high=1,
                                               size=(sample_count,
                                                     parameter_count))

@cudaq.kernel
def kernel(params: list[float]):

    qubits = cudaq.qvector(5)

    for i in range(5):
        rx(params[i], qubits[i])
```

These circuits can be broadcast through a single `observe` call (by default on one GPU); the code below times the process.

```python
import time

start_time = time.time()
cudaq.observe(kernel, h, parameters)
end_time = time.time()
print(end_time - start_time)
```

```text
3.185340642929077
```

Batching across multiple QPUs accelerates this greatly; first slice the parameter list into smaller arrays — here into four, for four GPUs.

```python
print('There are', parameters.shape[0], 'parameter sets to execute')

xi = np.split(
    parameters,
    4)  # Split the parameters into 4 arrays since 4 GPUs are available.

print('Split parameters into', len(xi), 'batches of', xi[0].shape[0], ',',
      xi[1].shape[0], ',', xi[2].shape[0], ',', xi[3].shape[0])
```

```text
There are now 10000 parameter sets split into 4 batches of 2500 , 2500 , 2500 , 2500
```

Asynchronous results are collected in a list (`asyncresults`) and retrieved later with `get`. The loop below passes each parameter set to `observe_async` with a `qpu_id` designating which of the four GPUs runs it. Expect up to a 4x speedup, varying by problem size.

```python
# Timing the execution on a single GPU vs 4 GPUs,
# one will see a nearly 4x performance improvement if 4 GPUs are available.

cudaq.set_target("nvidia", option="mqpu")
asyncresults = []
num_gpus = cudaq.num_available_gpus()

start_time = time.time()
for i in range(len(xi)):
    for j in range(xi[i].shape[0]):
        qpu_id = i * num_gpus // len(xi)
        asyncresults.append(
            cudaq.observe_async(kernel, h, xi[i][j, :], qpu_id=qpu_id))
result = [res.get() for res in asyncresults]
end_time = time.time()
print(end_time - start_time)
```

```text
1.1754660606384277
```

### 9. Optimizers & Gradients

> Source: https://nvidia.github.io/cuda-quantum/latest/examples/python/optimizers_gradients.html

Many quantum algorithms require the optimization of quantum circuit parameters with respect to an expectation value. CUDA-Q provides a comprehensive suite of optimization tools for hybrid quantum-classical algorithms like VQE (Variational Quantum Eigensolver).

This notebook will demonstrate:

1. **Built-in CUDA-Q Optimizers**: Adam, SGD, SPSA, COBYLA, NelderMead, LBFGS, and GradientDescent
2. **Optimizer Parameters**: Detailed configuration options with defaults and tuning guidance
3. **Gradient Strategies**: CentralDifference, ForwardDifference, and ParameterShift
4. **Third-Party Optimizers**: Integration with SciPy
5. **Parallel Parameter Shift**: Multi-GPU gradient computation

#### CUDA-Q Optimizer Overview

CUDA-Q includes the following optimizers:

##### Gradient-Free Optimizers (no gradients required):
- **COBYLA**: Constrained Optimization BY Linear Approximations
- **NelderMead**: Simplex-based derivative-free optimizer
- **SPSA**: Simultaneous Perturbation Stochastic Approximation (excellent for noisy functions)

##### Gradient-Based Optimizers (require gradients):
- **Adam**: Adaptive Moment Estimation with momentum (recommended for most cases)
- **SGD**: Stochastic Gradient Descent
- **LBFGS**: Limited-memory BFGS quasi-Newton method
- **GradientDescent**: Basic gradient descent

First, let's set up the kernel and Hamiltonian that we'll use throughout the examples.

```python
import cudaq
from cudaq import spin
import numpy as np

hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

@cudaq.kernel
def kernel(angles: list[float]):
    qubits = cudaq.qvector(2)
    x(qubits[0])
    ry(angles[0], qubits[1])
    x.ctrl(qubits[1], qubits[0])  

initial_params = np.random.normal(0, np.pi, 1)
```

#### 1. Built-in CUDA-Q Optimizers and Gradients

CUDA-Q provides several optimizers with configurable parameters. Let's explore the most commonly used optimizers: **Adam**, **SGD**, and **SPSA**.

##### 1.1 Adam Optimizer with Parameter Configuration

**Adam (Adaptive Moment Estimation)** combines momentum and adaptive learning rates for efficient optimization. It's particularly effective for problems with noisy gradients.

**Configurable parameters:** `step_size`, `beta1`, `beta2`, `epsilon`, `batch_size`, `f_tol`, `max_iterations`, `initial_parameters` — defaults and semantics: see [Optimizers](#optimizers) in Part 5.

The optimizer and gradient are specified below. An objective function is defined which uses a lambda expression to evaluate the cost (a CUDA-Q `observe` expectation value). The gradient is calculated using the `compute` method.

```python
# Configure Adam optimizer with custom parameters
optimizer = cudaq.optimizers.Adam()
optimizer.step_size = 0.1                      # Learning rate
optimizer.beta1 = 0.9                          # First moment decay
optimizer.beta2 = 0.999                        # Second moment decay
optimizer.epsilon = 1e-8                       # Numerical stability
optimizer.max_iterations = 100                 # Maximum iterations
optimizer.initial_parameters = initial_params  # Set initial parameters

# Use CentralDifference gradient strategy
gradient = cudaq.gradients.CentralDifference()

def objective_function(parameter_vector: list[float],
                       hamiltonian=hamiltonian,
                       gradient_strategy=gradient,
                       kernel=kernel) -> tuple[float, list[float]]:
    """
    Objective function for gradient-based optimizers.
    Returns: (cost, gradient_vector)
    """
    get_result = lambda parameter_vector: cudaq.observe(kernel, hamiltonian, parameter_vector).expectation()

    cost = get_result(parameter_vector)
    
    gradient_vector = gradient_strategy.compute(parameter_vector, get_result, cost)

    return cost, gradient_vector
```

Now run the optimizer to find the optimal energy and parameters. Adam will use adaptive learning rates for each parameter.

```python
energy, parameter = optimizer.optimize(dimensions=1, function=objective_function)

print(f"\n=== Adam Optimizer Results ===")
print(f"Minimized <H> = {energy:.6f}")
print(f"Optimal parameters: {[round(p, 6) for p in parameter]}")
```

Output:

```text

=== Adam Optimizer Results ===
Minimized <H> = -1.744713
Optimal parameters: [-5.721116]
```

##### 1.2 SGD (Stochastic Gradient Descent) Optimizer

**SGD** is a fundamental optimization algorithm that updates parameters by taking steps proportional to the negative gradient.

**Configurable parameters:** `step_size`, `batch_size`, `f_tol`, `max_iterations`, `initial_parameters` (see [Optimizers](#optimizers) in Part 5).

SGD is simpler than Adam and can be effective when you understand your problem well enough to tune the learning rate appropriately.

```python
# Configure SGD optimizer
sgd_optimizer = cudaq.optimizers.SGD()
sgd_optimizer.step_size = 0.05       # Learning rate
sgd_optimizer.batch_size = 1         # Stochastic mode
sgd_optimizer.max_iterations = 100   # Maximum iterations
sgd_optimizer.f_tol = 1e-6           # Convergence tolerance
sgd_optimizer.initial_parameters = initial_params

# Run optimization
sgd_energy, sgd_params = sgd_optimizer.optimize(dimensions=1, function=objective_function)

print(f"\n=== SGD Optimizer Results ===")
print(f"Minimized <H> = {sgd_energy:.6f}")
print(f"Optimal parameters: {[round(p, 6) for p in sgd_params]}")
```

Output:

```text

=== SGD Optimizer Results ===
Minimized <H> = -1.748865
Optimal parameters: [-5.688733]
```

##### 1.3 SPSA (Simultaneous Perturbation Stochastic Approximation)

**SPSA** is a gradient-free stochastic optimization algorithm that is particularly useful for noisy objective functions (like quantum hardware with shot noise). It approximates gradients using simultaneous perturbations and requires only **2 function evaluations per iteration** regardless of problem dimension.

**Configurable parameters:** `step_size`, `gamma`, `max_iterations`, `initial_parameters` (see [Optimizers](#optimizers) in Part 5).

**Key Advantage**: SPSA does **not** require gradients, making it ideal for noisy functions and quantum hardware.

```python
# Configure SPSA optimizer
spsa_optimizer = cudaq.optimizers.SPSA()
spsa_optimizer.step_size = 0.3       # Evaluation step size
spsa_optimizer.gamma = 0.101         # Scaling exponent
spsa_optimizer.max_iterations = 100  # Maximum iterations
spsa_optimizer.initial_parameters = initial_params

# Define gradient-free objective function
def spsa_objective(parameter_vector: list[float]) -> float:
    """
    Objective function for gradient-free optimizers like SPSA.
    Returns: cost only (no gradient)
    """
    return cudaq.observe(kernel, hamiltonian, parameter_vector).expectation()

# Run optimization
spsa_energy, spsa_params = spsa_optimizer.optimize(dimensions=1, function=spsa_objective)

print(f"\n=== SPSA Optimizer Results ===")
print(f"Minimized <H> = {round(spsa_energy, 6)}")
print(f"Optimal parameters: {[round(p, 6) for p in spsa_params]}")
```

Output:

```text

=== SPSA Optimizer Results ===
Minimized <H> = -1.748668
Optimal parameters: [-5.681724]
```

#### 2. Third-Party Optimizers

CUDA-Q works alongside third-party optimization libraries like SciPy: the same VQE procedure can be run by defining a simple cost function and passing it to SciPy's standard `minimize`.

```python
from scipy.optimize import minimize

def cost(theta):

    exp_val = cudaq.observe(kernel, hamiltonian, theta).expectation()

    return exp_val

result = minimize(cost, initial_params ,method='COBYLA', options={'maxiter': 40})
print(result)
```

Output:

```text
 message: Optimization terminated successfully.
 success: True
  status: 1
     fun: -1.748865011330396
       x: [ 5.943e-01]
    nfev: 26
   maxcv: 0.0
```

#### 3. Parallel Parameter Shift Gradients

CUDA-Q's `mqpu` backend allows for parallel computation of parameter shift gradients using multiple simulated QPUs. Gradients computed this way can be used in any of the previously discussed optimization procedures.  Below is an example demonstrating how parallel gradient evaluation can be used for a VQE procedure. 

The parameter shift procedure computes two expectations values for each parameter shifted forwards and backwards. These are used to estimate the gradient contribution for that parameter.

The function below builds shifted parameter arrays `xplus`/`xminus`, then loops over the parameters using `cudaq.observe_async`, whose `qpu_id` argument selects the GPU that simulates each QPU (below, the gradient is batched over four available devices). The `g_plus`/`g_minus` results (accessed like `g_plus[1].expectation()`) are combined into finite differences to construct the gradient.

```python
import  numpy as np
# cudaq.set_target('nvidia', option = 'mqpu')

num_qpus = 1
epsilon =np.pi/4

def batched_gradient_function(kernel, parameters, hamiltonian, epsilon): 

    # Prepare an array of parameters corresponding to the plus and minus shifts
    x = np.tile(parameters, (len(parameters),1))
    xplus = x + (np.eye(x.shape[0]) * epsilon)
    xminus = x - (np.eye(x.shape[0]) * epsilon)

    g_plus = []
    g_minus = []
    gradient = []

    qpu_counter = 0 # Iterate over the number of GPU resources available
    
    
    for i in range(x.shape[0]): 

        g_plus.append(cudaq.observe_async(kernel,hamiltonian, xplus[i], qpu_id = qpu_counter%num_qpus))
        qpu_counter += 1 

        g_minus.append(cudaq.observe_async(kernel, hamiltonian, xminus[i], qpu_id = qpu_counter%num_qpus))
        qpu_counter += 1 
        
    # Use the expectation values to compute the gradient    
    gradient = [(g_plus[i].get().expectation() - g_minus[i].get().expectation()) / (2*epsilon) for i in range(len(g_minus))]

    return gradient
```

This function can be used in a VQE procedure as presented below. The `batched_gradient_function` is used to evaluate the gradient at each optimization step. This objective function returns the cost and gradient at the current parameter values and can be used with any SciPy optimizer that uses gradients (like L-BFGS-B).

```python
def objective_function(parameter_vector,
                       hamiltonian=hamiltonian,
                       kernel=kernel,
                       epsilon=epsilon):
    """
    Objective function for VQE with parallel parameter shift gradients.
    Computes both cost and gradient at the current parameter values.
    """
    # Compute cost at current parameters
    cost = cudaq.observe(kernel, hamiltonian, parameter_vector).expectation()
    
    # Compute gradient at current parameters using parallel parameter shift
    gradient_vector = batched_gradient_function(kernel, parameter_vector, hamiltonian, epsilon)

    return cost, gradient_vector
```

```python
# Run VQE optimization with parallel parameter shift gradients
result_vqe = minimize(objective_function, initial_params, method='L-BFGS-B', jac=True, tol=1e-8, options={'maxiter': 50})

print("\n=== VQE with Parallel Parameter Shift Gradients ===")
print(f"Optimized energy: {result_vqe.fun:.6f}")
print(f"Optimal parameters: {result_vqe.x}")
print(f"Number of iterations: {result_vqe.nit}")
print(f"Success: {result_vqe.success}")
```

### 10. Noisy Simulations

> Source: https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html

Quantum noise divides into coherent errors (systematic, e.g. device miscalibration implementing a rotation $\theta + \epsilon$ instead of $\theta$) and incoherent errors (decoherence — entanglement with the environment — yielding mixed states described by the density matrix formalism). Incoherent noise is modeled via quantum channels: linear, completely positive, trace-preserving maps given by Kraus operators $\{ K_i \}$ satisfying $\sum_{i} K_i^\dagger K_i = \mathbb{I}$.

The bit-flip channel flips the qubit with probability $p$ and leaves it unchanged with probability $1-p$. This can be represented by employing Kraus operators: 

$$K_0 = \sqrt{1-p} \begin{pmatrix}
    1 & 0 \\
    0 & 1
\end{pmatrix}$$

$$K_1 = \sqrt{p} \begin{pmatrix}
  0 & 1 \\
  1 & 0
\end{pmatrix}$$

Let's implement the bit-flip channel using CUDA-Q:

```python
import cudaq
from cudaq import spin

import numpy as np

# This example chooses the density matrix simulator target to model quantum noise.
# CUDA-Q also supports noisy trajectory simulation on statevector and tensor-network targets, including through PTSBE.
cudaq.set_target("density-matrix-cpu")
```

```python
# Let's define a simple kernel that we will add noise to.
qubit_count = 2

@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    x(qvector)

print(cudaq.draw(kernel, qubit_count))
```

Output:

```text
     ╭───╮
q0 : ┤ x ├
     ├───┤
q1 : ┤ x ├
     ╰───╯
```

```python
# In the ideal noiseless case, we get |11> 100% of the time.

ideal_counts = cudaq.sample(kernel, qubit_count, shots_count=1000)
ideal_counts.dump()
```

Output:

```text
{ 11:1000 }
```

```python
# First, we will define an out of the box noise channel. In this case,
# we choose depolarization noise. This depolarization will result in
# the qubit state decaying into a mix of the basis states, |0> and |1>,
# with our provided probability.
error_probability = 0.1
depolarization_channel = cudaq.DepolarizationChannel(error_probability)

# Other built in noise models 
bit_flip = cudaq.BitFlipChannel(error_probability) 
phase_flip = cudaq.PhaseFlipChannel(error_probability)
amplitude_damping = cudaq.AmplitudeDampingChannel(error_probability)

# We can also define our own, custom noise channels through
# Kraus operators. Here we will define two operators representing
# bit flip errors.

# Define the Kraus Error Operator as a complex ndarray.
kraus_0 = np.sqrt(1 - error_probability) * np.array([[1.0, 0.0], 
                                                     [0.0, 1.0]],
                                                    dtype=np.complex128)

kraus_1 = np.sqrt(error_probability) * np.array([[0.0, 1.0], 
                                                 [1.0, 0.0]],
                                                dtype=np.complex128)

# Add the Kraus Operator to create a quantum channel.
bitflip_channel = cudaq.KrausChannel([kraus_0, kraus_1])

# Add noise channels to our noise model.
noise_model = cudaq.NoiseModel()

# Apply the depolarization channel to any X-gate on the 0th qubit.
noise_model.add_channel("x", [0], depolarization_channel)
# Apply the bitflip channel to any X-gate on the 1st qubit.
noise_model.add_channel("x", [1], bitflip_channel)

# Due to the impact of noise, our measurements will no longer be uniformly
# in the |11> state.
noisy_counts = cudaq.sample(kernel,
                            qubit_count,
                            noise_model=noise_model,
                            shots_count=1000)
noisy_counts.dump()
```

Output:

```text
{ 00:8 01:54 10:95 11:843 }
```

```python
# We can also use noise models with the observe function

hamiltonian = spin.z(0)

noisy_result = cudaq.observe(kernel,
                             hamiltonian,
                             qubit_count,
                             noise_model=noise_model)

noisy_result.expectation()
```

Output:

```text
-0.8666666666666666
```

In addition to gate-based noise injection, CUDA-Q also supports fine-grained, explicit noise injection via the `cudaq.apply_noise` function. You can place this method in your kernels, and inject specific noise on specific qubits wherever you need.

```python
@cudaq.kernel
def inject_noise_example():
    q = cudaq.qvector(3)

    # Apply depolarization noise to the first qubit
    cudaq.apply_noise(cudaq.DepolarizationChannel, 0.1, q[0])

    # Perform gate operations
    h(q[1])
    x.ctrl(q[1], q[2])

    # Inject a Y error into the second qubit
    cudaq.apply_noise(cudaq.YError, 0.1, q[1])

    # Apply a general Pauli noise channel to the third qubit, where the 3 values indicate the probability of X, Y, and Z errors.
    cudaq.apply_noise(cudaq.Pauli1, 0.1, 0.1, 0.1, q[2])

# Define and apply a noise model
noise = cudaq.NoiseModel()
counts = cudaq.sample(inject_noise_example, noise_model=noise)
print(counts)
```

Output:

```text
{ 000:351 001:117 010:115 011:345 100:29 101:7 110:5 111:31 }
```

### 11. Pre-Trajectory Sampling with Batch Execution (PTSBE)

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/ptsbe.html

Pre-Trajectory Sampling with Batch Execution (PTSBE) is a method for
efficiently sampling noisy quantum circuits [Patti2025]. It builds on
quantum trajectory methods [Carmichael2007], which simulate noise by
stochastically selecting a Kraus operator at each noise site and evolving a
pure statevector rather than the full density matrix. Traditional trajectory
methods construct a new statevector for every measurement shot.
PTSBE instead pre-samples a set of unique noise
realizations (trajectories) and batches shots across them, so the number of
statevector simulations scales with unique trajectories but not the total shots.
Since noise pre-sampling and state post-sampling are only polynomial-complexity tasks while state construction is exponential, PTSBE gathers noisy quantum data orders of magnitude faster than traditional trajectory sampling — capturing vastly more shot data (e.g., training data for ML tasks such as AI decoders), or deployed proportionally to reproduce the exact statistics of the problem at a considerable speedup. PTSBE matches traditional trajectory-simulation accuracy at a fraction of the computational cost when the number of unique trajectories (errors) is much smaller than the total shot count [Patti2025].

#### Conceptual Overview

Noise applies a set of Kraus operators at each gate location; at each noise site the environment selects one with some probability. A **trajectory** is one complete assignment of Kraus operators across all noise sites in the circuit; its probability is the product of the chosen operators' probabilities.

PTSBE works in three phases:

| Phase | Name | Description |
| --- | --- | --- |
| 1 | Trajectory Sampling | Draw *T* unique trajectories from the full noise space using a sampling strategy. Each trajectory specifies which Kraus operator fires at every noise site. |
| 2 | Shot Allocation | Distribute the total *N* shots across the *T* trajectories according to a shot allocation strategy (e.g. proportional to trajectory probability). |
| 3 | Batch Execution | Simulate each trajectory as a pure-state circuit. The per-trajectory measurement outcomes are merged into a single `SampleResult`. |

Because trajectories are reused across many shots, the number of circuit
simulations scales with the number of unique trajectories *T*, not the shot
count *N*.

#### When to Use PTSBE

PTSBE is most beneficial when:

- The circuit has few distinct noise sites so the trajectory space is
  manageable.
- A large shot count is required (1 000 – 1 000 000+) so the reuse of
  trajectories provides a significant speed-up.
- The shots are intended for a data-hungry downstream task that is not necessarily
  inhibited by correlated sampling, such as training AI models

Benchmarks from the original paper [Patti2025] illustrate the potential
speed-ups:

- 35-qubit statevector simulation (magic state distillation): up to
  10⁶× speedup over conventional trajectory methods, producing one
  trillion shots on 4 NVIDIA H100 GPUs.
- 85-qubit tensor network simulation (magic state distillation): 16×
  speedup, producing one million shots.

PTSBE is particularly well-suited for generating large synthetic datasets of
noisy measurement outcomes, such as training data for machine-learning–based
quantum error correction (QEC) decoders [Patti2025].

PTSBE requires:

- A static circuit — no mid-circuit measurements or
  measurement-dependent conditional logic.
- A local simulator backend.

#### Quick Start

The example below simulates a two-qubit Bell circuit under depolarizing noise.

```python
import cudaq
from cudaq import ptsbe
from utils import bell, noise

result = ptsbe.sample(bell, shots_count=10_000, noise_model=noise)
print(result)
```

#### Usage Tutorial

##### Controlling the Number of Trajectories

By default, PTSBE generates up to `shots_count` unique trajectories.
For large shot counts set `max_trajectories` to cap trajectory generation
and gain the batching benefit:

```python
result = ptsbe.sample(
    bell,
    shots_count=100_000,
    noise_model=noise,
    max_trajectories=500,
)
```

##### Choosing a Trajectory Sampling Strategy

Three Python strategies control which trajectories are selected from the noise space.

- `cudaq.ptsbe.ProbabilisticSamplingStrategy` (default)
- `cudaq.ptsbe.OrderedSamplingStrategy`
- `cudaq.ptsbe.ExhaustiveSamplingStrategy`

```python
import cudaq
from utils import bell, noise

# Reproducible probabilistic sampling
result = cudaq.ptsbe.sample(
    bell,
    shots_count=10_000,
    noise_model=noise,
    sampling_strategy=cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=42),
)
print(result)

# Top-100 trajectories by probability
result = cudaq.ptsbe.sample(
    bell,
    shots_count=10_000,
    noise_model=noise,
    max_trajectories=100,
    sampling_strategy=cudaq.ptsbe.OrderedSamplingStrategy(),
)
print(result)
```

##### Shot Allocation Strategies

After trajectories are selected, shots are distributed across them:

Allocation types (`PROPORTIONAL`, `UNIFORM`, `LOW_WEIGHT_BIAS`, `HIGH_WEIGHT_BIAS`) and the `bias_strength` field: see the [PTSBE Submodule](#ptsbe-submodule) in Part 5.

```python
import cudaq
from utils import bell, noise

alloc = cudaq.ptsbe.ShotAllocationStrategy(
    cudaq.ptsbe.ShotAllocationType.LOW_WEIGHT_BIAS, bias_strength=2.0)

result = cudaq.ptsbe.sample(
    bell,
    shots_count=10_000,
    noise_model=noise,
    shot_allocation=alloc,
)
print(result)
```

##### Inspecting Execution Data

Set `return_execution_data=True` to attach the full execution trace —
circuit instructions, sampled trajectories, and per-trajectory counts — to
the result. This API is experimental and may be subject to change in future releases.

```python
import cudaq
from utils import bell, noise

result = cudaq.ptsbe.sample(
    bell,
    shots_count=1_000,
    noise_model=noise,
    return_execution_data=True,
)

data = result.ptsbe_execution_data

# Circuit structure. For Noise instructions, ``inst.params`` carries the
# channel's numeric parameters and ``inst.channel`` is a ``cudaq.KrausChannel``
# exposing ``.noise_type``, ``.parameters``, and ``.get_ops()``. For Gate and
# Measurement instructions ``inst.channel`` is ``None``.
Noise = cudaq.ptsbe.TraceInstructionType.Noise
for inst in data.instructions:
    print(inst.type, inst.name, inst.targets)
    if inst.type == Noise:
        print(f"  params={list(inst.params)}  "
              f"noise_type={inst.channel.noise_type}  "
              f"num_kraus_ops={len(inst.channel.get_ops())}")

# Trajectory details
for trajectory in data.trajectories:
    print(f"id={trajectory.trajectory_id}  p={trajectory.probability:.4f}"
          f"  shots={trajectory.num_shots}")
```

###### Trajectory vs Shot Trade-offs

The central tension in PTSBE is between trajectory count *T* and
shots per trajectory *N/T*.

Using more trajectories covers more of the noise space and reduces bias in the
estimated distribution, but since each trajectory is simulated independently the
simulation cost scales linearly with *T*.

Fewer trajectories mean each one accumulates more shots, which reduces
shot-noise variance per trajectory and lowers wall-clock time.

**Practical guidance**

As a rule of thumb, `max_trajectories` between 100 and 10 000 covers the
majority of practical use cases. Below 100, bias may dominate. Above 10 000,
the simulation cost approaches that of a conventional density-matrix run.
Useful to perform a warm-up run sweeping the number of trajectories to understand
the convergence behavior.

###### Backend Requirements

PTSBE requires a backend that supports trajectory-based noisy simulation.
The supported targets are those marked with `*` in the
[simulator table](https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html),
plus `density-matrix-cpu` and `qpp-cpu`:

Supported targets: `nvidia` (single GPU, general purpose), `nvidia, option=mgpu` (33+ qubits), `nvidia, option=mqpu` (multi-QPU async distribution), `tensornet` (exact, shallow circuits, thousands of qubits), `tensornet-mps` (approximate MPS, square-shaped circuits), and `qpp-cpu` (CPU, < 28 qubits) — see Part 3 for full backend details.

Set the target:

```python
# Single GPU (most common for production)
cudaq.set_target("nvidia")

# CPU density matrix
cudaq.set_target("density-matrix-cpu")

# Multi-GPU for large circuits
cudaq.set_target("nvidia", option="mgpu")

# Tensor network for wide shallow circuits
cudaq.set_target("tensornet")
```

See [backends](https://nvidia.github.io/cuda-quantum/latest/using/backends/backends.html) for full details on each target including precision and
qubit count limits.

###### References

- **[Carmichael2007]** Carmichael, H. J. *Quantum jumps revisited: An overview of quantum trajectory theory.* Quantum Future From Volta and Como to the Present and Beyond: Proceedings of the Xth Max Born Symposium Held in Przesieka, Poland, 24–27 September 1997. Berlin, Heidelberg: Springer Berlin Heidelberg, 2007. https://link.springer.com/chapter/10.1007/bfb0105336

- **[Patti2025]** Taylor L. Patti, Thien Nguyen, Justin G. Lietz, Alexander J. McCaskey, Brucek Khailany, *Augmenting Simulated Noisy Quantum Data Collection by Orders of Magnitude Using Pre-Trajectory Sampling with Batched Execution.* Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2025. https://arxiv.org/abs/2504.16297

### 12. Detector Error Models

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/dem_from_kernel.html

A *detector error model* (DEM) is a detector error matrix (capturing which
detectors each error mechanism flips) together with a noise model that
assigns a likelihood to each error mechanism. It is the input a decoder needs
to infer which errors occurred from a circuit's measurement record.

In CUDA-Q you declare the parity checks (*detectors*) and *logical observables*
directly inside a kernel, then extract the DEM with
`cudaq.dem_from_kernel`
as text in Stim's standard
[.dem file format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md),
which `stim.DetectorErrorModel` parses back into a decoder-ready model. The
measurements that feed the declarations are *measurement handles*
([measuring kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/measuring_kernels.html)).

Three kernel-side declarations are available:
`detector(m0, m1, ...)` declares one detector as a parity
constraint over the given measurements; `detectors(prev, curr)` declares `N`
detectors by pairing two equal-length handle vectors element-wise (the
standard form for cross-round detectors); and
`logical_observable(m0, m1, ...)` declares a logical observable.

The example below is a three-qubit bit-flip memory experiment: each round
measures the data qubits and pairs them with the previous round via
`detectors`, with a final `logical_observable` reading out the register.
In-kernel `apply_noise` seeds the error mechanisms. Each call applies a
single-qubit bit-flip channel (`cudaq::x_error` in C++, `cudaq.XError` in
Python) that applies a Pauli `X` with the given probability, so a flipped
data qubit shows up as a parity change in the next `detectors` pair.
See the [Python API reference](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html) for `apply_noise`
and the other predefined noise channels.

```python
# A 3-qubit bit-flip memory experiment. Each round measures the data qubits;
# cross-round detectors pair each measurement with its value in the previous
# round, and a final logical observable reads out the register. In-kernel
# `apply_noise` seeds the error mechanisms the detector error model reports.
@cudaq.kernel
def memory_experiment(rounds: int):
    data = cudaq.qvector(3)
    prev = mz(data)

    for r in range(rounds):
        cudaq.apply_noise(cudaq.XError, 0.01, data[0])
        cudaq.apply_noise(cudaq.XError, 0.01, data[1])
        cudaq.apply_noise(cudaq.XError, 0.01, data[2])

        curr = mz(data)
        # One detector per qubit, pairing this round with the previous one.
        cudaq.detectors(prev, curr)
        prev = curr

    cudaq.logical_observable(prev[0], prev[1], prev[2])
```

Pass the kernel (and a noise model) to `dem_from_kernel` to extract the DEM.

```python
# Generate the detector error model as Stim `.dem` text. A noise model must be
# supplied for the in-kernel `apply_noise` mechanisms to take effect. Parse the
# text with `stim.DetectorErrorModel(dem)` to drive a decoder.
noise = cudaq.NoiseModel()
dem = cudaq.dem_from_kernel(memory_experiment, 2, noise_model=noise)
print(dem)
```

The `.dem` text is a list of independent error mechanisms. Each
`error(p) D... L...` line gives one mechanism: its probability `p`, the
detectors it flips (its *symptoms*, `D`), and the logical observables it
flips (its *frame changes*, `L`).

Output DEM: With two rounds and three data qubits, there are six independent
`error` mechanisms: one bit-flip per qubit per round at the in-kernel
probability `0.01` (printed at full floating-point precision). Each error
flips one detector together with the logical observable `L0`:

```text
error(0.01000000000000000021) D0 L0
error(0.01000000000000000021) D1 L0
error(0.01000000000000000021) D2 L0
error(0.01000000000000000021) D3 L0
error(0.01000000000000000021) D4 L0
error(0.01000000000000000021) D5 L0
```

#### Limitations

* **Stabilizer (Clifford) circuits only.** The DEM formalism requires detectors
  to be deterministic under noise-free execution, which is only well defined for
  Clifford circuits; a non-Clifford gate raises a diagnostic.
* **No measurement-conditional control flow.** Branching on a measurement result
  changes the measurement count shot-to-shot and breaks the detector matrix
  model; such kernels are rejected.
* **Independent Pauli noise.** Each error mechanism is assumed independent.
* **Pre-decomposition.** The DEM reflects the abstract kernel circuit, not the
  hardware-decomposed circuit.

### 13. Constructing Operators

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/operators.html

This section covers defining and using spin operators, plus tools for more sophisticated operators.

#### Constructing Spin Operators

The `spin_op` type provides an abstraction for a general tensor product of Pauli spin operators, and their sums.

Spin operators are constructed using the `spin.z()`, `spin.y()`, `spin.x()`, and `spin.i()` functions, corresponding to the $Z$, $Y$, $X$, and $I$ Pauli operators. For example, `spin.z(0)` corresponds to a Pauli Z
operation acting on qubit 0. The example below demonstrates how to construct the following operator
2 $XYX$ - 3 $ZZY$.

```python
import cudaq
from cudaq import spin
```

There are a number of convenient methods for combining, comparing, iterating through, and extracting information from spin operators and can be referenced [here](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.SpinOperator) in the API.

#### Pauli Words and Exponentiating Pauli Words

The `pauli_word` type specifies a string of Pauli operations (e.g. `XYXZ`) and is convenient for applying operations based on exponentiated Pauli words. The code below demonstrates how a list of Pauli words, along with their coefficients, are provided as kernel inputs and converted into operators by the `exp_pauli` function.

The code below applies the following operation: $e^{i(0.432XYZ)} + e^{i(0.324IXX)}$

```python
words = ['XYZ', 'IXX']
coefficients = [0.432, 0.324]

@cudaq.kernel
def kernel(coefficients: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(3)

    for i in range(len(coefficients)):
        exp_pauli(coefficients[i], q, words[i])
```

### 14. Performance Optimizations

> Source: https://nvidia.github.io/cuda-quantum/latest/examples/python/performance_optimizations.html

This section highlights features advanced users can leverage to increase performance.

#### Gate Fusion

Gate fusion combines consecutive gates into a single operation (see figure below; fusion defaults and environment variables are covered in Part 3). Set `CUDAQ_MGPU_FUSE` to select the fusion degree, e.g. `CUDAQ_MGPU_FUSE=4 python c2h2VQE.py --target nvidia --target-option fp64,mgpu`

![figure](https://nvidia.github.io/cuda-quantum/latest/_images/gate-fuse.png)

The importance of gate fusion is system dependent, but can have a large influence on the performance of the simulation.  See the example below for a 24 qubit VQE experiment where changing the fusion level resulted in significant performance boosts.

![figure](https://nvidia.github.io/cuda-quantum/latest/_images/gatefusion.png)

### 15. Using Quantum Hardware Providers

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/hardware_providers.html

CUDA-Q contains support for using a set of hardware providers.
For more information about executing quantum kernels on different hardware
backends, please take a look at [hardware](https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware.html). Each subsection below shows how to run kernels on that provider's backends.

#### Amazon Braket

```python
import cudaq

# NOTE: Amazon Braket credentials must be set before running this program.
# Amazon Braket costs apply.
cudaq.set_target("braket")

# The default device is SV1, state vector simulator. Users may choose any of
# the available devices by supplying its `ARN` with the `machine` parameter.
# For example,
# ```
# cudaq.set_target("braket", machine="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
# ```

# Create the kernel we'd like to execute
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])

# Execute and print out the results.

# Option A:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by Amazon Braket.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

async_counts = async_results.get()
print(async_counts)

# Option B:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from Amazon Braket.
counts = cudaq.sample(kernel)
print(counts)
```

#### Anyon Technologies

```python
import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`

# To use the Anyon target you will need to set up credentials in `~/.anyon_config`
# The configuration file should contain your Anyon Technologies username and password:
# credentials: {"username":"<username>","password":"<password>"}

# Set the target to the default QPU
cudaq.set_target("anyon")

# You can specify a specific machine via the machine parameter:
# ```
# cudaq.set_target("anyon", machine="telegraph-8q")
# ```
# or for the larger system:
# ```
# cudaq.set_target("anyon", machine="berkeley-25q")
# ```

# Create the kernel we'd like to execute on Anyon.
@cudaq.kernel
def ghz():
    """Maximally entangled state between 5 qubits."""
    q = cudaq.qvector(5)
    h(q[0])
    for i in range(4):
        x.ctrl(q[i], q[i + 1])
    return mz(q)

# Execute on Anyon and print out the results.

# Option A (recommended):
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# remotely on Anyon's superconducting QPU. This is ideal for
# longer running jobs.
future = cudaq.sample_async(ghz)
# ... classical optimization code can run while job executes ...

# Can write the future to file:
with open("future.txt", "w") as outfile:
    print(future, file=outfile)

# Then come back and read it in later.
with open("future.txt", "r") as infile:
    restored_future = cudaq.AsyncSampleResult(infile.read())

# Get the results of the restored future.
async_counts = restored_future.get()
print("Asynchronous results:")
async_counts.dump()

# Option B:
# By using the synchronous `cudaq.sample`, the kernel
# will be executed on Anyon and the calling thread will be blocked
# until the results are returned.
counts = cudaq.sample(ghz)
print("\nSynchronous results:")
counts.dump()
```

#### Infleqtion

```python
import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`
cudaq.set_target("infleqtion")

# Create the kernel we'd like to execute on Infleqtion.
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])

# Note: All measurements must be terminal when performing the sampling.

# Execute on Infleqtion and print out the results.

# Option A (recommended):
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by the Superstaq API. This is ideal when submitting via a queue
# over the cloud.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

# We can either retrieve the results later in the program with
# ```
# async_counts = async_results.get()
# ```
# or we can also write the job reference (`async_results`) to
# a file and load it later or from a different process.
file = open("future.txt", "w")
file.write(str(async_results))
file.close()

# We can later read the file content and retrieve the job
# information and results.
same_file = open("future.txt", "r")
retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

counts = retrieved_async_results.get()
print(counts)

# Option B:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from Superstaq.
counts = cudaq.sample(kernel)
print(counts)
```

#### IonQ

```python
import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`
cudaq.set_target("ionq")

# Create the kernel we'd like to execute on IonQ.
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])

# Note: All qubits will be measured at the end upon performing
# the sampling. You may encounter a pre-flight error on IonQ
# backends if you include explicit measurements.

# Execute on IonQ and print out the results.

# Option A:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by IonQ. This is ideal when submitting via a queue over
# the cloud.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

# We can either retrieve the results later in the program with
# ```
# async_counts = async_results.get()
# ```
# or we can also write the job reference (`async_results`) to
# a file and load it later or from a different process.
file = open("future.txt", "w")
file.write(str(async_results))
file.close()

# We can later read the file content and retrieve the job
# information and results.
same_file = open("future.txt", "r")
retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

counts = retrieved_async_results.get()
print(counts)

# Option B:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from IonQ.
counts = cudaq.sample(kernel)
print(counts)
```

#### IQM

```python
import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`
cudaq.set_target("iqm", url="http://localhost/")

# Crystal_5 QPU architecture:
#       QB1
#        |
# QB2 - QB3 - QB4
#        |
#       QB5

# Create the kernel we'd like to execute on IQM.
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(5)
    h(qvector[2])  # QB3
    x.ctrl(qvector[2], qvector[0])
    mz(qvector)

# Execute on IQM Server and print out the results.

# Option A:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by IQM Server. This is ideal when submitting via a queue over
# the cloud.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

# We can either retrieve the results later in the program with
# ```
# async_counts = async_results.get()
# ```
# or we can also write the job reference (`async_results`) to
# a file and load it later or from a different process.
file = open("future.txt", "w")
file.write(str(async_results))
file.close()

# We can later read the file content and retrieve the job
# information and results.
same_file = open("future.txt", "r")
retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

counts = retrieved_async_results.get()
print(counts)

# Option B:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from IQM Server.
counts = cudaq.sample(kernel)
print(counts)
```

#### OQC

```python
import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`

# To use the OQC target you will need to set the following environment variables
# OQC_URL
# OQC_EMAIL
# OQC_PASSWORD
# To setup an account, contact oqc_qcaas_support@oxfordquantumcircuits.com

cudaq.set_target("oqc")

# Create the kernel we'd like to execute on OQC.
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])
    mz(qvector)

# Option A:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by OQC. This is ideal when submitting via a queue over
# the cloud.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

# We can either retrieve the results later in the program with
# ```
# async_counts = async_results.get()
# ```
# or we can also write the job reference (`async_results`) to
# a file and load it later or from a different process.
file = open("future.txt", "w")
file.write(str(async_results))
file.close()

# We can later read the file content and retrieve the job
# information and results.
same_file = open("future.txt", "r")
retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

counts = retrieved_async_results.get()
print(counts)

# Option B:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from OQC.
counts = cudaq.sample(kernel)
print(counts)
```

#### Pasqal

For QRMI-routed Pasqal jobs, specify `pasqal` as the target; the `machine`
argument is supplied by QRMI at runtime.

```python
import cudaq
from cudaq.operators import RydbergHamiltonian, ScalarOperator
from cudaq.dynamics import Schedule

# This example illustrates how to use Pasqal's EMU_MPS emulator over Pasqal's cloud via CUDA-Q.
# It uses the direct `pasqal` target with Pasqal credentials.
# For the QRMI-routed flow use a supported cluster and set `machine="qrmi"`
# (see QRMI docs).
#
# To obtain the authentication token for the cloud  we recommend logging in with
# Pasqal's Python SDK. See our documentation https://docs.pasqal.com/cloud/ for more.
#
# Contact Pasqal at help@pasqal.com or through https://community.pasqal.com for assistance.
#
# Visit the documentation portal, https://docs.pasqal.com/, to find further
# documentation on Pasqal's devices, emulators and the cloud platform.
#
# For more details on the EMU_MPS emulator see the documentation of the open-source
# package: https://pasqal-io.github.io/emulators/latest/emu_mps/.
from pasqal_cloud import SDK
import os

# We recommend leaving the password empty in an interactive session as you will be
# prompted to enter it securely via the command line interface.
sdk = SDK(
    username=os.environ.get("PASQAL_USERNAME"),
    password=os.environ.get("PASQAL_PASSWORD", None),
)

os.environ["PASQAL_AUTH_TOKEN"] = str(sdk.user_token())

# It is also mandatory to specify the project against which the execution will be billed.
# Uncomment this line to set it from Python, or export it as an environment variable
# prior to execution. You can find your projects here: https://portal.pasqal.cloud/projects.
# ```
# os.environ['PASQAL_PROJECT_ID'] = 'your project id'
# ```

# Set the target including specifying optional arguments like target machine
cudaq.set_target("pasqal",
                 machine=os.environ.get("PASQAL_MACHINE_TARGET", "EMU_MPS"))

# ```
## To target QPU set FRESNEL as the machine, see our cloud portal for latest machine names
# cudaq.set_target("pasqal", machine="FRESNEL")
# ```

# Define the 2-dimensional atom arrangement
a = 5e-6
register = [(a, 0), (2 * a, 0), (3 * a, 0)]
time_ramp = 0.000001
time_max = 0.000003
# Times for the piece-wise linear waveforms
steps = [0.0, time_ramp, time_max - time_ramp, time_max]
schedule = Schedule(steps, ["t"])
# Rabi frequencies at each step
omega_max = 1000000
delta_end = 1000000
delta_start = 0.0
omega = ScalarOperator(lambda t: omega_max
                       if time_ramp < t.real < time_max else 0.0)
# Global phase at each step
phi = ScalarOperator.const(0.0)
# Global detuning at each step
delta = ScalarOperator(lambda t: delta_end
                       if time_ramp < t.real < time_max else delta_start)

async_result = cudaq.evolve_async(RydbergHamiltonian(atom_sites=register,
                                                     amplitude=omega,
                                                     phase=phi,
                                                     delta_global=delta),
                                  schedule=schedule,
                                  shots_count=100).get()
async_result.dump()

## Sample result
# ```
# {'001': 16, '010': 23, '100': 19, '000': 42}
# ```
```

#### Quantinuum

```python
import cudaq
import os

# You only have to set the target once! No need to redefine it for every
# execution call on your kernel.
# By default, we will submit to the Quantinuum syntax checker.
## NOTE: It is mandatory to specify the Nexus project by name or ID.
# Update and un-comment the line below.
# ```
# cudaq.set_target("quantinuum", project="nexus_project")
# ```
# Or use environment variable
# ```
# os.environ["QUANTINUUM_NEXUS_PROJECT"] = "nexus_project"
# ```
cudaq.set_target("quantinuum",
                 project=os.environ.get("QUANTINUUM_NEXUS_PROJECT", None))

# Create the kernel we'd like to execute on Quantinuum.
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])

# Submit to Quantinuum's endpoint and confirm the program is valid.

# Option A:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been executed by the Quantinuum service.
# We will use the synchronous call to submit to the syntax
# checker to confirm the validity of the program.
syntax_check = cudaq.sample(kernel)
if (syntax_check):
    print("Syntax check passed! Kernel is ready for submission.")

# Now we can update the target to the Quantinuum emulator and
# execute our program.
cudaq.set_target("quantinuum",
                 machine="H2-1E",
                 project=os.environ.get("QUANTINUUM_NEXUS_PROJECT", None))

# Option B:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by Quantinuum. This is ideal when submitting via a queue over
# the cloud.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

# We can either retrieve the results later in the program with
# ```
# async_counts = async_results.get()
# ```
# or we can also write the job reference (`async_results`) to
# a file and load it later or from a different process.
file = open("future.txt", "w")
file.write(str(async_results))
file.close()

# We can later read the file content and retrieve the job
# information and results.
same_file = open("future.txt", "r")
retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

counts = retrieved_async_results.get()
print(counts)
```

#### Quantum Circuits, Inc.

```python
import cudaq

# Make sure to export or otherwise present your user token via the environment,
# e.g., using export:
# ```
# export QCI_AUTH_TOKEN="your token here"
# ```
#
# The example will run on QCI's AquSim simulator by default.

cudaq.set_target("qci")

@cudaq.kernel
def teleportation():

    # Initialize a three qubit quantum circuit
    qubits = cudaq.qvector(3)

    # Random quantum state on qubit 0.
    rx(3.14, qubits[0])
    ry(2.71, qubits[0])
    rz(6.62, qubits[0])

    # Create a maximally entangled state on qubits 1 and 2.
    h(qubits[1])
    cx(qubits[1], qubits[2])

    cx(qubits[0], qubits[1])

    h(qubits[0])
    m1 = mz(qubits[0])
    m2 = mz(qubits[1])

    if m1 == 1:
        z(qubits[2])

    if m2 == 1:
        x(qubits[2])

    mz(qubits)

print(cudaq.sample(teleportation))
```

#### Quantum Machines

```python
import cudaq
import math

# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the `url` argument, and set the `api_key`.
cudaq.set_target("quantum_machines",
                 url="http://host.docker.internal:8080",
                 api_key="1234567890",
                 executor="mock")

qubit_count = 5

# Maximally entangled state between 5 qubits
@cudaq.kernel
def all_h():
    qvector = cudaq.qvector(qubit_count)

    for i in range(qubit_count - 1):
        h(qvector[i])

    s(qvector[0])
    r1(math.pi / 2, qvector[1])
    mz(qvector)

# Submit synchronously
cudaq.sample(all_h).dump()
```

#### QuEra Computing

```python
import cudaq
from cudaq.operators import RydbergHamiltonian, ScalarOperator
from cudaq.dynamics import Schedule
import numpy as np

## NOTE: QuEra Aquila system is available via Amazon Braket.
# Credentials must be set before running this program.
# Amazon Braket costs apply.

# This example illustrates how to use QuEra's Aquila device on Braket with CUDA-Q.
# It is a CUDA-Q implementation of the getting started materials for Braket available here:
# https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html

cudaq.set_target("quera")

# Define the 2-dimensional atom arrangement
a = 5.7e-6
register = []
register.append(tuple(np.array([0.5, 0.5 + 1 / np.sqrt(2)]) * a))
register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), 0.5]) * a))
register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), -0.5]) * a))
register.append(tuple(np.array([0.5, -0.5 - 1 / np.sqrt(2)]) * a))
register.append(tuple(np.array([-0.5, -0.5 - 1 / np.sqrt(2)]) * a))
register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), -0.5]) * a))
register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), 0.5]) * a))
register.append(tuple(np.array([-0.5, 0.5 + 1 / np.sqrt(2)]) * a))

time_max = 4e-6  # seconds
time_ramp = 1e-7  # seconds
omega_max = 6300000.0  # rad / sec
delta_start = -5 * omega_max
delta_end = 5 * omega_max

# Times for the piece-wise linear waveforms
steps = [0.0, time_ramp, time_max - time_ramp, time_max]
schedule = Schedule(steps, ["t"])
# Rabi frequencies at each step
omega = ScalarOperator(lambda t: omega_max
                       if time_ramp < t.real < time_max else 0.0)
# Global phase at each step
phi = ScalarOperator.const(0.0)
# Global detuning at each step
delta = ScalarOperator(lambda t: delta_end
                       if time_ramp < t.real < time_max else delta_start)

async_result = cudaq.evolve_async(RydbergHamiltonian(atom_sites=register,
                                                     amplitude=omega,
                                                     phase=phi,
                                                     delta_global=delta),
                                  schedule=schedule,
                                  shots_count=10).get()
async_result.dump()

## Sample result
# ```
# {
#   __global__ : { 12121222:1 21202221:1 ... }
#    post_sequence : { 01010111:1 10101010:2 ... }
#    pre_sequence : { 11101111:1 11111111:9 }
# }
# ```

## Interpreting result
# `pre_sequence` has the measurement bits, one for each atomic site, before the
# quantum evolution is run. The count is aggregated across shots. The value is
# 0 if site is empty, 1 if site is filled.
# `post_sequence` has the measurement bits, one for each atomic site, at the
# end of the quantum evolution. The count is aggregated across shots. The value
# is 0 if atom is in Rydberg state or site is empty, 1 if atom is in ground
# state.
# `__global__` has the aggregate of the state counts from all the successful
# shots. The value is 0 if site is empty, 1 if atom is in Rydberg state (up
# state spin) and 2 if atom is in ground state (down state spin).
```

#### Scaleway

```python
import cudaq

# NOTE: Scaleway credentials must be set before running this program.
# Scaleway costs apply.
cudaq.set_target("scaleway", max_duration="10m")

# The default device is EMU-CUDAQ-H100, a state vector simulator running on an H100 GPU. Users may choose any of
# the available devices by supplying its name with the `machine` parameter.
# For example,
# ```
# cudaq.set_target("scaleway", machine="EMU-CUDAQ-H100")
# ```
# To ensure we keep the same QPU session between runs, we can also specify the `deduplication_id` parameter.
# For example,
# ```
# cudaq.set_target("scaleway", machine="EMU-CUDAQ-H100", deduplication_id="my_unique_id_1234")
# ```
# Users may also specify QPU session duration limits with the `max_duration` and `max_idle_duration` parameters.
# For example,
# ```
# cudaq.set_target("scaleway", machine="EMU-CUDAQ-H100", max_duration="30m", max_idle_duration="5m")
# ```

# Create the kernel we'd like to execute
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])

# Execute and print out the results.

# Option A:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by Scaleway.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

async_counts = async_results.get()
print(async_counts)

# Option B:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from Scaleway QaaS.
counts = cudaq.sample(kernel)
print(counts)
```

#### TII

```python
import cudaq
import os

# Set the target at the beginning of the program.
cudaq.set_target("tii",
                 device="tii-sim",
                 project=os.environ.get("TII_PROJECT", None))

# Create the kernel.
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])
    mz(qvector)

# Note: Increase shots count for better distribution of results.
# Using small number here for testing purposes.
SHOTS = 50

# Execute on synchronously on the TII cloud and print out the results.
counts = cudaq.sample(kernel, shots_count=SHOTS)
print(counts)
```

### 16. When to Use sample vs. run

> Source: https://nvidia.github.io/cuda-quantum/latest/using/examples/sample_vs_run.html

#### Introduction

Starting with CUDA-Q 0.14.0, `sample` no longer supports kernels that branch
on measurement results (measurement-dependent control flow). Kernels containing
patterns such as `if mz(q):` or `if (result) { ... }` where `result`
comes from a measurement must now use `run` instead.

This breaking change creates a clearer API separation:

- Use `sample` for **aggregate measurement statistics** (counts dictionaries).
- Use `run` for **shot-by-shot execution** with measurement-dependent control
  flow and individual return values.

#### Usage Guidelines

**Use** `sample` **when:**

- You want aggregate measurement statistics (histograms).
- Your kernel has no measurement-dependent control flow.
- You only need final measurement distributions.
- You are using the `explicit_measurements` option, which concatenates all
  measurement results in execution order rather than re-measuring qubits at the
  end of the kernel. See the sample specification
  for details.

**Use** `run` **when:**

- You need shot-by-shot measurement values.
- Your kernel has conditionals based on measurement results.
- You want to return computed values from the kernel.
- You need to store or analyze individual shot data.

For the full API specification, see the sample and
run sections in the Algorithmic Primitives documentation.
For a usage guide, see [Running your first CUDA-Q Program](https://nvidia.github.io/cuda-quantum/latest/using/basics/run_kernel.html).

#### What Is Supported with `sample`

Kernels without measurement-dependent control flow continue to work exactly as
before. This includes implicit measurements, explicit measurements without
conditionals, partial qubit measurement, mid-circuit measurement for
reset patterns, and the `explicit_measurements` option (described above).

```python
@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])

@cudaq.kernel
def reset_pattern():
    q = cudaq.qubit()
    h(q)
    mz(q)
    reset(q)
    x(q)

print("Implicit measurements:")
cudaq.sample(bell).dump()

print("\nMid-circuit measurement with reset:")
cudaq.sample(reset_pattern).dump()

print("\nWith explicit_measurements option:")
cudaq.sample(reset_pattern, explicit_measurements=True).dump()
```

#### What Is Not Supported with `sample`

Kernels that branch on measurement results can no longer be used with
`sample` or `sample_async`. Attempting to do so will raise a runtime error.

This includes both inline conditionals on measurements and conditionals on
variables holding measurement results:

```python
@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    h(q[0])
    r = mz(q[0])
    if r:               # ERROR
        x(q[1])

cudaq.sample(kernel)    # raises RuntimeError
```

The error message will read:

```text
`cudaq::sample` and `cudaq::sample_async` no longer support kernels that
branch on measurement results. Kernel '<name>' uses conditional feedback.
Use `cudaq::run` or `cudaq::run_async` instead. See CUDA-Q documentation
for migration guide.
```

#### How to Migrate

Migrating a kernel from `sample` to `run` requires three changes.

##### Step 1: Add a return type to the kernel

`run` requires kernels to return a non-void value. Instead of relying on
implicit measurement at the end of the circuit, explicitly `return` the
measurement results you need.

```python
# Before (no return type, used with sample)
@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    h(q[0])
    r = mz(q[0])
    if r:
        x(q[1])

# After (returns a value, used with run)
@cudaq.kernel
def kernel() -> bool:
    q = cudaq.qvector(2)
    h(q[0])
    r = mz(q[0])
    if r:
        x(q[1])
    return mz(q[1])
```

##### Step 2: Replace `sample` with `run`

```python
# Before
counts = cudaq.sample(kernel, shots_count=1000)

# After
results = cudaq.run(kernel, shots_count=1000)
```

> **Note:** The default `shots_count` for `run` is 100, compared to 1000 for `sample`. Specify `shots_count` explicitly if you need a particular number of shots.

##### Step 3: Update result processing

`sample` returns a `cudaq.SampleResult` (a counts dictionary mapping bit strings
to frequencies). `run` returns a list (Python) or `std::vector` (C++) of
individual return values -- one per shot. If you need a counts-dictionary view,
you can reconstruct it from the individual results:

```python
from collections import Counter

results = cudaq.run(multi_measure, shots_count=1000)
counts = Counter(
    ''.join('1' if bit else '0' for bit in result) for result in results)
print(dict(counts))
```

#### Migration Examples

##### Example 1: Simple conditional logic

A kernel that measures one qubit and conditionally applies a gate on another.

```python
@cudaq.kernel
def simple_conditional() -> bool:
    q = cudaq.qvector(2)
    h(q[0])
    r = mz(q[0])
    if r:
        x(q[1])
    return mz(q[1])

results = cudaq.run(simple_conditional, shots_count=100)
n_ones = sum(results)
print(f"Measured |1> {n_ones} out of {len(results)} shots")
```

##### Example 2: Returning multiple measurement results

A kernel that performs multiple mid-circuit measurements with conditional logic
and returns all results as a list.

```python
@cudaq.kernel
def multi_measure() -> list[bool]:
    q = cudaq.qvector(3)
    h(q)
    r0 = mz(q[0])
    r1 = mz(q[1])
    if r0 and r1:
        x(q[2])
    r2 = mz(q[2])
    return [r0, r1, r2]

results = cudaq.run(multi_measure, shots_count=100)
for shot in results[:5]:
    print(''.join('1' if b else '0' for b in shot))
```

##### Example 3: Quantum teleportation

Teleportation of a qubit state requires conditional corrections based on
Bell-basis measurements.

```python
@cudaq.kernel
def teleport() -> list[bool]:
    results = [False, False, False]
    q = cudaq.qvector(3)
    x(q[0])

    h(q[1])
    x.ctrl(q[1], q[2])

    x.ctrl(q[0], q[1])
    h(q[0])

    results[0] = mz(q[0])
    results[1] = mz(q[1])

    if results[1]:
        x(q[2])
    if results[0]:
        z(q[2])

    results[2] = mz(q[2])
    return results

runs = cudaq.run(teleport, shots_count=100)
assert all(r[2] for r in runs), "Teleportation failed"
print(f"Teleportation succeeded on all {len(runs)} shots")
```

#### Additional Notes

- Users of `sample_async` with conditional-feedback kernels should migrate to
  `run_async`. See the run specification for the
  asynchronous API.

- `run` supports a variety of return types including scalars, vectors/lists,
  and user-defined data structures. See the
  run specification for the complete list of supported
  types and their requirements.

- Assigning measurement results to named variables in kernels passed to
  `sample` is deprecated and will be removed in a future release. Use `run`
  to retrieve individual measurement results.

---

## Part 2: Dynamics Simulation

> Source: https://nvidia.github.io/cuda-quantum/latest/using/dynamics.html

CUDA-Q simulates quantum dynamics — the time evolution of quantum systems or models — via the `evolve` API. In simulation mode, CUDA-Q provides the `dynamics` backend target, based on the cuQuantum library and optimized for performance and scale on NVIDIA GPUs.

### Quick Start

A simple time-evolution workflow comprises the following steps:

#### 1. Define a quantum system model

A quantum system model is defined by a Hamiltonian. For example, a superconducting [transmon](https://en.wikipedia.org/wiki/Transmon) qubit can be modeled by the following Hamiltonian

$$H = \frac{\omega_z}{2} \sigma_z + \omega_x \cos(\omega_d t)\,\sigma_x,$$

where $\sigma_z$ and $\sigma_x$ are Pauli Z and X operators, respectively.

Using CUDA-Q `operator`, the above time-dependent Hamiltonian can be set up as follows.

```python
omega_z = 6.5
omega_x = 4.0
omega_d = 0.5

import numpy as np
from cudaq import spin, ScalarOperator

# Qubit Hamiltonian
hamiltonian = 0.5 * omega_z * spin.z(0)
# Add modulated driving term to the Hamiltonian
hamiltonian += omega_x * ScalarOperator(lambda t: np.cos(omega_d * t)) * spin.x(0)
```

In particular, `ScalarOperator` provides an easy way to model arbitrary time-dependent control signals. Details about CUDA-Q `operator`, including builtin operators that it supports, can be found in the [Operator](#operator) section below.

#### 2. Set up the evolution simulation

The below code snippet shows how to simulate the time-evolution of the above system with `cudaq.evolve`.

```python
import cudaq
import cupy as cp
from cudaq.dynamics import Schedule

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Dimensions of sub-systems: a single two-level system.
dimensions = {0: 2}

# Initial state of the system (ground state).
rho0 = cudaq.State.from_data(
    cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

# Schedule of time steps.
t_final = 1.0
n_steps = 100
steps = np.linspace(0, t_final, n_steps)
schedule = Schedule(steps, ["t"])

# Run the simulation.
evolution_result = cudaq.evolve(
    hamiltonian,
    dimensions,
    schedule,
    rho0,
    observables=[spin.x(0), spin.y(0), spin.z(0)],
    collapse_operators=[],
    store_intermediate_results=cudaq.IntermediateResultSave.ALL)
```

The simulation requires:

*   The system model: a Hamiltonian plus any decoherence terms (`collapse_operators`).
*   The dimensionality of each component system (`evolve` supports arbitrary multi-level systems, e.g. photonic Fock space).
*   The initial quantum state.
*   The time schedule (time steps) of the evolution.
*   Any 'observable' operators whose expectation values should be measured on the evolving state.

> **Note:** By default, `evolve` will only return the final state and expectation values. To save intermediate results (at each time step specified in the schedule), the `store_intermediate_results` flag must be set.

#### 3. Retrieve and plot the results

After the simulation, the final state and expectation values are available — and, with `store_intermediate_results=cudaq.IntermediateResultSave.ALL`, the intermediate values at each time step.

> **Note:** Storing intermediate states can be memory-intensive, especially for large systems. If you only need the intermediate expectation values, you can set `store_intermediate_results` to `cudaq.IntermediateResultSave.EXPECTATION_VALUE` instead.

For example, we can plot the Pauli expectation value for the above simulation as follows.

```python
get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]

import matplotlib.pyplot as plt

plt.plot(steps, get_result(0, evolution_result))
plt.plot(steps, get_result(1, evolution_result))
plt.plot(steps, get_result(2, evolution_result))
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
```

At each time step, `evolve` records one expectation value per observable; these are converted into per-observable sequences for plotting.

Examples that illustrate how to use the `dynamics` target are available in the [CUDA-Q repository](https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/python/dynamics).

### Operator

CUDA-Q provides builtin definitions for commonly-used operators, such as the ladder operators ($a$ and $a^\dagger$) of a harmonic oscillator, the Pauli spin operators for a two-level system, etc.

Here is a list of those operators.

| Name | Description |
| --- | --- |
| `identity` | Identity operator |
| `zero` | Zero or null operator |
| `annihilate` | Bosonic annihilation operator ($a$) |
| `create` | Bosonic creation operator ($a^\dagger$) |
| `number` | Number operator of a bosonic mode (equivalent to $a^\dagger a$) |
| `parity` | Parity operator of a bosonic mode (defined as $e^{i\pi a^\dagger a}$) |
| `displace` | Displacement operator of complex amplitude $\alpha$ (`displacement`). It is defined as $e^{\alpha a^\dagger - \alpha^* a}$. |
| `squeeze` | Squeezing operator of complex squeezing amplitude $z$ (`squeezing`). It is defined as $\exp(\frac{1}{2}(z^* a^2 - z a^{\dagger 2}))$. |
| `position` | Position operator (equivalent to $(a^\dagger + a)/2$) |
| `momentum` | Momentum operator (equivalent to $i(a^\dagger - a)/2$) |
| `spin.x` | Pauli $\sigma_x$ operator |
| `spin.y` | Pauli $\sigma_y$ operator |
| `spin.z` | Pauli $\sigma_z$ operator |
| `spin.plus` | Pauli raising ($\sigma_+$) operator |
| `spin.minus` | Pauli lowering ($\sigma_-$) operator |

For example, the Jaynes-Cummings model — the interaction between a two-level atom and a light (boson) field — has the Hamiltonian

$$H = \omega_c a^\dagger a + \omega_a \frac{\sigma_z}{2} + \frac{\Omega}{2}(a\sigma_+ + a^\dagger \sigma_-).$$

This Hamiltonian can be converted to CUDA-Q `Operator` representation with

```python
from cudaq import operators

hamiltonian = omega_c * operators.create(1) * operators.annihilate(1) \
                + (omega_a / 2) * spin.z(0) \
                + (Omega / 2) * (operators.annihilate(1) * spin.plus(0) + operators.create(1) * spin.minus(0))
```

In the above code snippet, we map the cavity light field to degree index 1 and the two-level atom to degree index 0. The description of composite quantum system dynamics is independent from the Hilbert space of the system components. The latter is specified by the dimension map that is provided to the `cudaq.evolve` call.

Builtin operators support both dense and multi-diagonal sparse formats. Depending on the sparsity of operator matrix and/or the sub-system dimension, CUDA-Q will either use the dense or multi-diagonal data formats for optimal performance.

Specifically, the following environment variable options are applicable to the `dynamics` target. Any environment variables must be set prior to setting the target or running `import cudaq`.

**Additional environment variable options for the `dynamics` target**

| Option | Value | Description |
| --- | --- | --- |
| `CUDAQ_DYNAMICS_MIN_MULTIDIAGONAL_DIMENSION` | Non-negative number | The minimum sub-system dimension on which the operator acts to activate multi-diagonal data format. For example, if a minimum dimension configuration of `N` is set, all operators acting on degrees of freedom (sub-system) whose dimension is less than or equal to `N` would always use the dense format. The final data format to be used depends on the next configuration. The default is 4. |
| `CUDAQ_DYNAMICS_MAX_DIAGONAL_COUNT_FOR_MULTIDIAGONAL` | Non-negative number | The maximum number of diagonals for multi-diagonal representation. If the operator matrix has more diagonals than this value, the dense format will be used. Default is 1, i.e., operators with only one diagonal line (center, lower, or upper) will use the multi-diagonal sparse storage. |

### Time-Dependent Dynamics

The examples above used time-independent Hamiltonians. CUDA-Q provides multiple ways to construct operators with explicit time dependence.

#### 1. Time-dependent coefficient

CUDA-Q `ScalarOperator` can be used to wrap a Python function that returns the coefficient value at a specific time.

As an example, we will look at a time-dependent Hamiltonian of the form $H = H_0 + f(t)H_1$, where $f(t)$ is the time-dependent driving strength given as $\cos(\omega t)$.

The following code sets up the problem

```python
# Define the static (drift) and control terms
H0 = spin.z(0)
H1 = spin.x(0)
H = H0 + ScalarOperator(lambda t: np.cos(omega * t)) * H1
```

#### 2. Time-dependent operator

We can also construct a time-dependent operator from a function that returns a complex matrix representing the time dynamics of that operator.

As an example, let's look at the [displacement operator](https://en.wikipedia.org/wiki/Displacement_operator). It can be defined as follows:

```python
import numpy
import scipy
from cudaq import operators, NumericType
from numpy.typing import NDArray

def displacement_matrix(
        dimension: int,
        displacement: NumericType) -> NDArray[numpy.complexfloating]:
    """
    Returns the displacement operator matrix.
    Args:
        displacement: Amplitude of the displacement operator.
            See also https://en.wikipedia.org/wiki/Displacement_operator.
    """
    displacement = complex(displacement)
    term1 = displacement * operators.create(0).to_matrix({0: dimension})
    term2 = numpy.conjugate(displacement) * operators.annihilate(0).to_matrix(
        {0: dimension})
    return scipy.linalg.expm(term1 - term2)

# The second argument here indicates the defined operator
# acts on a single degree of freedom, which can have any dimension.
# An argument [2], for example, would indicate that it can only
# act on a single degree of freedom with dimension two.
operators.define("displace", [0], displacement_matrix)

def displacement(degree: int) -> operators.MatrixOperatorElement:
    """
    Instantiates a displacement operator acting on the given degree of freedom.
    """
    return operators.instantiate("displace", [degree])
```

The operator is parameterized by the `displacement` amplitude; to evolve under a time-dependent amplitude, define how it changes in time:

```python
import cudaq

# Define a system consisting of a single degree of freedom (0) with dimension 3.
system_dimensions = {0: 3}
system_operator = displacement(0)

# Define the time dependency of the system operator as a schedule that linearly
# increases the displacement parameter from 0 to 1.
time_dependence = Schedule(numpy.linspace(0, 1, 100), ['displacement'])
initial_state = cudaq.State.from_data(
    numpy.ones(3, dtype=numpy.complex128) / numpy.sqrt(3))

# Simulate the evolution of the system under this time dependent operator.
cudaq.evolve(system_operator, system_dimensions, time_dependence, initial_state)
```

To add a squeezing term and vary the squeezing and displacement amplitudes independently, instantiate a schedule with a custom function returning each parameter's value:

```python
system_operator = displacement(0) + operators.squeeze(0)

# Define a schedule such that displacement amplitude increases linearly in time
# but the squeezing amplitude decreases, that is follows the inverse schedule.
def parameter_values(time_steps):

    def compute_value(param_name, step_idx):
        match param_name:
            case 'displacement':
                return time_steps[int(step_idx)]
            case 'squeezing':
                return time_steps[-int(step_idx + 1)]
            case _:
                raise ValueError(f"value for parameter {param_name} undefined")

    return Schedule(range(len(time_steps)), system_operator.parameters.keys(),
                    compute_value)

time_dependence = parameter_values(numpy.linspace(0, 1, 100))
cudaq.evolve(system_operator, system_dimensions, time_dependence, initial_state)
```

### Super-operator Representation

The examples above assume Lindblad master-equation dynamics (Hamiltonian + collapse operators). To simulate an arbitrary state-evolution equation instead, the right-hand side can be provided as a generic super-operator: the `SuperOperator` class represents it as a linear combination (sum) of left and/or right multiplication actions of `Operator` instances.

As an example, we will look at specifying the Schrodinger's equation

$$\frac{d|\Psi\rangle}{dt} = -i H |\Psi\rangle$$

as a super-operator.

```python
import cudaq
from cudaq import spin, Schedule, RungeKuttaIntegrator
import numpy as np

hamiltonian = 2.0 * np.pi * 0.1 * spin.x(0)
steps = np.linspace(0, 1, 10)
schedule = Schedule(steps, ["t"])
dimensions = {0: 2}
# initial state
psi0 = cudaq.dynamics.InitialState.ZERO
# Super-operator applying `-iH|psi>` (the Schrodinger equation right-hand side)
se_super_op = cudaq.SuperOperator()
se_super_op += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
evolution_result = cudaq.evolve(se_super_op,
                                dimensions,
                                schedule,
                                psi0,
                                observables=[spin.z(0)],
                                store_intermediate_results=cudaq.IntermediateResultSave.ALL,
                                integrator=RungeKuttaIntegrator())
```

The super-operator, once constructed, can be used in the `evolve` API instead of the Hamiltonian and collapse operators as shown in the above examples.

### Numerical Integrators

For Python, CUDA-Q provides a set of numerical integrators, to be used with the `dynamics` backend target.

| Name | Description |
| --- | --- |
| `RungeKuttaIntegrator` | Explicit 4th-order Runge-Kutta method (default integrator) |
| `ScipyZvodeIntegrator` | Complex-valued variable-coefficient ordinary differential equation solver (provided by SciPy) |
| `CUDATorchDiffEqDopri5Integrator` | Runge-Kutta of order 5 of Dormand-Prince-Shampine (provided by `torchdiffeq`) |
| `CUDATorchDiffEqAdaptiveHeunIntegrator` | Runge-Kutta of order 2 (provided by `torchdiffeq`) |
| `CUDATorchDiffEqBosh3Integrator` | Runge-Kutta of order 3 of Bogacki-Shampine (provided by `torchdiffeq`) |
| `CUDATorchDiffEqDopri8Integrator` | Runge-Kutta of order 8 of Dormand-Prince-Shampine (provided by `torchdiffeq`) |
| `CUDATorchDiffEqEulerIntegrator` | Euler method (provided by `torchdiffeq`) |
| `CUDATorchDiffEqExplicitAdamsIntegrator` | Explicit Adams-Bashforth method (provided by `torchdiffeq`) |
| `CUDATorchDiffEqImplicitAdamsIntegrator` | Implicit Adams-Bashforth-Moulton method (provided by `torchdiffeq`) |
| `CUDATorchDiffEqMidpointIntegrator` | Midpoint method (provided by `torchdiffeq`) |
| `CUDATorchDiffEqRK4Integrator` | Fourth-order Runge-Kutta with 3/8 rule (provided by `torchdiffeq`) |

> **Note:** To use Torch-based integrators, users need to install `torchdiffeq` (e.g., with `pip install torchdiffeq`). This is an optional dependency of CUDA-Q, thus will not be installed by default.

> **Warning:** Torch-based integrators require a CUDA-enabled Torch installation. Depending on your platform (e.g., `aarch64`), the default Torch pip package may not have CUDA support.
>
> The below command can be used to verify your installation:
>
> ```bash
> python3 -c "import torch; print(torch.version.cuda)"
> ```
>
> An output of '`None`' means the Torch installation lacks CUDA support; install a CUDA-enabled Torch build instead (e.g., from source or their Docker images).

### Batch Simulation

CUDA-Q `dynamics` target supports batch simulation, which allows users to run multiple simulations simultaneously. This batching capability applies to (1) multiple initial states and/or (2) multiple Hamiltonians.

Batching can significantly improve performance when simulating many small identical system dynamics, e.g., parameter sweeping or tomography.

#### Batching initial states

Simulating multiple initial states with the same Hamiltonian:

```python
import cudaq
import cupy as cp
import numpy as np
from cudaq import spin, Schedule, RungeKuttaIntegrator
# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Qubit Hamiltonian
hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Initial states in the `SIC-POVM` set: https://en.wikipedia.org/wiki/SIC-POVM
psi_1 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))
psi_2 = cudaq.State.from_data(
    cp.array([1.0 / np.sqrt(3.0), np.sqrt(2.0 / 3.0)], dtype=cp.complex128))
psi_3 = cudaq.State.from_data(
    cp.array([
        1.0 / np.sqrt(3.0),
        np.sqrt(2.0 / 3.0) * np.exp(1j * 2.0 * np.pi / 3.0)
    ],
             dtype=cp.complex128))
psi_4 = cudaq.State.from_data(
    cp.array([
        1.0 / np.sqrt(3.0),
        np.sqrt(2.0 / 3.0) * np.exp(1j * 4.0 * np.pi / 3.0)
    ],
             dtype=cp.complex128))

# We run the evolution for all the SIC state to determine the process tomography.
sic_states = [psi_1, psi_2, psi_3, psi_4]
# Schedule of time steps.
steps = np.linspace(0, 10, 101)
schedule = Schedule(steps, ["time"])

# Run the batch simulation.
evolution_results = cudaq.evolve(
    hamiltonian,
    dimensions,
    schedule,
    sic_states,
    observables=[spin.x(0), spin.y(0), spin.z(0)],
    collapse_operators=[],
    store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
    integrator=RungeKuttaIntegrator())
```

#### Batching Hamiltonians

Batch-simulating multiple Hamiltonians:

```python
import cudaq
import cupy as cp
import numpy as np
from cudaq import spin, Schedule, ScalarOperator, RungeKuttaIntegrator
# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Dimensions of sub-system.
dimensions = {0: 2}

# Qubit resonant frequency
omega_z = 10.0 * 2 * np.pi

# Transverse term
omega_x = 2 * np.pi

# Harmonic driving frequency (sweeping in the +/- 10% range around the resonant frequency).
omega_drive = np.linspace(0.9 * omega_z, 1.1 * omega_z, 16)

# Initial state of the system (ground state).
psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

# Batch the Hamiltonian operator together
hamiltonians = [
    0.5 * omega_z * spin.z(0) + omega_x *
    ScalarOperator(lambda t, omega=omega: np.cos(omega * t)) * spin.x(0)
    for omega in omega_drive
]

# Initial states for each Hamiltonian in the batch.
# Here, we use the ground state for all Hamiltonians.
initial_states = [psi0] * len(hamiltonians)

# Schedule of time steps.
steps = np.linspace(0, 0.5, 5000)
schedule = Schedule(steps, ["t"])

# Run the batch simulation.
evolution_results = cudaq.evolve(
    hamiltonians,
    dimensions,
    schedule,
    initial_states,
    observables=[spin.x(0), spin.y(0), spin.z(0)],
    collapse_operators=[],
    store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
    integrator=RungeKuttaIntegrator())
```

Here each Hamiltonian in the batch corresponds to one initial state (the two lists have equal length). If only one initial state is provided, it is used for all Hamiltonians in the batch.

#### Retrieving batched results

The batch simulation returns a list of evolve result objects, one per Hamiltonian in the batch. Extract each Hamiltonian's expectation-value time series as follows:

```python
# Split the batched results into separate arrays for each observable.
all_exp_val_x = []
all_exp_val_y = []
all_exp_val_z = []
# Iterate over the evolution results in the batch:
for evolution_result in evolution_results:
    exp_val_x = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]
    exp_val_y = [
        exp_vals[1].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]
    exp_val_z = [
        exp_vals[2].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]

    all_exp_val_x.append(exp_val_x)
    all_exp_val_y.append(exp_val_y)
    all_exp_val_z.append(exp_val_z)
```

Each `all_exp_val_*` entry is then a nested list: one inner list of per-time-step expectation values per Hamiltonian in the batch.

Collapse operators and super-operators can also be batched in a similar manner. Specifically, if the `collapse_operators` parameter is a nested list of operators, then each set of collapse operators in the list will be applied to the corresponding Hamiltonian in the batch.

#### Batching requirements

Batching requires all Hamiltonians to share the same structure: the same number of product terms acting on the same degrees of freedom. Term order, coefficient values/callbacks, and the specific operators on those terms do not matter. Examples:

| First Hamiltonian | Second Hamiltonian | Batchable? |
| --- | --- | --- |
| $H_1 = \omega_1 \sigma_z(0)$ | $H_2 = \omega_2 \sigma_z(0)$ | Yes (different coefficients, same operator) |
| $H_1 = \omega_z \sigma_z(0) + \cos(\omega_x t) \sigma_x(1)$ | $H_2 = \omega_z \sigma_z(0) + \sin(\omega_x t) \sigma_x(1)$ | Yes (same structure, different callback coefficients) |
| $H_1 = \omega_z \sigma_z(0) + \cos(\omega_x t) \sigma_x(1)$ | $H_2 = \omega_z \sigma_z(0) + \cos(\omega_x t) \sigma_y(1)$ | Yes (different operators on the same degree of freedom) |
| $H_1 = \omega_z \sigma_z(0) + \cos(\omega_x t) \sigma_x(1)$ | $H_2 = \omega_z \sigma_z(0) + \cos(\omega_x t) \sigma_x(1) + \cos(\omega_y t) \sigma_y(1)$ | No (different number of product terms) |
| $H_1 = \omega_z \sigma_z(0) + \cos(\omega_x t) \sigma_{xx}(0, 1)$ | $H_2 = \omega_z \sigma_z(0) + \cos(\omega_x t) \sigma_x(0)\sigma_x(1)$ | No (different structures, two-body operators vs. tensor product of single-body operators) |

When the Hamiltonians are **not** batchable, CUDA-Q will still run the simulations, but each Hamiltonian will be simulated separately in a sequential manner. CUDA-Q will log a warning "The input Hamiltonian and collapse operators are not compatible for batching. Running the simulation in non-batched mode." when that happens.

> **Note:** Depending on the number of Hamiltonian operators together with factors such as the integrator, schedule step size, and whether intermediate results are stored, the batch simulation can be memory-intensive. If you encounter out-of-memory issues, the `max_batch_size` parameter can be used to limit the number of Hamiltonians that are batched together in one run. For example, if you set `max_batch_size=2`, then we will run the simulations in batches of 2 Hamiltonians at a time, i.e., the first two Hamiltonians will be simulated together, then the next two, and so on.

```python
# Run the batch simulation in batches of at most 2 Hamiltonians at a time.
results = cudaq.evolve(
    hamiltonians,
    dimensions,
    schedule,
    initial_states,
    observables=[spin.x(0), spin.y(0), spin.z(0)],
    collapse_operators=[],
    store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
    integrator=RungeKuttaIntegrator(),
    max_batch_size=2)  # Set the maximum batch size to 2
```

### Multi-GPU Multi-Node Execution

CUDA-Q `dynamics` target supports parallel execution on multiple GPUs. To enable parallel execution, the application must initialize MPI as follows.

```python
cudaq.mpi.initialize()

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Initial state (expressed as an enum)
psi0 = cudaq.dynamics.InitialState.ZERO

# Run the simulation
evolution_result = cudaq.evolve(
    H,
    dimensions,
    schedule,
    psi0,
    observables=[],
    collapse_operators=[],
    store_intermediate_results=cudaq.IntermediateResultSave.NONE,
    integrator=RungeKuttaIntegrator())

cudaq.mpi.finalize()
```

```bash
mpiexec -np <N> python3 program.py
```

where `N` is the number of processes.

Initializing MPI in the application (`cudaq.mpi.initialize()`) and launching via an MPI launcher activates the multi-node multi-GPU feature: the target detects the number of processes (GPUs) and distributes the computation across them.

> **Note:** The number of MPI processes must be a power of 2, one GPU per process.

> **Note:** Not all integrators are capable of handling distributed state. Errors will be raised if parallel execution is activated but the selected integrator does not support distributed state.

> **Note:** When running batched simulations in a multi-GPU multi-node environment, the batch size will be automatically divided by the number of MPI processes. Hence, the batch size needs to be divisible by the number of processes. For example, if the original batch size is 8 and there are 4 MPI processes, then each process (GPU) will simulate a batch size of 2. Errors will be raised if the batch size is not divisible by the number of processes.

Each process will return its own set of results. The user is responsible for gathering the results from all processes if needed.

### Examples

The [Dynamics Examples](https://nvidia.github.io/cuda-quantum/latest/using/examples/dynamics_examples.html) section of the docs contains a number of excellent dynamics examples demonstrating how to simulate basic physics models, specific qubit modalities, and utilize multi-GPU multi-Node capabilities.

---

## Part 3: GPU Simulator Backends & Performance Options

> Sources:
> - https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html (State Vector Simulators)
> - https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/tnsims.html (Tensor Network Simulators)

> **Note (applies to all GPU-accelerated backends below):** an NVIDIA GPU and CUDA runtime libraries are required; missing CUDA dependencies surface as an `Invalid simulator requested` error. MPI-based configurations (`mgpu`, multi-GPU `tensornet`) additionally require an MPI installation — missing MPI surfaces as a `failed to launch kernel` error. See [Dependencies and Compatibility](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility) for how to install these dependencies. A target set in the application code (`cudaq.set_target(...)`) overrides the `--target` command-line flag given at program invocation. Environment variables must be set before setting the target or running `import cudaq`.

### State Vector Simulators

#### CPU (`qpp-cpu`)

The `qpp-cpu` backend provides a state vector simulator based on the CPU-only, OpenMP threaded [Q++](https://github.com/softwareqinc/qpp) library. This backend is good for basic testing and experimentation with just a few qubits, but performs poorly for all but the smallest simulation and is the default target when running on CPU-only systems.

To execute a program on the `qpp-cpu` target even if a GPU-accelerated backend is available, use the following command:

```bash
python3 program.py [...] --target qpp-cpu
```

The target can also be defined in the application code by calling

```python
cudaq.set_target('qpp-cpu')
```

#### Single-GPU (`nvidia`)

The `nvidia` backend provides a state vector simulator accelerated with the `cuStateVec` library. The [cuStateVec documentation](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html) provides a detailed explanation for how the simulations are performed on the GPU.

The `nvidia` target supports multiple configurable options including specification of floating point precision.

To execute a program on the `nvidia` backend, use the following commands:

```bash
# Single Precision (Default):
python3 program.py [...] --target nvidia --target-option fp32

# Double Precision:
python3 program.py [...] --target nvidia --target-option fp64
```

The target can also be defined in the application code by calling

```python
cudaq.set_target('nvidia', option='fp64')
```

In the single-GPU mode, the `nvidia` backend provides the following environment variable options. It is worth drawing attention to gate fusion, a powerful tool for improving simulation performance, which is discussed in greater detail in [Performance Optimizations](https://nvidia.github.io/cuda-quantum/latest/examples/python/performance_optimizations.html).

**Environment variable options supported in single-GPU mode**

| Option | Value | Description |
| --- | --- | --- |
| `CUDAQ_FUSION_MAX_QUBITS` | positive integer | The max number of qubits used for gate fusion. The default value depends on [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) (CC) and the floating point precision selected for the simulator, as specified in the Default Gate Fusion Size table below. |
| `CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS` | integer greater than or equal to -1 | The max number of qubits used for diagonal gate fusion. The default value is set to `-1` and the fusion size will be automatically adjusted for the better performance. If 0, the gate fusion for diagonal gates is disabled. |
| `CUDAQ_FUSION_NUM_HOST_THREADS` | positive integer | Number of CPU threads used for circuit processing. The default value is `8`. |
| `CUDAQ_MAX_CPU_MEMORY_GB` | non-negative integer, or `NONE` | CPU memory size (in GB) allowed for state-vector migration. `NONE` means unlimited (up to physical memory constraints). Default is 0GB (disabled, variable is not set to any value). |
| `CUDAQ_MAX_GPU_MEMORY_GB` | positive integer, or `NONE` | GPU memory (in GB) allowed for on-device state-vector allocation. As the state-vector size exceeds this limit, host memory will be utilized for migration. `NONE` means unlimited (up to physical memory constraints). This is the default. |
| `CUDAQ_ALLOW_FP32_EMULATED` | `TRUE` (`1`, `ON`) or `FALSE` (`0`, `OFF`) | [Blackwell (compute capability 10.0+) only] Enable or disable floating point math emulation. If enabled, allows `FP32` emulation kernels using `BFloat16` (`BF16`) whenever possible. Enabled by default. |
| `CUDAQ_ENABLE_MEMPOOL` | `TRUE` (`1`, `ON`) or `FALSE` (`0`, `OFF`) | Enable or disable [CUDA memory pool](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/#memory_pools) for state vector allocation/deallocation. Enabled by default. |

> **Deprecated since version 0.8:** The `nvidia-fp64` target, which is equivalent to setting the `fp64` option on the `nvidia` target, is deprecated and will be removed in a future release.

> **Note:** In host-device simulation (i.e., `CUDAQ_MAX_CPU_MEMORY_GB` is not 0), the backend automatically switches between inner product (default) and operator matrix-based methods for expectation calculations (`cudaq::observe`) depending on whether a clone of the state can be allocated or not.
>
> For example, when `CUDAQ_MAX_GPU_MEMORY_GB` is unconstrained, the quantum state vector would consume all device memory before utilizing host memory. Thus, the backend would fall back to the operator matrix-based approach as cloning the state is not possible. For performance reasons, only Pauli operator matrices of up to 8 qubits (identity padding not included) are allowed in this mode. This constraint can be relaxed by setting the `CUDAQ_MATRIX_EXP_VAL_MAX_SIZE` environment variable. Users would need to take into account the full operator matrix size when increasing this setting.

#### Multi-GPU multi-node (`nvidia`, option `mgpu`)

The `nvidia` backend also provides a state vector simulator accelerated with the `cuStateVec` library with support for Multi-GPU, Multi-node distribution of the state vector.

This mode is required when the state vector cannot fit in a single GPU's memory. It runs within an MPI context (adjust `-np` to the available GPU resources):

See the [Divisive Clustering](https://nvidia.github.io/cuda-quantum/latest/applications/python/divisive_clustering_coresets.html) application to see how this backend can be used in practice.

```bash
# Double precision simulation:
mpiexec -np 2 python3 program.py [...] --target nvidia --target-option fp64,mgpu

# Single precision simulation:
mpiexec -np 2 python3 program.py [...] --target nvidia --target-option fp32,mgpu
```

> **Note:** If you installed CUDA-Q via `pip`, you will need to install the necessary MPI dependencies separately; please follow the instructions for installing dependencies in the [Project Description](https://pypi.org/project/cuda-quantum/#description).

In addition to using MPI in the simulator, you can use it in your application code by installing [mpi4py](https://mpi4py.readthedocs.io/), and invoking the program with the command

```bash
mpiexec -np 2 python3 -m mpi4py program.py [...] --target nvidia --target-option fp64,mgpu
```

The target can also be defined in the application code by calling

```python
cudaq.set_target('nvidia', option='mgpu,fp64')
```

> **Note:**
> - The order of the option settings are interchangeable. For example, `cudaq.set_target('nvidia', option='mgpu,fp64')` is equivalent to `cudaq.set_target('nvidia', option='fp64,mgpu')`.
> - The `nvidia` target has single-precision as the default setting. Thus, using `option='mgpu'` implies that `option='mgpu,fp32'`.

The number of processes and nodes should be always power-of-2.

Host-device state vector migration is also supported in the multi-GPU multi-node configuration.

In addition to those environment variable options supported in the single-GPU mode, the `nvidia` backend provides the following environment variable options particularly for the multi-node multi-GPU configuration.

**Additional environment variable options for multi-node multi-GPU mode**

| Option | Value | Description |
| --- | --- | --- |
| `CUDAQ_MGPU_LIB_MPI` | string | The shared library name for inter-process communication. The default value is `libmpi.so`. |
| `CUDAQ_MGPU_COMM_PLUGIN_TYPE` | `AUTO`, `EXTERNAL`, `OpenMPI`, or `MPICH` | Selecting `cuStateVec` `CommPlugin` for inter-process communication. The default is `AUTO`. If `EXTERNAL` is selected, `CUDAQ_MGPU_LIB_MPI` should point to an implementation of the `cuStateVec` `CommPlugin` interface. |
| `CUDAQ_MGPU_NQUBITS_THRESH` | positive integer | The qubit count threshold where state vector distribution is activated. Below this threshold, simulation is performed as independent (non-distributed) tasks across all MPI processes for optimal performance. Default is 25. |
| `CUDAQ_MGPU_FUSE` | positive integer | The max number of qubits used for gate fusion. The default value depends on [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) (CC) and the floating point precision selected for the simulator, as specified in the Default Gate Fusion Size table below. |
| `CUDAQ_MGPU_P2P_DEVICE_BITS` | positive integer | Specify the number of GPUs that can communicate by using GPUDirect P2P. Default value is 0 (P2P communication is disabled). |
| `CUDAQ_GPU_FABRIC` | `MNNVL`, `NVL`, `NONE`, or NVLink domain size (power of 2 integer) | Automatically set the number of P2P device bits based on the total number of processes when multi-node NVLink (`MNNVL`) is selected; or the number of processes per node when NVLink (`NVL`) is selected; or disable P2P (with `NONE`); or a specific NVLink domain size. |
| `CUDAQ_GLOBAL_INDEX_BITS` | comma-separated list of positive integers | Specify the network structure (faster to slower). E.g., for 32 MPI processes arranged as 4 groups of 8 with faster intra-group communication, set `3,2`: the `3` (= `log2(8)`) is the 8 fast-communicating processes per group, the `2` the 4 groups; the list sums to 5 (`2^5 = 32` processes). If unspecified, set based on P2P device bits. |
| `CUDAQ_HOST_DEVICE_MIGRATION_LEVEL` | positive integer | Position at which the migration index bits (CPU-GPU data transfers) are inserted into the `CUDAQ_GLOBAL_INDEX_BITS` list. If unset, they are appended at the end — a default optimized for systems with fast GPU-GPU interconnects (NVLink, InfiniBand, etc.) |
| `CUDAQ_DATA_TRANSFER_BUFFER_BITS` | positive integer greater than or equal to 24 | Specify the temporary buffer size (`1 << CUDAQ_DATA_TRANSFER_BUFFER_BITS` bytes) for inter-node data transfer. The default is set to 26 (64 MB). The minimum allowed value is 24 (16 MB). Depending on systems, setting a larger value to `CUDAQ_DATA_TRANSFER_BUFFER_BITS` can accelerate inter-node data transfers. |

> **Deprecated since version 0.8:** The `nvidia-mgpu` backend, which is equivalent to the multi-node multi-GPU double-precision option (`mgpu,fp64`) of the `nvidia` target, is deprecated and will be removed in a future release.

**Default Gate Fusion Size**

| Compute Capability | GPU | Default Gate Fusion Size |
| --- | --- | --- |
| 8.0 | NVIDIA A100 | 4 (`fp32`) or 5 (`fp64`) |
| 9.0 | NVIDIA H100, H200, GH200 | 5 (`fp32`) or 6 (`fp64`) |
| 10.0 | NVIDIA GB200, B200 | 5 (`fp32`) or 4 (`fp64`) |
| 10.3 | NVIDIA B300 | 5 (`fp32`) or 1 (`fp64`) |
| Others | | 4 (`fp32` and `fp64`) |

Gate fusion combines multiple gates at runtime — e.g., `x(qubit0)` and `x(qubit1)` become a single 4x4 matrix operation on the state vector instead of two 2x2 operations — reducing GPU memory bandwidth since the state vector is transferred in and out of memory fewer times. By default up to 4 gates are fused for single-GPU and up to 6 for multi-GPU simulations. The fusion level can **significantly** affect performance of some circuits; override it by setting `CUDAQ_MGPU_FUSE`:

```bash
CUDAQ_MGPU_FUSE=5 mpiexec -np 2 python3 program.py [...] --target nvidia --target-option mgpu,fp64
```

> **Note:** On multi-node systems without `MNNVL` support, the `nvidia` target in `mgpu` mode may fail to allocate memory. Users can disable `MNNVL` fabric-based memory sharing by setting the environment variable `UBACKEND_USE_FABRIC_HANDLE=0`.

### Tensor Network Simulators

CUDA-Q provides a couple of tensor-network simulator backends accelerated with the `cuTensorNet` library. Detailed technical information on the simulator can be found in the [cuTensorNet documentation](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html).

Tensor network simulators are suitable for large-scale simulation of certain classes of quantum circuits involving many qubits beyond the memory limit of state vector based simulators. For example, computing the expectation value of a Hamiltonian via `cudaq.observe` can be performed efficiently, thanks to `cuTensorNet` contraction optimization capability. On the other hand, conditional circuits, i.e., those with mid-circuit measurements or reset, despite being supported by both backends, may result in poor performance.

#### Multi-GPU multi-node (`tensornet`)

The `tensornet` backend represents quantum states and circuits as tensor networks in an exact form (no approximation). Measurement samples and expectation values are computed via tensor network contractions. This backend supports multi-GPU, multi-node distribution of tensor operations required to evaluate and simulate the circuit.

The `tensornet` target supports both single and double floating point precision.

To execute a program on the `tensornet` target using a *single GPU*, use the following commands:

```bash
# Double Precision (Default):
python3 program.py [...] --target tensornet

# Single Precision:
python3 program.py [...] --target tensornet --target-option fp32
```

The target can also be defined in the application code by calling

```python
cudaq.set_target('tensornet')                  # default double-precision
cudaq.set_target('tensornet', option='fp32')   # single-precision
```

If you have *multiple GPUs* available on your system, you can use MPI to automatically distribute parallelization across the visible GPUs.

> **Note:** If you installed the CUDA-Q Python wheels, distribution across multiple GPUs is currently not supported for this backend. We will add support for it in future releases. For more information, see this [GitHub issue](https://github.com/NVIDIA/cuda-quantum/issues/920).

Use the following commands to enable distribution across multiple GPUs (adjust the value of the `-np` flag as needed to reflect available GPU resources on your system):

```bash
mpiexec -np 2 python3 program.py [...] --target tensornet
```

Or, using MPI in your application code with [mpi4py](https://mpi4py.readthedocs.io/):

```bash
mpiexec -np 2 python3 -m mpi4py program.py [...] --target tensornet
```

> **Note:** MPI parallelization on the `tensornet` backend requires CUDA-Q's MPI support. Please refer to the instructions on how to [enable MPI parallelization](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#distributed-computing-with-mpi) within CUDA-Q. CUDA-Q containers are shipped with a pre-built MPI plugin; hence no additional setup is needed.

> **Note:** If the `CUTENSORNET_COMM_LIB` environment variable is set following the activation procedure described in the [cuTensorNet documentation](https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html#from-nvidia-devzone), the cuTensorNet MPI plugin will take precedence over the builtin support from CUDA-Q.

Specific aspects of the simulation can be configured by setting the following environment variables:

*   **`CUDA_VISIBLE_DEVICES=X`**: Makes the process only see GPU X on multi-GPU nodes. Each MPI process must only see its own dedicated GPU. For example, if you run 8 MPI processes on a DGX system with 8 GPUs, each MPI process should be assigned its own dedicated GPU via `CUDA_VISIBLE_DEVICES` when invoking `mpiexec` (or `mpirun`) commands.
*   **`CUDAQ_TIMING_TAGS=tags`**: When the environment variable includes 9 in the tag set, timing for the path-finding stage (Prepare) and contraction stage (Compute or Sample) are output for the user.
*   **`CUDAQ_TENSORNET_CONTROLLED_RANK=X`**: Specify the number of controlled qubits whereby the full tensor body of the controlled gate is expanded. If the number of controlled qubits is greater than this value, the gate is applied as a controlled tensor operator to the tensor network state. Default value is 1.
*   **`CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=X`**: Set this environment variable to `TRUE` (`ON`) or `FALSE` (`OFF`) to enable or disable contraction path reuse when computing expectation values. Default is `OFF`.
*   **`CUDAQ_TENSORNET_NUM_HYPER_SAMPLES=X`**: Specify the number of hyper samples used in the tensor network contraction path finder. Default value is 8 if not specified. Increasing this value will increase the path-finding time, but can decrease the contraction time if a better quality path is found (and vice versa). Hyper samples are processed in parallel using multiple host threads.
*   **`CUDAQ_TENSORNET_FIND_THREADS=X`**: Used to control the number of threads on the host used for path-finding. The default value is half of the available CPU hardware threads. For processors with 1 hardware thread per CPU core (no `SMT`), increasing this to equal the number of CPU cores can improve performance.
*   **`CUDAQ_TENSORNET_FIND_LIMIT=X`**: Set this environment variable to `TRUE` (`ON`) or `FALSE` (`OFF`) to enable or disable a heuristic to limit the path-finding time based on the predicted contraction time. When on, increasing the number of hyper samples may have no effect beyond a certain threshold due to enforcement of the time limit. Default is `ON`.
*   **`CUDAQ_TENSORNET_FIND_DETERMINISTIC=X`**: Set this environment variable to `TRUE` (`ON`) or `FALSE` (`OFF`) to enable or disable deterministic path-finding as controlled by the CUDA-Q `set_random_seed()` function. When on, the number of path-finding threads is limited to 1 and therefore this setting can significantly decrease performance. Default is `OFF`.

> **Note:** Setting the `CUDAQ_TENSORNET_*` environment variables will override any corresponding environment variables used by the `cuTensorNet` library.

#### Matrix product state (`tensornet-mps`)

The `tensornet-mps` backend is based on the matrix product state (MPS) representation of the state vector/wave function, exploiting the sparsity in the tensor network via tensor decomposition techniques such as QR and SVD. As such, this backend is an approximate simulator, whereby the number of singular values may be truncated to keep the MPS size tractable. The `tensornet-mps` backend only supports single-GPU simulation. Its approximate nature allows the `tensornet-mps` backend to handle a large number of qubits for certain classes of quantum circuits on a relatively small memory footprint.

The `tensornet-mps` target supports both single and double floating point precision.

To execute a program on the `tensornet-mps` target, use the following commands:

```bash
# Double Precision (Default):
python3 program.py [...] --target tensornet-mps

# Single Precision:
python3 program.py [...] --target tensornet-mps --target-option fp32
```

The target can also be defined in the application code by calling

```python
cudaq.set_target('tensornet-mps')                  # default double-precision
cudaq.set_target('tensornet-mps', option='fp32')   # single-precision
```

Specific aspects of the simulation can be configured by defining the following environment variables:

*   **`CUDAQ_MPS_MAX_BOND=X`**: The maximum number of singular values to keep (fixed extent truncation). Default: 64.
*   **`CUDAQ_MPS_ABS_CUTOFF=X`**: The cutoff for the largest singular value during truncation. Eigenvalues that are smaller will be trimmed out. Default: 1e-5.
*   **`CUDAQ_MPS_RELATIVE_CUTOFF=X`**: The cutoff for the maximal singular value relative to the largest eigenvalue. Eigenvalues that are smaller than this fraction of the largest singular value will be trimmed out. Default: 1e-5.
*   **`CUDAQ_MPS_SVD_ALGO=X`**: The SVD algorithm to use. Valid values are: `GESVD` (QR algorithm), `GESVDJ` (Jacobi method), `GESVDP` ([polar decomposition](https://epubs.siam.org/doi/10.1137/090774999)), `GESVDR` ([randomized methods](https://epubs.siam.org/doi/10.1137/090771806)). Default: `GESVDJ`.
*   **`CUDAQ_MPS_GAUGE=X`**: The optional gauge option to improve accuracy of the MPS simulation. Valid values are: `FREE` (gauge is disabled) or `SIMPLE` (simple update algorithm). By default, no gauge configuration is set, thus the default `cuquantum` MPS setting will be used (see the [cuQuantum doc](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/api/types.html#cutensornetstatempsgaugeoption-t)).

> **Note:** The parallelism of Jacobi method (the default `CUDAQ_MPS_SVD_ALGO` setting) gives GPU better performance on small and medium size matrices. If you expect a large number of singular values (e.g., increasing the `CUDAQ_MPS_MAX_BOND` setting), please adjust the `CUDAQ_MPS_SVD_ALGO` setting accordingly.

> **Note:** Both `tensornet-mps` and `tensornet` backends will allocate a scratch space on GPU device memory for their operations. For example, the scratch space can be used to store the contracted reduced density matrix to generate measurement bit strings.
>
> By default, these backends reserve 50% of free memory for their scratch space. This ratio can be customized using the `CUDAQ_TENSORNET_SCRATCH_SIZE_PERCENTAGE` environment variable. Valid setting must be between 5% and 95%. Users may encounter runtime errors, e.g., insufficient workspace or CUDA memory allocation errors, when setting `CUDAQ_TENSORNET_SCRATCH_SIZE_PERCENTAGE` toward its limits.

> **Note:** All floating-point data, e.g., gate matrices, noise channel Kraus operator matrices, contracted state vector, etc., are converted to the target's precision setting, if not already in that precision format. Hence, users would need to take into account potential precision loss when using the single precision setting.

---

## Part 4: Amazon Braket Backend

> Source: https://nvidia.github.io/cuda-quantum/latest/using/backends/cloud/braket.html

[Amazon Braket](https://aws.amazon.com/braket/) is a fully managed AWS service which provides Jupyter notebook environments, high-performance quantum circuit simulators, and secure, on-demand access to various quantum computers. To get started, users must enable Amazon Braket in their AWS account by following [these instructions](https://docs.aws.amazon.com/braket/latest/developerguide/braket-enable-overview.html). See the [Amazon Braket Documentation](https://docs.aws.amazon.com/braket/) and [Examples](https://github.com/amazon-braket/amazon-braket-examples/tree/main/examples/nvidia_cuda_q); available devices and regions are listed [here](https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html).

CUDA-Q programs can run on Amazon Braket as a [Hybrid Job](https://docs.aws.amazon.com/braket/latest/developerguide/braket-what-is-hybrid-job.html) — see the [Hybrid Jobs getting-started guide](https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-first.html) and the [CUDA-Q on Braket guide](https://docs.aws.amazon.com/braket/latest/developerguide/braket-using-cuda-q.html).

### Setting Credentials

After enabling Amazon Braket in AWS, set credentials using any of the documented [methods](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html). The simplest is the [AWS CLI](https://aws.amazon.com/cli/):

```bash
aws configure
```

Alternatively, set the following environment variables:

```bash
export AWS_DEFAULT_REGION="<region>"
export AWS_ACCESS_KEY_ID="<key_id>"
export AWS_SECRET_ACCESS_KEY="<access_key>"
export AWS_SESSION_TOKEN="<token>"
```

### Submitting

Select the submission target with `cudaq.set_target()`:

```python
cudaq.set_target("braket")
```

By default, jobs are submitted to the state vector simulator, `SV1`.

To specify which Amazon Braket device to use, set the `machine` parameter.

```python
device_arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
cudaq.set_target("braket", machine=device_arn)
```

where `arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet` refers to the IQM Garnet QPU.

To emulate the device locally, without submitting through the cloud, you can also set the `emulate` flag to `True`.

```python
cudaq.set_target("braket", emulate=True)
```

Set the number of shots for a kernel execution via the `shots_count` argument to `cudaq.sample` (default: 1000).

```python
cudaq.sample(kernel, shots_count=100)
```

For a complete example, see the Amazon Braket section of [15. Using Quantum Hardware Providers](#15-using-quantum-hardware-providers) in Part 1.

---

## Part 5: Python API Reference

> Source: https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html

### Program Construction

#### `cudaq.make_kernel(*args)`

Create a `Kernel`: An empty kernel function to be used for quantum program construction. This kernel is non-parameterized if it accepts no arguments, else takes the provided types as arguments.

Returns a kernel if it is non-parameterized, else a tuple containing the kernel and a `QuakeValue` for each kernel argument.

```python
# Example:
# Non-parameterized kernel.
kernel = cudaq.make_kernel()
```

```python
# Example:
# Parameterized kernel that accepts an `int` and `float` as arguments.
kernel, int_value, float_value = cudaq.make_kernel(int, float)
```

#### `class cudaq.PyKernel(argTypeList)`

The `Kernel` provides an API for dynamically constructing quantum circuits. The `Kernel` programmatically represents the circuit as an MLIR function using the Quake dialect.

##### `name`

The name of the `Kernel` function. Read-only.

**Type:** `str`

##### `arguments`

The arguments accepted by the `Kernel` function. Read-only.

**Type:** List[`QuakeValue`]

##### `argument_count`

The number of arguments accepted by the `Kernel` function. Read-only.

**Type:** int

#### `cudaq.Kernel`

alias of `PyKernel`

#### `class cudaq.PyKernelDecorator(function, verbose=False, defer_compilation=True, module=None, kernelName=None, signature=None, location=None, overrideGlobalScopedVars=None, decorator=None)`

The `PyKernelDecorator` serves as a standard Python decorator that takes the decorated function as input. The function AST is parsed and converted to a Quake MLIR representation. This is passed on to the CUDAQ runtime for execution at kernel call time.

By default, MLIR compilation is deferred until the first call to the kernel. If `defer_compilation` is set to `False`, the kernel will be compiled at declaration time instead.

##### `__call__(*args)`

Invoke the CUDA-Q kernel. JIT compilation of the kernel AOT Quake module to machine code will occur here.

##### `__str__()`

Return a string representation for this kernel as MLIR.

##### `beta_reduction(isEntryPoint, *args)`

Perform beta reduction on this kernel decorator in the current calling context. We are primary concerned with resolving the lambda lifted arguments, but the formal arguments may be supplied as well.

This beta reduction may happen in a context that is earlier than the actual call to the decorator. While this loses some of Python’s intrinsic dynamism, it allows Python kernels to be specialized and passed to algorithms written in C++ that call back to these Python kernels in a functional composition.

##### `cachedCompiledModule()`

Return the kernel’s CompiledModule cache slot, creating an empty one on first access.

##### `captured_variables()`

The list of variables captured by the kernel.

##### `compile()`

Compile the Python AST to portable Quake.

##### `enable_return_to_log()`

Enable translation from `return` statements to QIR output log

##### `static from_json(jStr, overrideDict=None)`

Convert a JSON string into a new PyKernelDecorator object.

##### `is_compiled()`

Whether the kernel has already been compiled.

##### `launch_args_required()`

This is a deeper query on the quake module. The quake module may have been specialized such that none of the arguments are, in fact, required to be provided in order to run the kernel. (Argument synthesis.)

This will analyze the designated entry-point kernel for the quake module and determine if any arguments are used and return the number used.

##### `merge_kernel(otherMod)`

Merge the kernel in this PyKernelDecorator (the ModuleOp) with the provided ModuleOp.

##### `merge_quake_source(quakeText)`

Merge a module of quake code from source text form into this decorator’s `qkeModule` attribute.

##### `prepare_call(*args, allow_no_args=False)`

Process call site arguments, capture lifted arguments and retrieve compiled module for kernel execution.

**Returns:**
*   `processed_args` (list) – The list of processed runtime arguments, including captured arguments.
*   `module` (Module) – A clone of the MLIR module to be used for kernel execution.

##### `process_call_arguments(*args, allow_no_args=False)`

Resolve the arguments passed to the decorator at call site.

##### `property qkeModule`

A target independent Quake MLIR representation of the kernel.

##### `resolve_captured_arguments()`

Resolve the captured arguments of the decorator.

These arguments get resolved in the scope of the kernel definition (lexical scoping).

##### `signatureWithCallables()`

returns True if and only if the entry-point contains callable arguments and/or return values.

##### `supports_compilation()`

Whether the kernel can be compiled for the current target.

##### `to_json()`

Convert `self` to a JSON-serialized version of the kernel such that `from_json` can reconstruct it elsewhere.

##### `static type_to_str(t)`

This converts types to strings in a clean JSON-compatible way. int -> ‘int’ list[float] -> ‘list[float]’ List[float] -> ‘list[float]’

#### `cudaq.kernel(function=None, **kwargs)`

The `cudaq.kernel` represents the CUDA-Q language function attribute that programmers leverage to indicate the following function is a CUDA-Q kernel and should be compile and executed on an available quantum coprocessor.

Verbose logging can be enabled via `verbose=True`.

### Kernel Execution

#### `cudaq.sample(kernel, *args, shots_count=1000, noise_model=None, explicit_measurements=False)`

Sample the state generated by the provided `kernel` at the given kernel `arguments` over the specified number of circuit executions (`shots_count`). Each argument in `arguments` provided can be a list or `ndarray` of arguments of the specified kernel argument type, and in this case, the `sample` functionality will be broadcasted over all argument sets and a list of `cudaq.SampleResult` instances will be returned.

**Parameters:**
*   **kernel** (`Kernel`) – The `Kernel` to execute `shots_count` times on the QPU.

*   ***arguments** (Optional[Any]) – The concrete values to evaluate the kernel function at. Leave empty if the kernel doesn’t accept any arguments. For example, if the kernel takes two `float` values as input, the `sample` call should be structured as `cudaq.sample(kernel, firstFloat, secondFloat)`. For broadcasting of the `sample` function, the arguments should be structured as a `list` or `ndarray` of argument values of the specified kernel argument type.

*   **shots_count** (Optional[int]) – The number of kernel executions on the QPU. Defaults to 1000. Key-word only.

*   **noise_model** (Optional[`NoiseModel`]) – The optional `NoiseModel` to add noise to the kernel execution on the simulator. Defaults to an empty noise model.

*   **explicit_measurements** (Optional[bool]) – Whether or not to concatenate measurements in execution order for the returned sample result.

**Returns:** A dictionary containing
the measurement count results for the `Kernel`, or a list of such results in the case of `sample` function broadcasting.

**Return type:** `SampleResult` or `list[SampleResult]`

#### `cudaq.sample_async(decorator, *args, shots_count=1000, explicit_measurements=False, noise_model=None, qpu_id=0)`

Asynchronously sample the state of the provided kernel `decorator` at the specified number of circuit executions (`shots_count`). When targeting a quantum platform with more than one QPU, the optional `qpu_id` allows for control over which QPU to enable. Will return a future whose results can be retrieved via `future.get()`.

**Parameters:**
*   **kernel** (`Kernel`) – The `Kernel` to execute `shots_count` times on the QPU.

*   ***arguments** (Optional[Any]) – The concrete values to evaluate the kernel function at.

*   **shots_count** (Optional[int]) – The number of kernel executions on the QPU. Defaults to 1000. Key-word only.

*   **explicit_measurements** (Optional[bool]) – A flag to indicate whether or not to concatenate measurements in execution order for the returned sample result.

*   **noise_model** (Optional[`NoiseModel`]) – The optional `NoiseModel` to add noise to the kernel execution on the simulator. Defaults to an empty noise model.

*   **qpu_id** (Optional[int]) – The optional identification for which QPU on the platform to target. Defaults to zero. Key-word only.

**Returns:** A dictionary containing the measurement count
results for the `Kernel`.

**Return type:** `AsyncSampleResult`

#### `cudaq.run(decorator, *args, shots_count=100, noise_model=None, qpu_id=0)`

#### `cudaq.run_async(decorator, *args, shots_count=100, noise_model=None, qpu_id=0)`

Run the provided `kernel` with the given kernel `arguments` over the specified number of circuit executions (`shots_count`) asynchronously on the specified `qpu_id`.

**Parameters:**
*   **kernel** (`Kernel`) – The `Kernel` to execute `shots_count` times on the QPU.

*   ***arguments** (Optional[Any]) – The concrete values to evaluate the kernel function at. For example, if the kernel takes two `float` values as input, the `run` call should be structured as `cudaq.run(kernel, firstFloat, secondFloat)`.

*   **shots_count** (Optional[int]) – The number of kernel executions on the QPU. Defaults to 100. Key-word only.

*   **noise_model** (Optional[`NoiseModel`]) – The optional `NoiseModel` to add noise to the kernel execution on the simulator. Defaults to an empty noise model.

*   **qpu_id** (Optional[int]) – The id of the QPU. Defaults to 0. Key-word only.

**Returns:**
A handle, which can be waited on via a `get()` method, which returns an array of `kernel` return values. The length of the list is equal to `shots_count`.

**Return type:** `AsyncRunResult`

#### `cudaq.observe(kernel, spin_operator, *args, shots_count=-1, noise_model=None, num_trajectories=None, execution=None, qpu_id=0)`

Compute the expected value of the `spin_operator` with respect to the `kernel`. If the input `spin_operator` is a list of `SpinOperator` then compute the expected value of every operator in the list and return a list of results. If the kernel accepts arguments, it will be evaluated with respect to `kernel(*arguments)`. Each argument in `arguments` provided can be a list or `ndarray` of arguments of the specified kernel argument type, and in this case, the `observe` functionality will be broadcasted over all argument sets and a list of `observe_result` instances will be returned. If both the input `spin_operator` and `arguments` are broadcast lists, a nested list of results over `arguments` then `spin_operator` will be returned.

**Parameters:**
*   **kernel** (`Kernel`) – The `Kernel` to evaluate the expectation value with respect to.

*   **spin_operator** (`SpinOperator` or `list[SpinOperator]`) – The Hermitian spin operator to calculate the expectation of, or a list of such operators.

*   ***arguments** (Optional[Any]) – The concrete values to evaluate the kernel function at.

*   **shots_count** (Optional[int]) – The number of shots to use for QPU execution. Defaults to -1 implying no shots-based sampling. Key-word only.

*   **noise_model** (Optional[`NoiseModel`]) – The optional `NoiseModel` to add noise to the kernel execution on the simulator. Defaults to an empty noise model.

*   **num_trajectories** (Optional[int]) – The optional number of trajectories for noisy simulation. Only valid if a noise model is provided. `Keyword` only.

*   **qpu_id** (Optional[int]) – The id of the QPU. Defaults to 0. Key-word only.

**Returns:**
A data-type containing the expectation value of the `spin_operator` with respect to the `kernel(*arguments)`, or a list of such results in the case of `observe` function broadcasting. If `shots_count` was provided, the `ObserveResult` will also contain a `SampleResult` dictionary.

**Return type:** `ObserveResult`

#### `cudaq.observe_async(kernel, spin_operator, *args, qpu_id=0, shots_count=-1)`

Compute the expected value of the `spin_operator` with respect to the `kernel` asynchronously. If the kernel accepts arguments, it will be evaluated with respect to `kernel(*arguments)`. When targeting a quantum platform with more than one QPU, the optional `qpu_id` allows for control over which QPU to enable. Will return a future whose results can be retrieved via `future.get()`.

**Parameters:**
*   **kernel** (`Kernel`) – The `Kernel` to evaluate the expectation value with respect to.

*   **spin_operator** (`SpinOperator`) – The Hermitian spin operator to calculate the expectation of.

*   ***arguments** (Optional[Any]) – The concrete values to evaluate the kernel function at.

*   **qpu_id** (Optional[int]) – The optional identification for which QPU on the platform to target. Defaults to zero. Key-word only.

*   **shots_count** (Optional[int]) – The number of shots to use for QPU execution. Defaults to -1 implying no shots-based sampling. Key-word only.

**Returns:** A future containing the result of the call
to observe.

**Return type:** `AsyncObserveResult`

#### `cudaq.get_state(kernel, *args)`

Return the `State` of the system after execution of the provided `kernel`.

**Parameters:**
*   **kernel** (`Kernel`) – The `Kernel` to execute on the QPU.

*   ***arguments** (Optional[Any]) – The concrete values to evaluate the kernel function at.

```python
# Example:
import numpy as np

# Define a kernel that will produce the all |11...1> state.
kernel = cudaq.make_kernel()
qubits = kernel.qalloc(3)
# Prepare qubits in the 1-state.
kernel.x(qubits)

# Get the state of the system. This will execute the provided kernel
# and, depending on the selected target, will return the state as a
# vector or matrix.
state = cudaq.get_state(kernel)
print(state)
```

#### `cudaq.get_state_async(kernel, *args, qpu_id=0)`

Asynchronously retrieve the state generated by the given quantum kernel. When targeting a quantum platform with more than one QPU, the optional `qpu_id` allows for control over which QPU to enable. Will return a future whose results can be retrieved via `future.get()`.

**Parameters:**
*   **kernel** (`Kernel`) – The `Kernel` to execute on the QPU.

*   ***arguments** (Optional[Any]) – The concrete values to evaluate the kernel function at.

*   **qpu_id** (Optional[int]) – The optional identification for which QPU on the platform to target. Defaults to zero. Key-word only.

**Returns:** Quantum state data. (state vector or density
matrix)

**Return type:** `AsyncStateResult`

#### `cudaq.vqe(*args, kernel=None, gradient_strategy=None, spin_operator=None, optimizer=None, parameter_count=None, argument_wrapper=None, shots=None)`

#### `cudaq.draw(decoratorOrFormat, *args)`

The CUDA-Q specification overloads draw. To meet that, this function uses parameter type checking. The two overloads for `cudaq.draw` are:

```python
cudaq.draw("<format>", kernel, opt_args...)
cudaq.draw(kernel, opt_args...)
```

The second overload is equivalent to using a format string of `"ascii"`.

#### `cudaq.translate(kernel, *args, format='qir:0.1')`

Return a `UTF-8` encoded string representing drawing of the execution path, i.e., the trace, of the provided `kernel`.

**Parameters:**
*   **format** (`str`) – format to translate to, <name[:version]>. Available format names: `qir`, `qir-full`, `qir-base`, `qir-adaptive`, `openqasm2`. QIR versions: `0.1` and `1.0`.

*   **kernel** (`Kernel`) – The `Kernel` to translate.

*   ***arguments** (Optional[Any]) – The concrete values to evaluate the kernel function at.

Note: Translating functions with arguments to OpenQASM 2.0 is not supported.

**Returns:**
The `UTF-8` encoded string of the circuit, without measurement operations.

```python
# Example:
import cudaq

@cudaq.kernel
def bell_pair():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])
    mz(q)

print(cudaq.translate(bell_pair, format="qir"))
```

Output (QIR, truncated):

```llvm
; ModuleID = 'LLVMDialectModule'
%Array = type opaque
%Result = type opaque
%Qubit = type opaque
...
define void @__nvqpp__mlirgen__function_variable_qreg._Z13variable_qregv() local_unnamed_addr {
  %1 = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 2)
  ... }
```

#### `cudaq.estimate_resources(kernel, *args, **kwargs)`

Performs resource counting on the given quantum kernel expression and returns an accounting of how many times each gate was applied, in addition to the total number of gates and qubits used.

**Parameters:**
*   **choice** (Any) – A choice function called to determine the outcome of measurements, in case control flow depends on measurements. Should only return either `True` or `False`. Invoking the kernel within the choice function is forbidden. Default: returns `True` or `False` with 50% probability.

*   **kernel** (`Kernel`) – The `Kernel` to count resources on

*   ***arguments** (Optional[Any]) – The concrete values to evaluate the kernel function at.

**Returns:** A dictionary containing the resource count results
for the `Kernel`.

**Return type:** `Resources`

#### `cudaq.dem_from_kernel(kernel, *args, noise_model=None)`

Generate a detector error model (DEM) from a CUDA-Q kernel.

Runs `kernel` under the internal `"dem"` execution context, captures the recorded circuit from the backend, and returns Stim’s standard `.dem` text via `stim::DetectorErrorModel::str()`. The active CUDA-Q target is unaffected; the analysis simulator is an internal, thread-local override.

**Parameters:**
*   **kernel** (`Kernel`) – The `Kernel` to analyze.

*   ***arguments** – Concrete argument values forwarded to the kernel invocation.

*   **noise_model** (`NoiseModel`, optional) – Noise model layered on top of any `apply_noise` ops already present in the kernel.

**Returns:**
UTF-8 string in Stim’s standard `.dem` file format. Consumers that need a structured DEM can parse it with `stim.DetectorErrorModel(text)`.

### Quantum Error Correction

These functions are called inside an `@cudaq.kernel` body to declare detectors and logical observables for detector-error-model generation with `cudaq.dem_from_kernel`, and to discriminate measurement handles. They are intercepted by the compiler; calling them at host scope raises `RuntimeError`. (`cudaq.to_integer` is likewise available inside kernels to pack discriminated bits into an integer.)

#### `cudaq.detector(*measurements)`

Define a detector over one or more measurement results.

A detector is a parity constraint: under noise-free execution the XOR of the referenced measurements is deterministic. Each call defines one detector. Arguments are individual `cudaq.measure_handle` values or a single `list[cudaq.measure_handle]`.

#### `cudaq.detectors(prev, curr)`

Define N detectors by pairing two measurement vectors element-wise.

Standard form for cross-round detectors: each detector `i` is the parity of `prev[i]` and `curr[i]`. Size agreement between `prev` and `curr` is checked at runtime.

#### `cudaq.logical_observable(*measurements, observable_index=0)`

Define a logical observable over one or more measurement results.

The variadic form uses `observable_index = 0`. Codes with `k` logical qubits should pass a single `list[cudaq.measure_handle]` and an explicit `observable_index` for each observable `0..k-1`.

#### `cudaq.to_bools(handles)`

Bulk-discriminate a `list[cudaq.measure_handle]` into a `list[bool]`. Device-only: this Python symbol exists so kernel code can call `cudaq.to_bools(...)`; the AST bridge intercepts the call and lowers it to a vector form `quake.discriminate` on `!cc.stdvec<!cc.measure_handle>`. Host-side invocation raises a `RuntimeError`.

### Backend Configuration

#### `cudaq.parse_args(args:Optional[Sequence[str]]=None)`

Parse command line arguments and initialize the CUDA-Q environment.

#### `cudaq.has_target(arg:str, /) → bool`

Return true if the `cudaq.Target` with the given name exists.

#### `cudaq.get_target(arg:str, /) → cudaq.Target`

#### `cudaq.get_target() → cudaq.Target`

Return the `cudaq.Target` with the given name. Will raise an exception if the name is not valid.

#### `cudaq.get_targets() → list[cudaq.Target]`

Return all available `cudaq.Target` instances on the current system.

#### `cudaq.set_target(arg0:cudaq.Target, /, **kwargs) → None`

#### `cudaq.set_target(arg0:str, /, **kwargs) → None`

Set the `cudaq.Target` to be used for CUDA-Q kernel execution, given either a `Target` instance or a target name (per the overloads above). Can provide optional, target-specific configuration data via Python kwargs.

#### `cudaq.reset_target() → None`

Reset the current `cudaq.Target` to the default.

#### `cudaq.set_noise(arg:cudaq.NoiseModel, /) → None`

Set the underlying noise model.

#### `cudaq.unset_noise() → None`

Clear backend simulation from any existing noise models.

#### `cudaq.register_set_target_callback(arg0:collections.abc.Callable[[cudaq.Target],None], arg1:str, /) → None`

Register a callback function to be executed when the runtime target is changed. The string `id` can be used to identify the callback for replacement/removal purposes.

#### `cudaq.unregister_set_target_callback(arg:str, /) → None`

Unregister a callback identified by the input identifier.

#### `cudaq.apply_noise(error_type, parameters..., targets...)`

This function is a type-safe injection of noise into a quantum kernel, occurring precisely at the call site of the function invocation. The function should be called inside CUDA-Q kernels (those annotated with `@cudaq.kernel`). The functionality is only supported for simulation targets, so it is automatically (and silently) stripped from any programs submitted to hardware targets.

**Parameters:**
*   **error_type** –

A subtype of `cudaq.KrausChannel` that implements/defines the desired noise mechanisms as Kraus channels (e.g. `cudaq.Depolarization2`). If you want to use a custom `cudaq.KrausChannel` (i.e. not built-in to CUDA-Q), it must first be registered _outside the kernel_ with `register_channel`, like this:

```python
class CustomNoiseChannel(cudaq.KrausChannel):
    num_parameters = 1
    num_targets = 1

    def __init__(self, params: list[float]):
        cudaq.KrausChannel.__init__(self)
        # Example: Create Kraus ops based on params
        p = params[0]
        k0 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]],
                      dtype=np.complex128)
        k1 = np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]],
                      dtype=np.complex128)

        # Create KrausOperators and add to channel
        self.append(cudaq.KrausOperator(k0))
        self.append(cudaq.KrausOperator(k1))

        self.noise_type = cudaq.NoiseModelType.Unknown

noise = cudaq.NoiseModel()
noise.register_channel(CustomNoiseChannel)
```

*   **parameters** –

The precise argument pack depend on the concrete `cudaq.KrausChannel` being used. The arguments are a concatenated list of parameters and targets. For example, to apply a 2-qubit depolarization channel, which has `num_parameters = 1` and `num_targets = 2`, one would write the call like this:

```python
q, r = cudaq.qubit(), cudaq.qubit()
cudaq.apply_noise(cudaq.Depolarization2, 0.1, q, r)
```

*   **targets** – The target qubits on which to apply the noise

#### `cudaq.initialize_cudaq(option:str|None=None, emulate:bool|None=None, target:str|None=None) → None`

Initialize the CUDA-Q environment.

#### `cudaq.num_available_gpus() → int`

The number of available GPUs detected on the system.

#### `cudaq.set_random_seed(arg:int, /) → None`

Provide the seed for backend quantum kernel simulation.

### Dynamics

#### `cudaq.evolve(hamiltonian, dimensions={}, schedule=None, initial_state=None, collapse_operators=[], observables=[], store_intermediate_results=IntermediateResultSave.NONE, integrator=None, shots_count=None, max_batch_size=None) → EvolveResult | Sequence[EvolveResult]`

Accepted types: `hamiltonian` is a single operator (`MatrixOperator`, `SpinOperator`, `BosonOperator`, `FermionOperator`, one of their `*Term` variants, or `ScalarOperator`), a `SuperOperator`, or a sequence of operators / `SuperOperator`s (batch simulation). `initial_state` is a `State`, an `InitialStateType` enum value, or a sequence of states (batching). `collapse_operators` is a flat sequence of operators, or a nested sequence with one list per Hamiltonian in a batch. `store_intermediate_results` is a `cudaq.IntermediateResultSave` value (or bool). `integrator` is an optional `BaseIntegrator`.

Computes the time evolution of one or more initial state(s) under the defined operator(s).

**Parameters:**
*   **hamiltonian** – Operator that describes the behavior of a quantum system without noise.

*   **dimensions** – A mapping that specifies the number of levels, that is the dimension, of each degree of freedom that any of the operator arguments acts on.

*   **schedule** – A sequence that generates a mapping of keyword arguments to their respective value. The keyword arguments are the parameters needed to evaluate any of the operators passed to `evolve`. All required parameters for evaluating an operator and their documentation, if available, can be queried by accessing the `parameter` property of the operator.

*   **initial_state** – A single state or a sequence of states of a quantum system.

*   **collapse_operators** – A sequence of operators that describe the influence of noise on the quantum system.

*   **observables** – A sequence of operators for which to compute their expectation value during evolution. If `store_intermediate_results` is not None, the expectation values are computed after each step in the schedule, and otherwise only the final expectation values at the end of the evolution are computed.

*   **shots_count** – Optional integer, if provided, it is the number of shots to use for QPU execution.

**Returns:**
A single evolution result if a single initial state is provided, or a sequence of evolution results representing the data computed during the evolution of each initial state. See `EvolveResult` for more information about the data computed during evolution.

#### `cudaq.evolve_async(hamiltonian, dimensions={}, schedule=None, initial_state=None, collapse_operators=[], observables=[], store_intermediate_results=IntermediateResultSave.NONE, integrator=None, shots_count=None) → AsyncEvolveResult | Sequence[AsyncEvolveResult]`

Same parameter semantics as `cudaq.evolve`, except `hamiltonian` must be a single operator (no `SuperOperator` sequence) and there is no `max_batch_size`.

Asynchronously computes the time evolution of one or more initial state(s) under the defined operator(s). See `cudaq.evolve` for more details about the parameters passed here.

**Returns:**
The handle to a single evolution result if a single initial state is provided, or a sequence of handles to the evolution results representing the data computed during the evolution of each initial state. See the `EvolveResult` for more information about the data computed during evolution.

#### `class cudaq.Schedule(steps:Iterable[Any], parameters:Iterable[str], get_value:Optional[Callable[[str,Any],numpy.complexfloating|complex|float|int]]=None)`

Represents an iterator that produces all values needed for evaluating an operator expression at different time steps.

#### `class cudaq.dynamics.integrator.BaseIntegrator(**kwargs)`

An abstract wrapper around ODE integrator to ensure a common interface for master equation solver usage.

#### `cudaq.dynamics.helpers.InitialState`

alias of `InitialStateType`

#### `class cudaq.InitialStateType(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)`

Enumeration describing the initial state type to be created in the backend

#### `class cudaq.IntermediateResultSave(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)`

Enum to specify how intermediate results should be saved during the dynamics evolution.

### Operators

#### `cudaq.operators.OperatorSum`

alias of `MatrixOperator` | `SpinOperator` | `BosonOperator` | `FermionOperator`

#### `cudaq.operators.ProductOperator`

alias of `MatrixOperatorTerm` | `SpinOperatorTerm` | `BosonOperatorTerm` | `FermionOperatorTerm`

#### `cudaq.operators.ElementaryOperator`

alias of `SpinOperatorElement` | `BosonOperatorElement` | `FermionOperatorElement` | `MatrixOperatorElement`

#### `class cudaq.operators.ScalarOperator(*args, **kwargs)`

##### `classmethod const(constant_value:numpy.complexfloating|complex|float|int) → ScalarOperator`

Creates a scalar operator that has a constant value.

##### `evaluate`

Evaluated value of the operator.

##### `is_constant`

Returns true if the scalar is a constant value.

##### `property parameters`

Returns a dictionary that maps each parameter name to its description.

##### `to_matrix(dimensions:Mapping[int,int]={}, **kwargs:numpy.complexfloating|complex|float|int) → ndarray[Any,dtype[complexfloating]]`

Class method for consistency with other operator classes. Invokes the generator with the given keyword arguments.

**Parameters:**
*   **dimensions** – (unused, passed for consistency) A mapping that specifies the number of levels, that is the dimension, of each degree of freedom that the operator acts on.

*   **kwargs** – Keyword arguments needed to evaluate the generator. All required parameters and their documentation, if available, can be queried by accessing the `parameter` property.

**Returns:**
An array with a single element corresponding to the value of the operator for the given keyword arguments.

#### `class cudaq.operators.RydbergHamiltonian(atom_sites:Iterable[tuple[float,float]], amplitude:ScalarOperator, phase:ScalarOperator, delta_global:ScalarOperator, atom_filling:Optional[Iterable[int]]=[], delta_local:Optional[tuple[cudaq.ScalarOperator,Iterable[float]]]=None)`

Representation for the time-dependent Hamiltonian which is simulated by analog neutral-atom machines such as QuEra’s Aquila and Pasqal’s Fresnel. Ref: https://docs.aws.amazon.com/braket/latest/developerguide/braket-quera-submitting-analog-program-aquila.html#braket-quera-ahs-program-schema

Instantiate an operator consumable by the `evolve` API using the supplied parameters.

**Parameters:**
*   **atom_sites** – List of 2-d coordinates where the tweezers trap atoms.

*   **amplitude** – time and value points of driving amplitude, Omega(t).

*   **phase** – time and value points of driving phase, phi(t).

*   **delta_global** – time and value points of driving detuning, Delta_global(t).

*   **atom_filling** – typing.Optional. Marks atoms that occupy the trap sites with 1, and empty sites with 0. If not provided, all are set to 1, i.e. filled.

*   **delta_local** – typing.Optional. A tuple of time and value points of the time-dependent factor of the local detuning magnitude, Delta_local(t), and site-dependent factor of the local detuning magnitude, h_k, a dimensionless number between 0.0 and 1.0

#### `class cudaq.SuperOperator(*args, **kwargs)`

##### `left_multiply`

##### `left_right_multiply`

##### `right_multiply`

#### `cudaq.operators.define(id:str, expected_dimensions:Sequence[int], create:Callable[[...],ndarray[Any,dtype[complexfloating]]], override:bool=False) → None`

Defines a matrix operator element with the given id. After definition, an the defined elementary operator can be instantiated by providing the operator id as well as the degree(s) of freedom that it acts on. A matrix operator element is a parameterized object acting on certain degrees of freedom. To evaluate an operator, for example to compute its matrix, the level, that is the dimension, for each degree of freedom it acts on must be provided, as well as all additional parameters. Additional parameters must be provided in the form of keyword arguments.

Note: The dimensions passed during operator evaluation are automatically validated against the expected dimensions specified during definition - the `create` function does not need to do this.

**Parameters:**
*   **op_id** – A string that uniquely identifies the defined operator.

*   **expected_dimensions** – defines the number of levels, that is the dimension, for each degree of freedom in canonical (that is sorted) order. A negative or zero value for one (or more) of the expected dimensions indicates that the operator is defined for any dimension of the corresponding degree of freedom.

*   **create** – Takes any number of complex-valued arguments and returns the matrix representing the operator in canonical order. If the matrix can be defined for any number of levels for one or more degree of freedom, the `create` function must take an argument called `dimension` (or `dim` for short), if the operator acts on a single degree of freedom, and an argument called `dimensions` (or `dims` for short), if the operator acts on multiple degrees of freedom.

*   **override** – if True it allows override the definition. (default: False)

#### `cudaq.operators.instantiate(op_id:str, degrees:Union[int,Iterable[int]]) → MatrixOperatorTerm`

Instantiates a product operator containing a previously defined operator element.

**Parameters:**
*   **operator_id** – The id of the operator element as specified when it was defined.

*   **degrees** – The degree(s) of freedom that the operator acts on.

#### Spin Operators

#### `class cudaq.operators.spin.SpinOperator(*args, **kwargs)`

Deprecated members: `for_each_pauli`/`for_each_term` (use standard iteration), `get_coefficient` (use `evaluate_coefficient` per term), `get_qubit_count` (use `qubit_count`), `get_term_count` (use `term_count`), `get_raw_data`, `is_identity` (will only be supported per term in future releases), `to_string` (use `str` or `get_pauli_word` per term).

##### `canonicalize`

Overloaded function.

1. `canonicalize(self) -> cudaq.SpinOperator`

Removes all identity operators from the operator.

2. `canonicalize(self, arg: collections.abc.Set[int], /) -> cudaq.SpinOperator`

Expands the operator to act on all given degrees, applying identities as needed. If an empty set is passed, canonicalizes all terms in the sum to act on the same degrees of freedom.

##### `copy`

Creates a copy of the operator.

##### `property degrees`

Returns a vector that lists all degrees of freedom that the operator targets. The order of degrees is from smallest to largest and reflects the ordering of the matrix returned by `to_matrix`. Specifically, the indices of a statevector with two qubits are {00, 01, 10, 11}. An ordering of degrees {0, 1} then indicates that a state where the qubit with index 0 equals 1 with probability 1 is given by the vector {0., 1., 0., 0.}.

##### `distribute_terms`

Partitions the terms of the sums into the given number of separate sums.

##### `dump`

Prints the string representation of the operator to the standard output.

##### `empty`

##### `empty_op`

##### `from_json`

##### `from_word`

##### `identity`

##### `property max_degree`

Returns the smallest index of the degrees of freedom that the operator targets.

##### `property min_degree`

Returns the smallest index of the degrees of freedom that the operator targets.

##### `property parameters`

Returns a dictionary that maps each parameter name to its description.

##### `property qubit_count`

Return the number of qubits this operator acts on.

##### `random`

##### `serialize`

Returns the serialized data representation of the operator.

##### `property term_count`

Returns the number of terms in the operator.

##### `to_json`

Convert spin_op to JSON string: `'[d1, d2, d3, …]'`

##### `to_matrix`

Overloaded function.

1. `to_matrix(self, dimensions: collections.abc.Mapping[int, int] | None = None, parameters: collections.abc.Mapping[str, complex] | None = None, invert_order: bool = False) -> object`

Returns the matrix representation of the operator.The matrix is ordered according to the convention (endianness) used in CUDA-Q, and the ordering returned by `degrees`. This order can be inverted by setting the optional `invert_order` argument to `True`. See also the documentation for `degrees` for more detail.

2. `to_matrix(self, arg0: collections.abc.Mapping[int, int], /, **kwargs) -> object`

Same behavior as the overload above (positional `dimensions`, parameters as keyword arguments).

3. `to_matrix(self, **kwargs) -> object`

Returns the matrix representation of the operator, passing parameters as keyword arguments.

##### `to_sparse_matrix`

Return the sparse matrix representation of the operator. This representation is a `Tuple[list[complex], list[int], list[int]]`, encoding the non-zero values, rows, and columns of the matrix. This format is supported by `scipy.sparse.csr_array`.The matrix is ordered according to the convention (endianness) used in CUDA-Q, and the ordering returned by `degrees`. This order can be inverted by setting the optional `invert_order` argument to `True`. See also the documentation for `degrees` for more detail.

##### `trim`

Removes all terms from the sum for which the absolute value of the coefficient is below the given tolerance.

#### `class cudaq.operators.spin.SpinOperatorTerm(*args, **kwargs)`

Deprecated members: `for_each_pauli` (use standard iteration), `distribute_terms` (instantiate a `SpinOperator` and call it there), `get_coefficient` (use `evaluate_coefficient`), `get_qubit_count` (use `qubit_count`), `get_raw_data`, `to_string` (use `str` or `get_pauli_word`).

##### `canonicalize`

Overloaded function.

1. `canonicalize(self) -> cudaq.SpinOperatorTerm`

Removes all identity operators from the operator.

2. `canonicalize(self, arg: collections.abc.Set[int], /) -> cudaq.SpinOperatorTerm`

Expands the operator to act on all given degrees, applying identities as needed. The canonicalization will throw a runtime exception if the operator acts on any degrees of freedom that are not included in the given set.

##### `property coefficient`

Returns the unevaluated coefficient of the operator. The coefficient is a callback function that can be invoked with the `evaluate` method.

##### `copy`

Creates a copy of the operator.

##### `property degrees`

Returns a vector that lists all degrees of freedom that the operator targets, ordered smallest to largest; ordering semantics are identical to `SpinOperator.degrees`.

##### `dump`

Prints the string representation of the operator to the standard output.

##### `evaluate_coefficient`

Returns the evaluated coefficient of the product operator. The parameters is a map of parameter names to their concrete, complex values.

##### `from_json`

##### `get_binary_symplectic_form`

Gets the binary symplectic representation of this operator.

##### `get_pauli_word`

Gets the Pauli word representation of this product operator.

##### `is_identity`

Checks if all operators in the product are the identity. Note: this function returns true regardless of the value of the coefficient.

##### `property max_degree`

Returns the smallest index of the degrees of freedom that the operator targets.

##### `property min_degree`

Returns the smallest index of the degrees of freedom that the operator targets.

##### `property ops_count`

Returns the number of operators in the product.

##### `property parameters`

Returns a dictionary that maps each parameter name to its description.

##### `property qubit_count`

Return the number of qubits this operator acts on.

##### `serialize`

Returns the serialized data representation of the operator.

##### `property term_count`

Returns the number of terms in the operator. Always returns 1.

##### `property term_id`

The term id uniquely identifies the operators and targets (degrees) that they act on, but does not include information about the coefficient.

##### `to_json`

Convert spin_op to JSON string: `'[d1, d2, d3, …]'`

##### `to_matrix`

Same overloads and semantics as `SpinOperator.to_matrix`.

##### `to_sparse_matrix`

Same as `SpinOperator.to_sparse_matrix`: returns `Tuple[list[complex], list[int], list[int]]` (non-zero values, rows, columns; `scipy.sparse.csr_array`-compatible), ordered per `degrees`, invertible via `invert_order=True`.

#### `class cudaq.operators.spin.SpinOperatorElement(*args, **kwargs)`

##### `as_pauli`

Returns the Pauli representation of the operator.

##### `property degrees`

Returns a vector that lists all degrees of freedom that the operator targets.

##### `property target`

Returns the degree of freedom that the operator targets.

##### `to_matrix`

Returns the matrix representation of the operator.

##### `to_string`

Returns the string representation of the operator.

#### `enum cudaq.spin.Pauli(value)`

An enumeration representing the types of Pauli matrices.

Valid values are as follows:

##### `X = Pauli.X`

##### `Y = Pauli.Y`

##### `Z = Pauli.Z`

##### `I = Pauli.I`

#### Fermion Operators

#### `class cudaq.operators.fermion.FermionOperator(*args, **kwargs)`

Member API identical to `SpinOperator` (same signatures, overloads, and semantics) for: `canonicalize`, `copy`, `degrees`, `distribute_terms`, `dump`, `empty`, `identity`, `max_degree`, `min_degree`, `parameters`, `term_count`, `to_matrix`, `to_sparse_matrix`, `trim`.

#### `class cudaq.operators.fermion.FermionOperatorTerm(*args, **kwargs)`

Member API identical to `SpinOperatorTerm` (same signatures, overloads, and semantics) for: `coefficient`, `copy`, `degrees`, `dump`, `evaluate_coefficient`, `max_degree`, `min_degree`, `ops_count`, `parameters`, `term_id`, `to_matrix`, `to_sparse_matrix`.

##### `canonicalize`

Overloaded function.

1. `canonicalize(self) -> cudaq.FermionOperatorTerm`

Removes all identity operators from the operator.

2. `canonicalize(self, arg: collections.abc.Set[int], /) -> cudaq.FermionOperatorTerm`

Expands the operator to act on all given degrees, applying identities as needed. The canonicalization will throw a runtime exception if the operator acts on any degrees of freedom that are not included in the given set.

##### `is_identity`

Checks if all operators in the product are the identity. Note: this function returns true regardless of the value of the coefficient.

#### `class cudaq.operators.fermion.FermionOperatorElement(*args, **kwargs)`

##### `property degrees`

Returns a vector that lists all degrees of freedom that the operator targets.

##### `property target`

Returns the degree of freedom that the operator targets.

##### `to_matrix`

Returns the matrix representation of the operator.

##### `to_string`

Returns the string representation of the operator.

#### Boson Operators

#### `class cudaq.operators.boson.BosonOperator(*args, **kwargs)`

Member API identical to `SpinOperator` (same signatures, overloads, and semantics) for: `canonicalize`, `copy`, `degrees`, `distribute_terms`, `dump`, `empty`, `identity`, `max_degree`, `min_degree`, `parameters`, `term_count`, `to_matrix`, `to_sparse_matrix`, `trim`.

#### `class cudaq.operators.boson.BosonOperatorTerm(*args, **kwargs)`

Member API identical to `SpinOperatorTerm` (same signatures, overloads, and semantics) for: `coefficient`, `copy`, `degrees`, `dump`, `evaluate_coefficient`, `max_degree`, `min_degree`, `ops_count`, `parameters`, `term_id`, `to_matrix`, `to_sparse_matrix`.

##### `canonicalize`

Overloaded function.

1. `canonicalize(self) -> cudaq.BosonOperatorTerm`

Removes all identity operators from the operator.

2. `canonicalize(self, arg: collections.abc.Set[int], /) -> cudaq.BosonOperatorTerm`

Expands the operator to act on all given degrees, applying identities as needed. The canonicalization will throw a runtime exception if the operator acts on any degrees of freedom that are not included in the given set.

##### `is_identity`

Checks if all operators in the product are the identity. Note: this function returns true regardless of the value of the coefficient.

#### `class cudaq.operators.boson.BosonOperatorElement(*args, **kwargs)`

##### `property degrees`

Returns a vector that lists all degrees of freedom that the operator targets.

##### `property target`

Returns the degree of freedom that the operator targets.

##### `to_matrix`

Returns the matrix representation of the operator.

##### `to_string`

Returns the string representation of the operator.

#### General Operators

#### `class cudaq.operators.MatrixOperator(*args, **kwargs)`

Member API identical to `SpinOperator` (same signatures, overloads, and semantics) for: `canonicalize`, `copy`, `distribute_terms`, `dump`, `empty`, `identity`, `max_degree`, `min_degree`, `parameters`, `term_count`, `to_matrix`, `trim`.

##### `property degrees`

Returns a vector that lists all degrees of freedom that the operator targets.

#### `class cudaq.operators.MatrixOperatorTerm(*args, **kwargs)`

Member API identical to `SpinOperatorTerm` (same signatures, overloads, and semantics) for: `coefficient`, `copy`, `degrees`, `dump`, `evaluate_coefficient`, `max_degree`, `min_degree`, `ops_count`, `parameters`, `term_id`, `to_matrix`.

##### `canonicalize`

Overloaded function.

1. `canonicalize(self) -> cudaq.MatrixOperatorTerm`

Removes all identity operators from the operator.

2. `canonicalize(self, arg: collections.abc.Set[int], /) -> cudaq.MatrixOperatorTerm`

Expands the operator to act on all given degrees, applying identities as needed. The canonicalization will throw a runtime exception if the operator acts on any degrees of freedom that are not included in the given set.

##### `is_identity`

Checks if all operators in the product are the identity. Note: this function returns true regardless of the value of the coefficient.

#### `class cudaq.operators.MatrixOperatorElement(*args, **kwargs)`

##### `classmethod define(id:str, expected_dimensions:Sequence[int], create:Callable[[...],ndarray[Any,dtype[complexfloating]]], override:bool=False) → None`

Creates the definition of an elementary operator with the given id.

##### `property degrees`

Returns a vector that lists all degrees of freedom that the operator targets.

##### `property expected_dimensions`

The number of levels, that is the dimension, for each degree of freedom in canonical order that the operator acts on. A value of zero or less indicates that the operator is defined for any dimension of that degree.

##### `property id`

Returns the id used to define and instantiate the operator.

##### `property parameters`

Returns a dictionary that maps each parameter name to its description.

##### `to_matrix`

Returns the matrix representation of the operator.

##### `to_string`

Returns the string representation of the operator.

#### `cudaq.operators.custom.define(id:str, expected_dimensions:Sequence[int], create:Callable[[...],ndarray[Any,dtype[complexfloating]]], override:bool=False) → None`

Defines a matrix operator element with the given id. After definition, an the defined elementary operator can be instantiated by providing the operator id as well as the degree(s) of freedom that it acts on. A matrix operator element is a parameterized object acting on certain degrees of freedom. To evaluate an operator, for example to compute its matrix, the level, that is the dimension, for each degree of freedom it acts on must be provided, as well as all additional parameters. Additional parameters must be provided in the form of keyword arguments.

Note: The dimensions passed during operator evaluation are automatically validated against the expected dimensions specified during definition - the `create` function does not need to do this.

**Parameters:**
*   **op_id** – A string that uniquely identifies the defined operator.

*   **expected_dimensions** – defines the number of levels, that is the dimension, for each degree of freedom in canonical (that is sorted) order. A negative or zero value for one (or more) of the expected dimensions indicates that the operator is defined for any dimension of the corresponding degree of freedom.

*   **create** – Takes any number of complex-valued arguments and returns the matrix representing the operator in canonical order. If the matrix can be defined for any number of levels for one or more degree of freedom, the `create` function must take an argument called `dimension` (or `dim` for short), if the operator acts on a single degree of freedom, and an argument called `dimensions` (or `dims` for short), if the operator acts on multiple degrees of freedom.

*   **override** – if True it allows override the definition. (default: False)

#### `cudaq.operators.custom.instantiate(op_id:str, degrees:Union[int,Iterable[int]]) → MatrixOperatorTerm`

Instantiates a product operator containing a previously defined operator element.

**Parameters:**
*   **operator_id** – The id of the operator element as specified when it was defined.

*   **degrees** – The degree(s) of freedom that the operator acts on.

### Data Types

#### `class cudaq.SimulationPrecision(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)`

Enumeration describing the precision of the underlying simulation.

#### `class cudaq.Target`

The `cudaq.Target` represents the underlying infrastructure that CUDA-Q kernels will execute on. Instances of `cudaq.Target` describe what simulator they may leverage, the quantum_platform required for execution, and a description for the target.

##### `property description`

A string describing the features for this `cudaq.Target`.

##### `get_precision`

Return the simulation precision for the current target.

##### `is_emulated`

Returns true if the emulation mode for the target has been activated.

##### `is_remote`

Returns true if the target consists of a remote REST QPU.

##### `property name`

The name of the `cudaq.Target`.

##### `num_qpus`

Return the number of QPUs available in this `cudaq.Target`.

##### `property platform`

The name of the quantum_platform implementation this `cudaq.Target` leverages.

##### `property simulator`

The name of the simulator this `cudaq.Target` leverages. This will be empty for physical QPUs.

#### `class cudaq.State`

A data-type representing the quantum state of the internal simulator. This type is not user-constructible and instances can only be retrieved via the `cudaq.get_state(...)` function or the static `cudaq.State.from_data()` method.

##### `amplitude`

Overloaded function.

1. `amplitude(self, arg: collections.abc.Sequence[int], /) -> complex`

Return the amplitude of a state in computational basis.

```python
# Example:
# Create a simulation state.
state = cudaq.get_state(kernel)
# Return the amplitude of |0101>, assuming this is a 4-qubit state.
amplitude = state.amplitude([0,1,0,1])
```

1. `amplitude(self, arg: str, /) -> complex`

Return the amplitude of a state in computational basis.

(The string overload works the same way: `amplitude = state.amplitude('0101')`.)

##### `amplitudes`

Overloaded function.

1. `amplitudes(self, arg: collections.abc.Sequence[collections.abc.Sequence[int]], /) -> list[complex]`

Return the amplitude of a list of states in computational basis.

```python
# Example:
# Create a simulation state.
state = cudaq.get_state(kernel)
# Return the amplitude of |0101> and |1010>, assuming this is a 4-qubit state.
amplitudes = state.amplitudes([[0,1,0,1], [1,0,1,0]])
```

1. `amplitudes(self, arg: collections.abc.Sequence[str], /) -> list[complex]`

Return the amplitudes of a list of states in computational basis.

(The string overload works the same way: `amplitudes = state.amplitudes(['0101', '1010'])`.)

##### `dump`

Print the state to the console.

##### `from_data`

##### `getTensor`

Return the `idx` tensor making up this state representation.

##### `getTensors`

Return all the tensors that comprise this state representation.

##### `get_state_refval`

Convert the address of the state object to an integer.

##### `is_on_gpu`

Return True if this state is on the GPU.

##### `num_qubits`

Returns the number of qubits represented by this state.

##### `overlap`

Overloaded function.

1. `overlap(self, arg: cudaq.State, /) -> complex`

Compute the overlap between the provided `State`’s.

2. `overlap(self, arg: object, /) -> complex`

Compute the overlap between the provided `State`’s.

3. `overlap(self, arg: object, /) -> complex`

Compute overlap with general CuPy device array.

##### `to_numpy`

Convert to a NumPy array.

#### `class cudaq.Tensor`

The `Tensor` describes a pointer to simulation data as well as the rank and extents for that tensorial data it represents.

#### `class cudaq.QuakeValue(mlirValue, pyKernel, size=None)`

A `QuakeValue` represents a handle to an individual function argument of a `Kernel`, or a return value from an operation within it. As documented in `make_kernel()`, a `QuakeValue` can hold values of the following types: int, float, list/List, `qubit`, or `qvector`. The `QuakeValue` can also hold kernel operations such as qubit allocations and measurements.

The arithmetic operators below each raise **RuntimeError** if the underlying `QuakeValue` type is not a float. Example: after `kernel, value = cudaq.make_kernel(float)`, expressions such as `value + 5.0`, `5.0 + value`, `value - 5.0`, `5.0 - value`, `-value`, `value * 5.0`, `5.0 * value` each return a new `QuakeValue`.

##### `__add__(other)`

Return the sum of `self` (`QuakeValue`) and `other` (float).

##### `__radd__(other)`

Return the sum of `other` (float) and `self` (`QuakeValue`).

##### `__sub__(other)`

Return the difference of `self` (`QuakeValue`) and `other` (float).

##### `__rsub__(other)`

Return the difference of `other` (float) and `self` (`QuakeValue`).

##### `__neg__()`

Return the negation of `self` (`QuakeValue`).

##### `__mul__(other)`

Return the product of `self` (`QuakeValue`) with `other` (float).

##### `__rmul__(other)`

Return the product of `other` (float) with `self` (`QuakeValue`).

##### `__getitem__(idx)`

Return the element of `self` at the provided `index`.

**Note:** Only `list` or `qvector` type `QuakeValue`’s may be indexed.

**Parameters:**
**index** (int) – The element of `self` that you’d like to return.

**Returns:**
A new `QuakeValue` for the `index` element of `self`.

**Return type:** `QuakeValue`

**Raises:**
**RuntimeError** – if `self` is a non-subscriptable `QuakeValue`.

##### `slice(startIdx, count)`

Return a slice of the given `QuakeValue` as a new `QuakeValue`.

**Note:** The underlying `QuakeValue` must be a `list` or `veq`.

**Parameters:**
*   **start** (int) – The index to begin the slice from.

*   **count** (int) – The number of elements to extract after the `start` index.

**Returns:**
A new `QuakeValue` containing a slice of `self` from the `start` element to the `start + count` element.

**Return type:** `QuakeValue`

#### `class cudaq.qubit(*args, **kwargs)`

The qubit is the primary unit of information in a quantum computer. Qubits can be created individually or as part of larger registers.

#### `cudaq.qreg`

alias of `qvector`

#### `class cudaq.qvector(*args, **kwargs)`

An owning, dynamically sized container for qubits. The semantics of the `qvector` follows that of a `std::vector` or list for qubits.

#### `class cudaq.measure_handle(*args, **kwargs)`

A handle to a measurement event recorded inside a CUDA-Q kernel.

Returned by `mz` / `mx` / `my` inside an `@cudaq.kernel` body (scalar form on a single qubit; vector form on a `qvector` / `qview`). The classical outcome is read by coercing the handle to `bool` in any Python `bool` context, and the AST bridge inserts a `quake.discriminate` at the coercion site. `cudaq.to_bools(handles)` is the bulk counterpart on a `list[measure_handle]`.

Instantiating `cudaq.measure_handle()` at host scope raises `KernelTypeError` (a `RuntimeError` subclass) since it is device-only.

#### `class cudaq.ComplexMatrix(*args, **kwargs)`

The `ComplexMatrix` is a thin wrapper around a matrix of complex<double> elements.

##### `__getitem__`

Return the matrix element at i, j.

##### `__str__`

Returns the string representation of the matrix.

##### `dump`

Prints the matrix to the standard output.

##### `minimal_eigenvalue`

Return the lowest eigenvalue for this `ComplexMatrix`.

##### `num_columns`

Returns the number of columns in the matrix.

##### `num_rows`

Returns the number of rows in the matrix.

##### `to_numpy`

Overloaded function.

1. `to_numpy(self) -> object`

Convert to a NumPy array.

2. `to_numpy(self) -> object`

Convert `ComplexMatrix` to numpy.ndarray.

#### `class cudaq.SampleResult(*args, **kwargs)`

A data-type containing the results of a call to `sample()`. This includes all measurement counts data from both mid-circuit and terminal measurements.

**Note:** Conditional logic on mid-circuit measurements is no longer supported with `sample`. Use `run` instead.

##### `__getitem__`

Return the measurement counts for the given `bitstring`.

**Parameters:**
**bitstring** (str) – The binary string to return the measurement data of.

**Returns:**
The number of times the given `bitstring` was measured during the `shots_count` number of executions on the QPU.

**Return type:** float

##### `__iter__`

Iterate through the `SampleResult` dictionary.

##### `__len__`

Return the number of elements in `self`. Equivalent to the number of uniquely measured bitstrings.

##### `clear`

Clear out all metadata from `self`.

##### `count`

Return the number of times the given bitstring was observed.

**Parameters:**
*   **bitstring** (str) – The binary string to return the measurement counts for.

*   **register_name** (Optional[str]) – The optional measurement register name to extract the probability from. Defaults to the ‘__global__’ register.

**Returns:**
The number of times the given bitstring was measured during the experiment.

**Return type:** int

##### `deserialize`

Deserialize this SampleResult from an existing vector of integers adhering to the implicit encoding.

##### `dump`

Print a string of the raw measurement counts data to the terminal.

##### `expectation`

Return the expectation value in the Z-basis of the `Kernel` that was sampled.

##### `expectation_z`

Return the expectation value in the Z-basis of the `Kernel` that was sampled.

##### `get_marginal_counts`

Extract the measurement counts data for the provided subset of qubits (`marginal_indices`).

**Parameters:**
*   **marginal_indices** (list[int]) – A list of the qubit indices to extract the measurement data from.

*   **register_name** (Optional[str]) – The optional measurement register name to extract the counts data from. Defaults to the ‘__global__’ register.

**Returns:**
A new `SampleResult` dictionary containing the extracted measurement data.

**Return type:** `SampleResult`

##### `get_register_counts`

Extract the provided sub-register (`register_name`) as a new `SampleResult`.

##### `get_sequential_data`

Return the data from the given register (`register_name`) as it was collected sequentially. A list of measurement results, not collated into a map.

##### `get_total_shots`

Get the total number of shots in the sample result

##### `items`

Return the key/value pairs in this `SampleResult` dictionary.

##### `most_probable`

Return the bitstring that was measured most frequently in the experiment.

**Parameters:**
**register_name** (Optional[str]) – The optional measurement register name to extract the most probable bitstring from. Defaults to the ‘__global__’ register.

**Returns:**
The most frequently measured binary string during the experiment.

**Return type:** str

##### `probability`

Return the probability of measuring the given `bitstring`.

**Parameters:**
*   **bitstring** (str) – The binary string to return the measurement probability of.

*   **register_name** (Optional[str]) – The optional measurement register name to extract the probability from. Defaults to the ‘__global__’ register.

**Returns:**
The probability of measuring the given `bitstring`. Equivalent to the proportion of the total times the bitstring was measured vs. the number of experiments (`shots_count`).

**Return type:** float

##### `property register_names`

(self) -> list[str]

##### `serialize`

Serialize this SampleResult to a vector of integer encoding.

##### `values`

Return all values (the counts) in this `SampleResult` dictionary.

#### `class cudaq.AsyncSampleResult(impl, mod=None)`

#### `class cudaq.ObserveResult(*args, **kwargs)`

A data-type containing the results of a call to `observe()`. This includes any measurement counts data, as well as the global expectation value of the user-defined `spin_operator`.

##### `counts`

Overloaded function.

1. `counts(self) -> cudaq.SampleResult`

Returns a `SampleResult` dictionary with the measurement results from the experiment. The result for each individual term of the `spin_operator` is stored in its own measurement register. Each register name corresponds to the string representation of the spin term (without any coefficients).

2. `counts(self, sub_term: cudaq.SpinOperatorTerm) -> cudaq.SampleResult`

3. `counts(self, sub_term: object) -> cudaq.SampleResult`

Given a `sub_term` of the global `spin_operator` that was passed to `observe()`, return its measurement counts.

**Parameters:**
**sub_term** (`SpinOperator`) – An individual sub-term of the `spin_operator`.

**Returns:**
The measurement counts data for the individual `sub_term`.

**Return type:** `SampleResult`

4. `counts(self, sub_term: cudaq.SpinOperator) -> cudaq.SampleResult`

Deprecated - ensure to pass a SpinOperatorTerm instead of a SpinOperator

##### `dump`

Dump the raw data from the `SampleResult` that are stored in `ObserveResult` to the terminal.

##### `expectation`

Overloaded function.

1. `expectation(self) -> float`

Return the expectation value of the `spin_operator` that was provided in `observe()`.

2. `expectation(self, sub_term: cudaq.SpinOperatorTerm) -> float`

3. `expectation(self, sub_term: object) -> float`

Return the expectation value of an individual `sub_term` of the global `spin_operator` that was passed to `observe()`.

**Parameters:**
**sub_term** (`SpinOperatorTerm`) – An individual sub-term of the `spin_operator`.

**Returns:**
The expectation value of the `sub_term` with respect to the `Kernel` that was passed to `observe()`.

**Return type:** float

4. `expectation(self, sub_term: cudaq.SpinOperator) -> float`

Deprecated - ensure to pass a SpinOperatorTerm instead of a SpinOperator

##### `get_spin`

Return the `SpinOperator` corresponding to this `ObserveResult`.

#### `class cudaq.AsyncObserveResult(*args, **kwargs)`

A data-type containing the results of a call to `observe_async()`.

The `AsyncObserveResult` contains a future, whose `ObserveResult` may be returned via an invocation of the `get` method.

This kicks off a wait on the current thread until the results are available.

See future for more information on this programming pattern.

##### `get`

Returns the `ObserveResult` from the asynchronous observe execution.

#### `class cudaq.AsyncStateResult`

A data-type containing the results of a call to `get_state_async()`. The `AsyncStateResult` models a future-like type, whose `State` may be returned via an invocation of the `get` method. This kicks off a wait on the current thread until the results are available. See future for more information on this programming pattern.

##### `get`

Return the `State` from the asynchronous `get_state` accessor execution.

#### `class cudaq.OptimizationResult(*args, **kwargs)`

Result of an optimization: (opt_value, optimal_parameters). optimize() returns a tuple; this type is for type hints and wrapping.

##### `property opt_value`

(self) -> float

##### `property optimal_parameters`

(self) -> list[float]

#### `class cudaq.EvolveResult(*args, **kwargs)`

Stores the execution data from an invocation of `evolve()`.

##### `expectation_values`

Stores the expectation values, that is the results from the calls to `observe()`, at each step in the schedule produced by a call to `evolve()`, including the final expectation values. Each entry corresponds to one observable provided in the `evolve()` call. This property is only populated saving intermediate results was requested in the call to `evolve()`. This value will be None if no intermediate results were requested, or if no observables were specified in the call.

##### `final_expectation_values`

Stores the final expectation values, that is the results produced by calls to `observe()`, triggered by a call to `evolve()`. Each entry corresponds to one observable provided in the `evolve()` call. This value will be None if no observables were specified in the call.

##### `final_state`

Stores the final state produced by a call to `evolve()`. Represent the state of a quantum system after time evolution under a set of operators, see the `evolve()` documentation for more detail. Returns None if no states are available.

##### `intermediate_states`

Stores all intermediate states, meaning the state after each step in a defined schedule, produced by a call to `evolve()`, including the final state. This property is only populated if saving intermediate results was requested in the call to `evolve()`.

#### `class cudaq.AsyncEvolveResult`

Stores the execution data from an invocation of `evolve_async()`.

##### `get`

Retrieve the evolution result from the asynchronous evolve execution .

#### `class cudaq.Resources(*args, **kwargs)`

A data-type containing the results of a call to `estimate_resources()`. This includes all gate counts.

##### `clear`

Clear out all metadata from `self`.

##### `count`

Overloaded function.

1. `count(self, arg: str, /) -> int`

Get the number of occurrences of a given gate with any number of controls

2. `count(self) -> int`

Get the total number of occurrences of all gates

##### `count_controls`

Get the number of occurrences of a given gate with the given number of controls

##### `property depth`

The circuit depth (longest gate chain on any qubit).

##### `depth_for_arity`

Get circuit depth considering only gates of a specific qubit arity. Returns 0 if no gates of that arity exist.

##### `dump`

Print a string of the raw resource counts data to the terminal.

##### `property gate_count_by_arity`

Gate counts by qubit arity, as a dict mapping arity to count.

##### `gate_count_for_arity`

Get gate count for a specific qubit arity (total qubits including controls and targets). Returns 0 if no gates of that arity exist.

##### `property multi_qubit_depth`

Max depth across all gate widths >= 2.

##### `property multi_qubit_gate_count`

Total count of gates with 2 or more qubits.

##### `property num_qubits`

The total number of qubits allocated in the kernel.

##### `property num_used_qubits`

The number of qubits touched by at least one quantum operation.

##### `property per_qubit_depth`

Per-qubit circuit depth (all gates), as a dict mapping qubit index to depth.

##### `to_dict`

Return a dictionary of the raw resource counts that are stored in `self`.

#### Optimizers

The following methods are available on every optimizer instance (`GradientDescent`, `COBYLA`, `NelderMead`, `LBFGS`, `Adam`, `SGD`, `SPSA`):

##### `optimize(dimensions:int, function) → tuple[float,list[float]]`

Run the optimization procedure.

**Parameters:**
*   **dimensions** – The number of parameters to optimize

*   **function** – The objective function to minimize

**Returns:**
tuple of (optimal_value, optimal_parameters)

##### `requires_gradients() → bool`

Check whether this optimizer requires gradient information.

**Returns:**
True if gradients required, False otherwise

#### Properties common to all optimizers

Every optimizer class below (`GradientDescent`, `COBYLA`, `NelderMead`, `LBFGS`, `Adam`, `SGD`, `SPSA`) also exposes the following properties with identical semantics:

##### `property initial_parameters`

Initial values for the optimization parameters (optional).

Provides a starting point for the optimization. If not specified, the optimizer typically initializes parameters to zeros. Good initial parameter values can significantly improve convergence speed and help avoid poor local minima. The length must match the problem dimension.

**Example:** `optimizer.initial_parameters = [0.5, -0.3, 1.2]`

**Type:** list[float]

##### `property lower_bounds`

Lower bounds for optimization parameters (optional).

Constrains the search space by specifying minimum allowed values for each parameter. When specified, the length must match the problem dimension.

**Example:** `optimizer.lower_bounds = [-2.0, -2.0]  # For 2D problem`

**Type:** list[float]

##### `property upper_bounds`

Upper bounds for optimization parameters (optional).

Constrains the search space by specifying maximum allowed values for each parameter. When specified, the length must match the problem dimension.

**Example:** `optimizer.upper_bounds = [2.0, 2.0]  # For 2D problem`

**Type:** list[float]

##### `property max_iterations`

Maximum number of optimizer iterations (default: unlimited).

Sets an upper bound on the number of function evaluations or iterations the optimizer will perform. If not set, the optimizer may run until convergence or until another stopping criterion is met.

**Type:** int

#### `class cudaq.optimizers.GradientDescent(*args, **kwargs)`

#### `class cudaq.optimizers.COBYLA(*args, **kwargs)`

#### `class cudaq.optimizers.NelderMead(*args, **kwargs)`

#### `class cudaq.optimizers.LBFGS(*args, **kwargs)`

#### `class cudaq.optimizers.Adam(*args, **kwargs)`

##### `property batch_size`

Number of samples per batch (default: 1).

For stochastic optimization, determines how many samples are used to compute each gradient estimate. Batch size of 1 corresponds to online learning. Larger batch sizes can provide more stable gradient estimates but require more computation per iteration.

**Type:** int

##### `property beta1`

Exponential decay rate for the first moment estimates (default: 0.9).

Controls the exponential moving average of past gradients (momentum term). Values are typically in the range [0.9, 0.999]. Higher values give more weight to past gradients, providing smoother updates but slower adaptation.

**Type:** float

##### `property beta2`

Exponential decay rate for the second moment estimates (default: 0.999).

Controls the exponential moving average of past squared gradients. Values are typically in the range [0.99, 0.9999]. Higher values provide more stable learning rates but slower adaptation to changing gradient magnitudes.

**Type:** float

##### `property epsilon`

Small constant for numerical stability (default: 1e-8).

Added to the denominator to prevent division by zero when computing adaptive learning rates. Should be a small positive value, typically between 1e-8 and 1e-6.

**Type:** float

##### `property f_tol`

Convergence tolerance on the objective function value (default: 1e-4).

Optimization terminates when the change in objective function value between iterations falls below this threshold. Smaller values lead to tighter convergence but may require more iterations.

**Type:** float

##### `property step_size`

Learning rate (step size) for parameter updates (default: 0.01).

Controls the magnitude of parameter updates at each iteration. Typical values range from 0.001 to 0.1. The effective learning rate is adapted per parameter based on gradient history. Start with 0.001 or 0.01 and adjust based on convergence behavior.

**Type:** float

#### `class cudaq.optimizers.SGD(*args, **kwargs)`

##### `property batch_size`

Number of samples per batch (default: 1).

For stochastic optimization, determines how many samples are used to compute each gradient estimate. Batch size of 1 corresponds to true stochastic gradient descent. Larger batch sizes (mini-batch SGD) can provide more stable gradient estimates but require more computation per iteration.

**Type:** int

##### `property f_tol`

Convergence tolerance on the objective function value (default: 1e-4).

Optimization terminates when the change in objective function value between iterations falls below this threshold. Smaller values lead to tighter convergence but may require more iterations. Note that with stochastic gradients, convergence may be noisy.

**Type:** float

##### `property step_size`

Learning rate (step size) for parameter updates (default: 0.01).

Controls the magnitude of parameter updates at each iteration. The update rule is: x_new = x_old - step_size * gradient. Typical values range from 0.001 to 0.1. Too large values can cause divergence, while too small values lead to slow convergence.

**Type:** float

#### `class cudaq.optimizers.SPSA(*args, **kwargs)`

##### `property gamma`

Scaling exponent for the step size schedule (default: 0.101).

Controls how the step size decreases over iterations. The step size at iteration k is proportional to (A + k + 1)^(-gamma), where A is a stability constant. Common values are in the range [0.1, 0.6].

**Type:** float

##### `property step_size`

Evaluation step size for gradient approximation (default: 0.3).

Controls the magnitude of perturbations used to approximate gradients. Larger values provide coarser gradient estimates but may be more robust to noise. Typical values range from 0.1 to 0.5.

**Type:** float

#### Gradients

#### `class cudaq.gradients.gradient`

#### `class cudaq.gradients.CentralDifference(*args, **kwargs)`

##### `compute`

Compute the gradient of the provided `parameter_vector` with respect to its loss function, using the `CentralDifference` method.

#### `class cudaq.gradients.ForwardDifference(*args, **kwargs)`

##### `compute`

Compute the gradient of the provided `parameter_vector` with respect to its loss function, using the `ForwardDifference` method.

#### `class cudaq.gradients.ParameterShift(*args, **kwargs)`

##### `compute`

Compute the gradient of the provided `parameter_vector` with respect to its loss function, using the `ParameterShift` method.

#### Noisy Simulation

#### `class cudaq.NoiseModel(*args, **kwargs)`

The `NoiseModel` defines a set of `KrausChannel`’s applied to specific qubits after the invocation of specified quantum operations.

##### `__init__`

Construct a noise model with all built-in channels pre-registered.

##### `add_all_qubit_channel`

Add the given `KrausChannel` to be applied after invocation of the specified quantum operation on arbitrary qubits.

**Parameters:**
*   **operator** (str) – The quantum operator to apply the noise channel to.

*   **channel** (cudaq.KrausChannel) – The `KrausChannel` to apply to the specified `operator` on any arbitrary qubits.

*   **num_controls** – Number of control bits. Default is 0 (no control bits).

##### `add_channel`

Overloaded function.

1. `add_channel(self, operator: str, qubits: collections.abc.Sequence[int], channel: cudaq.KrausChannel) -> None`

Add the given `KrausChannel` to be applied after invocation of the specified quantum operation.

**Parameters:**
*   **operator** (str) – The quantum operator to apply the noise channel to.

*   **qubits** (List[int]) – The qubit/s to apply the noise channel to.

*   **channel** (cudaq.KrausChannel) – The `KrausChannel` to apply to the specified `operator` on the specified `qubits`.

2. `add_channel(self, operator: str, pre: collections.abc.Callable[[collections.abc.Sequence[int], collections.abc.Sequence[float]], cudaq.KrausChannel]) -> None`

Add the given `KrausChannel` generator callback to be applied after invocation of the specified quantum operation.

**Parameters:**
*   **operator** (str) – The quantum operator to apply the noise channel to.

*   **pre** (Callable) – The callback which takes qubits operands and gate parameters and returns a concrete `KrausChannel` to apply to the specified `operator`.

##### `get_channels`

Return the `KrausChannel`’s that make up this noise model.

#### `class cudaq.BitFlipChannel(*args, **kwargs)`

Models the decoherence of the qubit state. Its constructor expects a float value, `probability`, representing the probability that the qubit flips from the 1-state to the 0-state, or vice versa. E.g, the probability of a random X-180 rotation being applied to the qubit.

The Kraus Channels are thereby defined to be:

K_0 = sqrt(1 - probability) * I

K_1 = sqrt(probability ) * X

The probability of the qubit remaining in the same state is therefore `1 - probability`.

##### `__init__`

Overloaded function.

1. `__init__(self, arg: collections.abc.Sequence[float], /) -> None`

2. `__init__(self, probability: float) -> None`

Initialize the `BitFlipChannel` with the provided `probability`.

#### `class cudaq.PhaseFlipChannel(*args, **kwargs)`

Models the decoherence of the qubit phase. Its constructor expects a float value, `probability`, representing the probability of a random Z-180 rotation being applied to the qubit.

The Kraus Channels are thereby defined to be:

K_0 = sqrt(1 - probability) * I

K_1 = sqrt(probability ) * Z

The probability of the qubit phase remaining untouched is therefore `1 - probability`.

##### `__init__`

Overloaded function.

1. `__init__(self, arg: collections.abc.Sequence[float], /) -> None`

2. `__init__(self, probability: float) -> None`

Initialize the `PhaseFlipChannel` with the provided `probability`.

#### `class cudaq.DepolarizationChannel(*args, **kwargs)`

Models the decoherence of the qubit state and phase into a mixture ” of the computational basis states, `|0>` and `|1>`.

The Kraus Channels are thereby defined to be:

K_0 = sqrt(1 - probability) * I

K_1 = sqrt(probability / 3) * X

K_2 = sqrt(probability / 3) * Y

K_3 = sqrt(probability / 3) * Z

where I, X, Y, Z are the 2x2 Pauli matrices.

The constructor expects a float value, `probability`, representing the probability the state decay will occur. The qubit will remain untouched, therefore, with a probability of `1 - probability`. And the X,Y,Z operators will be applied with a probability of `probability / 3`.

For `probability = 0.0`, the channel will behave noise-free. For `probability = 0.75`, the channel will fully depolarize the state. For `probability = 1.0`, the channel will be uniform.

##### `__init__`

Overloaded function.

1. `__init__(self, arg: collections.abc.Sequence[float], /) -> None`

2. `__init__(self, probability: float) -> None`

Initialize the `DepolarizationChannel` with the provided `probability`.

#### `class cudaq.AmplitudeDampingChannel(*args, **kwargs)`

Models the dissipation of energy due to system interactions with the environment.

The Kraus Channels are thereby defined to be:

K_0 = diag(1, sqrt(1 - probability))

K_1 = [[0, sqrt(probability)], [0, 0]]

Its constructor expects a float value, `probability`, representing the probability that the qubit will decay to its ground state. The probability of the qubit remaining in the same state is therefore `1 - probability`.

##### `__init__`

Overloaded function.

1. `__init__(self, arg: collections.abc.Sequence[float], /) -> None`

2. `__init__(self, probability: float) -> None`

Initialize the `AmplitudeDampingChannel` with the provided `probability`.

#### `class cudaq.PhaseDamping(*args, **kwargs)`

A Kraus channel that models the single-qubit phase damping error. This is similar to AmplitudeDamping, but for phase.

#### `class cudaq.XError(*args, **kwargs)`

A Pauli error that applies the X operator when an error occurs. It is the same as BitFlipChannel.

#### `class cudaq.YError(*args, **kwargs)`

A Pauli error that applies the Y operator when an error occurs.

#### `class cudaq.ZError(*args, **kwargs)`

A Pauli error that applies the Z operator when an error occurs. It is the same as PhaseFlipChannel.

#### `class cudaq.Pauli1(*args, **kwargs)`

A single-qubit Pauli error that applies either an X error, Y error, or Z error. The probability of each X, Y, or Z error is supplied as a parameter.

#### `class cudaq.Pauli2(*args, **kwargs)`

A 2-qubit Pauli error that applies one of the following errors, with the probabilities specified as a vector. Possible errors: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, and ZZ.

#### `class cudaq.Depolarization1(*args, **kwargs)`

The same as DepolarizationChannel (single qubit depolarization)

#### `class cudaq.Depolarization2(*args, **kwargs)`

A 2-qubit depolarization error that applies one of the following errors. Possible errors: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, and ZZ.

#### `class cudaq.KrausChannel(*args, **kwargs)`

The `KrausChannel` is composed of a list of `KrausOperator`’s and is applied to a specific qubit or set of qubits.

##### `__getitem__`

Return the `KrausOperator` at the given index in this `KrausChannel`.

##### `append`

Add a `KrausOperator` to this `KrausChannel`.

##### `get_ops`

Return the `KrausOperator`’s in this `KrausChannel`.

##### `property noise_type`

(self) -> cudaq.NoiseModelType

##### `property parameters`

(self) -> list[float]

#### `class cudaq.KrausOperator(*args, **kwargs)`

The `KrausOperator` is represented by a matrix and serves as an element of a quantum channel such that `Σ K_i† K_i = I`.

##### `property col_count`

The number of columns in the matrix representation of this `KrausOperator`.

##### `property row_count`

The number of rows in the matrix representation of this `KrausOperator`.

### MPI Submodule

#### `cudaq.mpi.initialize() → None`

Initialize MPI if available.

#### `cudaq.mpi.rank() → int`

Return the rank of this process.

#### `cudaq.mpi.num_ranks() → int`

Return the total number of ranks.

#### `cudaq.mpi.all_gather(arg0:int, arg1:collections.abc.Sequence[float], /) → list[float]`

#### `cudaq.mpi.all_gather(arg0:int, arg1:collections.abc.Sequence[int], /) → list[int]`

Gather and scatter the `local` list (floats or integers, per the overloads above), returning a concatenation of all lists across all ranks. The total global list size must be provided.

#### `cudaq.mpi.broadcast(arg0:collections.abc.Sequence[float], arg1:int, arg2:int, /) → list[float]`

Broadcast an array from a process (rootRank) to all other processes. The size of broadcast array must be provided.

#### `cudaq.mpi.is_initialized() → bool`

Returns true if MPI has already been initialized.

#### `cudaq.mpi.split_communicator(color:int, key:int|None=None) → int`

Splits the current communicator into sub-communicators based on the input color and key.

Ranks that pass the same color are placed in the same new communicator. The key controls the rank ordering within that new communicator.

**Parameters:**
*   **color** (int) – Split color. Ranks with the same color join the same communicator.

*   **key** (Optional[int]) – Rank-ordering key within the new communicator. Defaults to `None`, which uses the current rank in the original communicator as the split key.

**Returns:**
Integer representation of the new communicator pointer (`comm_ptr`).

**Return type:** int

Example:

```python
import cudaq

cudaq.mpi.initialize()
cudaq.set_target("tensornet")

world_rank = cudaq.mpi.rank()
world_size = cudaq.mpi.num_ranks()

# Split the world communicator into QPU groups of two ranks each.
# With four ranks, ranks 0 and 1 use color 0, while ranks 2 and 3 use color 1.
ranks_per_qpu = 2
if world_size % ranks_per_qpu != 0:
    raise RuntimeError("World size must be a multiple of ranks_per_qpu.")

qpu_id = world_rank // ranks_per_qpu
qpu_comm = cudaq.mpi.split_communicator(color=qpu_id)

cudaq.mpi.set_communicator(qpu_comm)
```

#### `cudaq.mpi.set_communicator(commPtr:int) → None`

Sets the communicator of the backend simulator based on the input communicator address (as an integer). MPI must be initialized. If the selected target does not support MPI-based distributed simulation, CUDA-Q emits a warning and ignores this call.

**Parameters:**
**commPtr** (int) – Integer representation of the communicator pointer (`comm_ptr`) for the backend simulator to use. This can be returned by `cudaq.mpi.split_communicator` or by taking the address of a live `mpi4py` communicator with `MPI._addressof(comm)`.

Examples:

Using `cudaq.mpi.split_communicator`:

```python
import cudaq

cudaq.mpi.initialize()
cudaq.set_target("tensornet")

world_rank = cudaq.mpi.rank()
ranks_per_qpu = 2
qpu_id = world_rank // ranks_per_qpu
qpu_comm = cudaq.mpi.split_communicator(qpu_id)

cudaq.mpi.set_communicator(qpu_comm)
```

Using `mpi4py`:

```python
import cudaq
from mpi4py import MPI

cudaq.set_target("tensornet")

world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
ranks_per_qpu = 2
qpu_id = world_rank // ranks_per_qpu
qpu_comm = world_comm.Split(color=qpu_id, key=world_rank)

cudaq.mpi.set_communicator(MPI._addressof(qpu_comm))
```

When using `mpi4py`, keep the communicator object alive while CUDA-Q uses it.

#### `cudaq.mpi.finalize() → None`

Finalize MPI.

### PTSBE Submodule

The `cudaq.ptsbe` submodule implements Pre-Trajectory Sampling with Batch Execution (PTSBE). For a conceptual overview and usage tutorial see Pre-Trajectory Sampling with Batch Execution (PTSBE).

#### Sampling Functions

#### `cudaq.ptsbe.sample(kernel, *args, shots_count=1000, noise_model=None, max_trajectories=None, sampling_strategy=None, shot_allocation=None, return_execution_data=False, include_sequential_data=False)`

Sample using Pre-Trajectory Sampling with Batch Execution (`PTSBE`).

Pre-samples noise realizations (trajectories) and batches circuit executions by unique noise configuration, enabling efficient noisy sampling of many shots.

When called with list arguments (broadcast mode), executes the kernel for each set of arguments and returns a list of results.

**Parameters:**
*   **kernel** – The quantum kernel to execute.

*   **shots_count** (int) – Number of measurement shots. Defaults to 1000.

*   **noise_model** – Optional noise model for gate-based noise. Noise can also be specified inside the kernel via `cudaq.apply_noise()`; both can be used together.

*   **max_trajectories** (int or `None`) – Maximum unique trajectories to generate. `None` means use the number of shots. Note for large shot counts setting a maximum is recommended to get the benefits of PTS.

*   **sampling_strategy** (`PTSSamplingStrategy` or `None`) – Strategy for trajectory generation. `None` uses the default probabilistic sampling strategy.

*   **shot_allocation** (`ShotAllocationStrategy` or `None`) – Strategy for allocating shots across trajectories. `None` uses the default proportional (weight-based) allocation.

*   **return_execution_data** (bool) – Include circuit structure, trajectory specifications, and per-trajectory measurement outcomes in the returned result. Defaults to `False`.

*   **include_sequential_data** (bool) – Populate per-shot sequential bitstring data on the result. Defaults to `False`.

**Returns:** Measurement results. Returns a list of results
in broadcast mode.

**Return type:** `SampleResult`

**Raises:**
**RuntimeError** – If the kernel is invalid or arguments are invalid.

#### `cudaq.ptsbe.sample_async(kernel, *args, shots_count=1000, noise_model=None, max_trajectories=None, sampling_strategy=None, shot_allocation=None, return_execution_data=False, include_sequential_data=False)`

Asynchronously sample using PTSBE. Returns a future whose result can be retrieved via `.get()`.

**Parameters:**
*   **kernel** – The quantum kernel to execute.

*   **shots_count** (int) – Number of measurement shots. Defaults to 1000.

*   **noise_model** – Optional noise model for gate-based noise; noise can also be specified in the kernel via `cudaq.apply_noise()`.

*   **max_trajectories** (int or `None`) – Maximum unique trajectories.

*   **sampling_strategy** (`PTSSamplingStrategy` or `None`) – Strategy for trajectory generation.

*   **shot_allocation** (`ShotAllocationStrategy` or `None`) – Strategy for allocating shots across trajectories.

*   **return_execution_data** (bool) – Include execution data in the result.

*   **include_sequential_data** (bool) – Populate per-shot sequential data.

**Returns:** A future whose `.get()` returns the
`SampleResult`.

**Return type:** `AsyncPTSBESampleResult`

**Raises:**
**RuntimeError** – If the kernel is invalid or arguments are invalid.

* * *

#### Result Type

#### `class cudaq.ptsbe.PTSBESampleResult`

PTSBE sample result with optional execution data.

##### `has_execution_data`

Check if execution data is available.

##### `property ptsbe_execution_data`

PTSBE execution data if return_execution_data was True, None otherwise.

* * *

#### Trajectory Sampling Strategies

#### `class cudaq.ptsbe.PTSSamplingStrategy`

Base class for trajectory sampling strategies.

##### `name`

Get the name of this strategy.

#### `class cudaq.ptsbe.ProbabilisticSamplingStrategy(*args, **kwargs)`

Sample trajectories randomly based on their occurrence probabilities.

#### `class cudaq.ptsbe.OrderedSamplingStrategy(*args, **kwargs)`

Sample trajectories sorted by probability in descending order.

#### `class cudaq.ptsbe.ExhaustiveSamplingStrategy(*args, **kwargs)`

Enumerate all possible trajectories in lexicographic order.

* * *

#### Shot Allocation Strategy

#### `class cudaq.ptsbe.ShotAllocationStrategy(*args, **kwargs)`

Strategy for allocating shots across selected trajectories.

##### `property bias_strength`

Bias factor for weighted strategies. Default value is 2.0.

##### `property type`

The allocation strategy type.

#### `class cudaq.ptsbe.ShotAllocationType(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)`

Strategy type for allocating shots across trajectories.

##### `HIGH_WEIGHT_BIAS = 3`

##### `LOW_WEIGHT_BIAS = 2`

##### `PROPORTIONAL = 0`

##### `UNIFORM = 1`

* * *

#### Execution Data

#### `class cudaq.ptsbe.PTSBEExecutionData`

Container for PTSBE execution data including circuit structure, trajectory specifications, and per-trajectory measurement outcomes.

##### `count_instructions`

Count instructions of a given type.

##### `get_trajectory`

Look up a trajectory by its ID. Returns None if not found.

##### `property instructions`

(self) -> list[cudaq.ptsbe.TraceInstruction]

##### `property trajectories`

(self) -> list[cudaq.ptsbe.KrausTrajectory]

#### `class cudaq.ptsbe.TraceInstruction`

Single operation in the execution trace.

##### `property channel`

(self) -> object

##### `property controls`

(self) -> list[int]

##### `property name`

(self) -> str

##### `property params`

(self) -> list[float]

##### `property targets`

(self) -> list[int]

##### `property type`

(self) -> cudaq.ptsbe.TraceInstructionType

#### `class cudaq.ptsbe.TraceInstructionType(value, names=<not given>, *values, module=None, qualname=None, type=None, start=1, boundary=None)`

Type discriminator for trace instructions.

* * *

#### Trajectory and Selection Types

#### `class cudaq.ptsbe.KrausTrajectory`

Complete specification of one noise trajectory with outcomes.

##### `property kraus_selections`

(self) -> list[cudaq.ptsbe.KrausSelection]

##### `property measurement_counts`

(self) -> dict[str, int]

##### `property multiplicity`

Number of times this trajectory was sampled.

##### `property num_shots`

(self) -> int

##### `property probability`

(self) -> float

##### `property trajectory_id`

(self) -> int

##### `property weight`

Allocation weight for shot distribution.

#### `class cudaq.ptsbe.KrausSelection`

Reference to a single Kraus operator selection.

##### `property circuit_location`

(self) -> int

##### `property is_error`

(self) -> bool

##### `property kraus_operator_index`

(self) -> int

##### `property op_name`

(self) -> str

##### `property qubits`

(self) -> list[int]
