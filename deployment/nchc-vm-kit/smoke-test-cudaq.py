"""Small CPU/GPU CUDA-Q acceptance test for the NCHC course image."""

import cudaq


@cudaq.kernel
def ghz(qubit_count: int):
    qubits = cudaq.qvector(qubit_count)
    h(qubits[0])
    for index in range(qubit_count - 1):
        x.ctrl(qubits[index], qubits[index + 1])
    mz(qubits)


def assert_ghz_counts(counts, qubit_count: int) -> None:
    allowed = {"0" * qubit_count, "1" * qubit_count}
    observed = set(counts)
    unexpected = observed - allowed
    if unexpected:
        raise AssertionError(f"Unexpected GHZ measurement(s): {sorted(unexpected)}")


def run_target(target: str, qubit_count: int) -> None:
    cudaq.set_target(target)
    counts = cudaq.sample(ghz, qubit_count, shots_count=1000)
    assert_ghz_counts(counts, qubit_count)
    print(f"PASS target={target} qubits={qubit_count} counts={counts}")


if __name__ == "__main__":
    cudaq.set_random_seed(1234)
    run_target("qpp-cpu", 8)
    run_target("nvidia", 20)
