"""Static acceptance checks for the notebooks shipped in the course archive."""

from __future__ import annotations

import json
import sys
from pathlib import Path


COURSE_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/cudaq-agentic-coding")
NOTEBOOKS = (
    "_intro_cudaq.ipynb",
    "_intro_Ising_Calibration.ipynb",
    "00_notebook.ipynb",
)


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def validate_notebook(relative_path: str) -> None:
    path = COURSE_ROOT / relative_path
    with path.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    kernel_name = notebook.get("metadata", {}).get("kernelspec", {}).get("name")
    if kernel_name != "python3":
        raise AssertionError(f"{relative_path}: expected python3 kernel, found {kernel_name!r}")

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if not code_cells:
        raise AssertionError(f"{relative_path}: no code cells")

    error_outputs = []
    for index, cell in enumerate(code_cells, start=1):
        compile(source_text(cell), f"{relative_path}:code-cell-{index}", "exec")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error_outputs.append((index, output.get("ename"), output.get("evalue")))

    if error_outputs:
        raise AssertionError(f"{relative_path}: saved error output(s): {error_outputs}")

    unexecuted = {
        index
        for index, cell in enumerate(code_cells, start=1)
        if cell.get("execution_count") is None
    }
    allowed_unexecuted = {1} if relative_path == "_intro_Ising_Calibration.ipynb" else set()
    unexpected_unexecuted = unexecuted - allowed_unexecuted
    if unexpected_unexecuted:
        raise AssertionError(
            f"{relative_path}: unexpected unexecuted code cells {sorted(unexpected_unexecuted)}"
        )

    print(
        f"PASS {relative_path}: kernel=python3 code_cells={len(code_cells)} "
        f"errors=0 allowed_unexecuted={sorted(unexecuted)}"
    )


if __name__ == "__main__":
    for notebook_path in NOTEBOOKS:
        validate_notebook(notebook_path)
