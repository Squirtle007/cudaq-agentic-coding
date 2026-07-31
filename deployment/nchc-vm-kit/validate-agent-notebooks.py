#!/usr/bin/env python3
"""Validate the four notebooks produced during the isolated OpenCode exercise."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


DEFAULT_NOTEBOOKS = [f"{index:02d}_notebook.ipynb" for index in range(1, 5)]


def joined_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def validate(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"cannot read notebook JSON: {exc}"], warnings

    kernel = notebook.get("metadata", {}).get("kernelspec", {}).get("name")
    if kernel != "python3":
        errors.append(f"kernel must be python3 (found {kernel!r})")

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if not 1 <= len(code_cells) <= 6:
        errors.append(f"expected 1-6 code cells (found {len(code_cells)})")

    assert_count = 0
    all_code: list[str] = []
    for number, cell in enumerate(code_cells, 1):
        source = joined_source(cell)
        all_code.append(source)
        try:
            tree = ast.parse(source)
            assert_count += sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
        except SyntaxError as exc:
            errors.append(f"code cell {number} does not compile: {exc}")
        if cell.get("execution_count") is None:
            errors.append(f"code cell {number} was not executed")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    f"code cell {number} contains {output.get('ename', 'an error')}: "
                    f"{output.get('evalue', '')}"
                )

    if assert_count != 1:
        errors.append(f"expected exactly one Python assert statement (found {assert_count})")

    code = "\n".join(all_code)
    compact = re.sub(r"\s+", "", code)
    common = ["importcudaq", "cudaq.set_target"]
    for token in common:
        if token not in compact:
            errors.append(f"missing required code pattern: {token}")
    if 'option="fp32"' not in compact and "option='fp32'" not in compact:
        errors.append("missing required code pattern: option=fp32")

    name = path.name
    required_by_step = {
        "01_notebook.ipynb": ["NelderMead", "20000", "50", "rx(", "x.ctrl("],
        "02_notebook.ipynb": ["tensornet", "2000", "NelderMead", "0.1", "xy_kernel"],
        "03_notebook.ipynb": [
            "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE",
            "CUDAQ_TENSORNET_NUM_HYPER_SAMPLES",
            "2000",
        ],
        "04_notebook.ipynb": [
            "CUDAQ_MPS_MAX_BOND",
            "CUDAQ_MPS_ABS_CUTOFF",
            "tensornet-mps",
            "2000",
        ],
    }
    for token in required_by_step.get(name, []):
        if token not in code:
            errors.append(f"missing Step-specific pattern: {token}")

    if name == "01_notebook.ipynb":
        env_pos = code.find("CUDAQ_FUSION_MAX_QUBITS")
        import_pos = code.find("\nimport cudaq")
        if env_pos < 0 or import_pos < 0 or env_pos > import_pos:
            errors.append("CUDAQ_FUSION_MAX_QUBITS must be set before importing cudaq")

    if name in {"03_notebook.ipynb", "04_notebook.ipynb"}:
        first_code = all_code[0] if all_code else ""
        variable = (
            "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE"
            if name == "03_notebook.ipynb"
            else "CUDAQ_MPS_MAX_BOND"
        )
        env_pos = first_code.find(variable)
        import_pos = first_code.find("\nimport cudaq")
        if env_pos < 0 or import_pos < 0 or env_pos > import_pos:
            errors.append("required environment variables must be set before importing cudaq")

    if name == "04_notebook.ipynb":
        markdown = "\n".join(
            joined_source(cell)
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        ).lower()
        if "untrustworthy" not in markdown:
            errors.append("Step 4 must state that the MPS energy estimate is untrustworthy")

    if name == "02_notebook.ipynb" and re.search(r"\benergy\s*\(\s*\[", code):
        warnings.append(
            "a direct energy([...]) call was found; confirm the objective was invoked only "
            "by the five-iteration optimizer"
        )

    return errors, warnings


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} OUTPUT_DIR [NOTEBOOK ...]", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).expanduser().resolve()
    names = sys.argv[2:] or DEFAULT_NOTEBOOKS
    failed = False
    for name in names:
        path = root / name
        if not path.is_file():
            print(f"FAIL {name}: file not found")
            failed = True
            continue
        errors, warnings = validate(path)
        for warning in warnings:
            print(f"WARN {name}: {warning}")
        if errors:
            failed = True
            for error in errors:
                print(f"FAIL {name}: {error}")
        else:
            print(f"PASS {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
