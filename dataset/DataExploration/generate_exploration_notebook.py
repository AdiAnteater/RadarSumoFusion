from __future__ import annotations

import argparse
import json
from pathlib import Path


def _default_template_path(script_path: Path) -> Path:
    return script_path.parent / "sensor_capture_20260325_234141_exploration.ipynb"


def _replace_dataset_path_in_cells(cells: list[dict], dataset_dir: Path) -> bool:
    """
    Replace DATASET_DIR assignment in code cells.
    Returns True if a replacement was performed.
    """
    replacement = f'DATASET_DIR = Path(r"{dataset_dir}")\n'
    replaced = False

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if not isinstance(source, list):
            continue

        for i, line in enumerate(source):
            if isinstance(line, str) and line.strip().startswith("DATASET_DIR = Path("):
                source[i] = replacement
                replaced = True

    return replaced


def _clear_notebook_outputs(cells: list[dict]) -> None:
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def generate_notebook(template_path: Path, dataset_dir: Path, output_path: Path) -> None:
    if not template_path.exists():
        raise FileNotFoundError(f"Template notebook not found: {template_path}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    raw_text = template_path.read_text(encoding="utf-8")
    notebook = json.loads(raw_text)
    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("Unexpected notebook format: 'cells' is missing or invalid.")

    replaced = _replace_dataset_path_in_cells(cells, dataset_dir)
    if not replaced:
        raise ValueError("Could not find a DATASET_DIR assignment in the template notebook.")

    _clear_notebook_outputs(cells)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_template = _default_template_path(script_path)

    parser = argparse.ArgumentParser(
        description="Generate a reusable CARLA exploration notebook for a dataset folder."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the capture folder (must include camera_data.csv, radar_data.csv, camera_frames).",
    )
    parser.add_argument(
        "--template",
        default=str(default_template),
        help=f"Template notebook path (default: {default_template})",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output notebook path. Defaults to <dataset-dir>/<dataset-name>_exploration.ipynb",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    template_path = Path(args.template).expanduser().resolve()

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = dataset_dir / f"{dataset_dir.name}_exploration.ipynb"

    generate_notebook(template_path=template_path, dataset_dir=dataset_dir, output_path=output_path)
    print(f"Generated notebook: {output_path}")
    print("Open it in Cursor/Jupyter and run all cells.")


if __name__ == "__main__":
    main()
