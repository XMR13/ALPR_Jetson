"""CVAT → COCO converter (lightweight)

This tool prepares a COCO-style annotations file for detector training.

Supported inputs:
- A path to an existing COCO JSON file (e.g., a CVAT COCO export).
- A directory containing a likely COCO JSON (e.g., ``instances*.json``).

Notes
- This does not convert CVAT XML. Prefer exporting COCO from CVAT.
- The script validates minimal COCO structure and normalizes output
  to a single ``coco.json`` in the chosen output directory.

Usage
    python tools/coco_from_cvat.py --input <file_or_dir> --outdir <out>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _find_coco_json(root: Path) -> Path | None:
    candidates: List[str] = [
        "instances_default.json",
        "instances.json",
        "coco.json",
    ]
    # Also scan for any instances*.json file to be flexible.
    for p in [root / c for c in candidates]:
        if p.is_file():
            return p
    for p in root.rglob("*.json"):
        if p.name.startswith("instances"):
            return p
    return None


def _validate_min_coco(data: Dict[str, Any]) -> None:
    # Minimal COCO keys; we do not enforce full schema here.
    required = ["images", "annotations", "categories"]
    for k in required:
        if k not in data:
            raise ValueError(f"Missing key '{k}' in COCO JSON")
    if not isinstance(data["images"], list):
        raise ValueError("'images' must be a list")
    if not isinstance(data["annotations"], list):
        raise ValueError("'annotations' must be a list")
    if not isinstance(data["categories"], list):
        raise ValueError("'categories' must be a list")


def run(input_path: Path, outdir: Path) -> Path:
    if input_path.is_dir():
        src = _find_coco_json(input_path)
        if src is None:
            raise FileNotFoundError(
                f"No COCO JSON found under directory: {input_path}"
            )
    else:
        src = input_path
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _validate_min_coco(data)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "coco.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize CVAT COCO export to coco.json")
    p.add_argument("--input", required=True, help="Path to COCO JSON or export directory")
    p.add_argument("--outdir", required=True, help="Output directory for coco.json")
    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    src = Path(args.input)
    outdir = Path(args.outdir)
    try:
        out = run(src, outdir)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
