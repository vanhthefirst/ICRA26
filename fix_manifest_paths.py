"""
Run this once after downloading the .pt files from Google Drive.
It rewrites the path entries inside the manifest so they point to
your local download folder instead of the original server paths.

Usage:
    python fix_manifest_paths.py <manifest_path> <pt_folder_path>

Example (Windows):
    python fix_manifest_paths.py "C:\data\libero10_features\all_models_val_libero10_manifest.json" "C:\data\libero10_features"
"""

import json
import sys
from pathlib import Path


def fix_paths(manifest_path: str, pt_folder: str) -> None:
    manifest_path = Path(manifest_path)
    pt_folder = Path(pt_folder)

    with open(manifest_path) as f:
        manifest = json.load(f)

    for part in manifest["parts"]:
        filename = Path(part["path"]).name
        local_path = pt_folder / filename
        if not local_path.exists():
            print(f"WARNING: {local_path} not found — check your pt_folder path")
        part["path"] = str(local_path)

    out_path = manifest_path.parent / "all_models_val_libero10_manifest_local.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved updated manifest → {out_path}")
    print("Use this file when running run_table1.py and run_umap.py")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_manifest_paths.py <manifest_path> <pt_folder_path>")
        sys.exit(1)
    fix_paths(sys.argv[1], sys.argv[2])
