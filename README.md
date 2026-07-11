# Sketch-Prompted VLA

Circle-and-arrow visual instructions for disambiguated manipulation, built on
[UniVLA](https://github.com/OpenDriveLab/UniVLA). Full proposal:
[`sketch_prompted_vla_proposal.md`](sketch_prompted_vla_proposal.md).

## Status

Section 2 of the proposal (automatic dataset generation from LIBERO, no human
labeling for training) has a working proof of concept. See
[`report/section2_report.tex`](report/section2_report.tex) (or the compiled
[`report/section2_report.pdf`](report/section2_report.pdf)) for what was
built, what was verified, and what's left open (Section 2.6, negative and
distractor mining, is the current known gap).

## Layout

- `scripts/` — dataset generation and verification pipeline. Entry point is
  `scripts/generate_dataset_wsl.py`; the `*_wsl.py` scripts must be run
  inside WSL2 with a conda environment that has `robosuite` / `mujoco` /
  `libero` installed (they replay real LIBERO demonstrations through the
  actual simulator for ground truth).
- `outputs/` — generated annotations and verification figures.
  `outputs/dataset_poc/` holds the current proof-of-concept dataset
  (`dataset_metadata.json` plus rendered overlay images).
- `data/` — LIBERO demonstration HDF5s. Not tracked in git (see
  `.gitignore`); regenerate with `python scripts/download_data.py`.
- `UniVLA/` — the baseline latent-action VLA codebase, tracked as a git
  submodule pinned to the commit this project builds on. After cloning this
  repo, run `git submodule update --init` to pull it in.
- `report/` — write-ups tracking progress against the proposal, section by
  section.

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
cd sketch_vla
python scripts/download_data.py   # pulls LIBERO-Spatial HDF5s into data/
```

The dataset generation scripts (`scripts/*_wsl.py`) are written to run in
WSL2 against a real robosuite/MuJoCo/LIBERO environment, since they replay
demonstrations in the simulator to extract exact camera and object ground
truth rather than approximating it.
