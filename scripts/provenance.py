"""Version identity for anything that produces a number (eval repo).

Twin of `SketchPromptVLA-Pi:src/sketchvla/provenance.py`. Kept as a copy rather
than a shared package because the two repos are cloned independently and the
LIBERO scripts run under a different interpreter from the trainer.

Blocker A of `docs/SESSION_2026-09-01.md`: no result file in this project records
which code produced it. Pods cannot `git fetch` (the https remote prompts for
credentials), so code reaches them as base64 over the tty and the working tree
drifts from every commit; `/workspace/harness_repo` is not a git repository at
all. "The code produces two different results" is not mysterious under those
conditions -- neither version is pinned at the point a number is generated.

This module is the pin. `stamp()` describes a tree in a way that survives all
three failure modes:

  * a clean checkout                -> commit sha, branch
  * a hand-shipped, dirty tree      -> sha plus `tree_digest`, a blake2b over
                                       `git diff HEAD` and the contents of every
                                       untracked non-ignored file, so two trees
                                       at the same sha with different edits get
                                       different identities
  * a bundle that is not a repo     -> the `VERSION` file the bundle carries

Nothing here raises. A probe that cannot describe itself must still produce its
measurement; it just says `available: false` and the reader knows the number is
unattributable.

Usage -- one line at the point of writing, not at the point of computing:

    import provenance
    provenance.write_json(out_path, result)

Runs anywhere: stdlib only, and written to import under the py3.8 LIBERO client
venv as well as the pod's uv environment.

`write_json` adds a `_provenance` block naming every repo it can find: this one,
plus any tree named in `SKETCHVLA_PROVENANCE_ROOTS` (colon-separated), which is
how a run stamps the eval repo and the harness bundle alongside the trainer.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

ROOTS_ENV = "SKETCHVLA_PROVENANCE_ROOTS"
VERSION_FILE = "VERSION"
_DIGEST_BYTES = 8
_GIT_TIMEOUT_S = 60


def _git(root: pathlib.Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", "replace").strip()


def _untracked(root: pathlib.Path) -> list[str]:
    listing = _git(root, "ls-files", "--others", "--exclude-standard")
    return sorted(p for p in (listing or "").splitlines() if p)


def _modified(root: pathlib.Path) -> list[str]:
    listing = _git(root, "diff", "--name-only", "HEAD")
    return sorted(p for p in (listing or "").splitlines() if p)


def _tree_digest(root: pathlib.Path, paths: list[str]) -> str:
    """Identity for a tree that does not match its own commit.

    Hashes the *contents* of every modified and untracked file, so the
    base64-over-tty shipping path produces a stable, comparable identifier
    instead of a bare "dirty" flag that two different sessions both print
    identically. Content rather than `git diff` output on purpose: generating a
    diff over a repo carrying `outputs/` takes tens of seconds and renormalises
    line endings, and neither cost buys anything the file bytes do not.
    """
    h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    for rel in paths:
        path = root / rel
        h.update(rel.encode("utf-8", "replace"))
        try:
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()


def stamp(root: str | os.PathLike | None = None) -> dict:
    """Describe one tree. Never raises."""
    if root is None:
        root = pathlib.Path(__file__).resolve().parents[1]
    root = pathlib.Path(root).resolve()

    record: dict = {"root": str(root), "available": False}
    if not root.exists():
        record["reason"] = "path does not exist"
        return record

    version_file = root / VERSION_FILE
    if version_file.is_file():
        try:
            record["version_file"] = version_file.read_text().strip()
        except OSError:
            pass

    sha = _git(root, "rev-parse", "HEAD")
    if sha is None:
        record["reason"] = "not a git repository" if "version_file" not in record else "bundle"
        record["available"] = "version_file" in record
        return record

    untracked = _untracked(root)
    modified = _modified(root)
    record.update(
        available=True,
        sha=sha,
        branch=_git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        commit_time=_git(root, "show", "-s", "--format=%cI", "HEAD"),
        dirty=bool(untracked or modified),
        modified=modified,
        untracked=untracked,
    )
    if record["dirty"]:
        record["tree_digest"] = _tree_digest(root, sorted(set(modified) | set(untracked)))
    stashes = _git(root, "stash", "list")
    if stashes:
        record["stashes"] = stashes.splitlines()
    return record


def _extra_roots() -> list[pathlib.Path]:
    raw = os.environ.get(ROOTS_ENV, "")
    return [pathlib.Path(p) for p in raw.split(os.pathsep) if p.strip()]


def collect(*roots: str | os.PathLike) -> dict:
    """Stamp this tree, every root in `roots`, and every root in the env var."""
    seen: dict[str, dict] = {}
    for root in [None, *roots, *_extra_roots()]:
        record = stamp(root)
        seen.setdefault(pathlib.Path(record["root"]).name, record)
    return {
        "repos": seen,
        "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
    }


def write_json(path: str | os.PathLike, payload: dict, *roots: str | os.PathLike, indent: int = 2) -> None:
    """Write `payload` with a `_provenance` block. The only sanctioned way to
    emit a result file from this repo."""
    body = dict(payload)
    body["_provenance"] = collect(*roots)
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=indent))


def summary_line(*roots: str | os.PathLike) -> str:
    """One line for a log header, so a run's identity is greppable in stdout."""
    parts = []
    for name, record in collect(*roots)["repos"].items():
        if not record.get("available"):
            parts.append(f"{name}=UNKNOWN")
        elif record.get("sha"):
            short = record["sha"][:7]
            suffix = f"+{record['tree_digest']}" if record.get("dirty") else ""
            parts.append(f"{name}={short}{suffix}")
        else:
            parts.append(f"{name}={record.get('version_file', 'bundle')}")
    return "provenance " + " ".join(parts)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Print this tree's version identity.")
    ap.add_argument("roots", nargs="*", help="extra trees to stamp")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.json:
        print(json.dumps(collect(*args.roots), indent=2))
    else:
        print(summary_line(*args.roots))


if __name__ == "__main__":
    main()
