#!/usr/bin/env python
"""Capsule entry point: run the one-session analysis across every session.

Invoked by ``code/run`` when you press "Reproducible Run". For each session
found in the attached datacube it executes the analysis notebook(s) headlessly
and writes the executed notebook plus an HTML render to
``/results/<session_id>/``.

The notebooks read their two inputs from the environment, so nothing inside
them has to change between a local run and a capsule run:

    DATACUBE_ROOT   where the session folders live
    SESSION_ID      which session to analyse

Optional overrides for this script:

    RESULTS_DIR     output location (default /root/capsule/results)
    SESSION_IDS     comma-separated subset instead of "every session found"
    NOTEBOOKS       comma-separated notebook filenames instead of all of them
    NB_TIMEOUT      per-cell timeout in seconds, -1 for no limit (default -1)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat

CODE_DIR = Path(__file__).resolve().parent
NB_DIR = CODE_DIR / "notebooks"
ZARR_SUFFIX = ".nwb.zarr"

ROOT = Path(os.environ.get("DATACUBE_ROOT", "/root/capsule/data/dynamicrouting_datacube"))
RESULTS = Path(os.environ.get("RESULTS_DIR", "/root/capsule/results"))
TIMEOUT = os.environ.get("NB_TIMEOUT", "-1")


def discover_sessions():
    """Session ids (``<subject>_<date>``) that have an NWB zarr store on disk."""
    if not ROOT.is_dir():
        return []
    found = set()
    for session_dir in ROOT.iterdir():
        if session_dir.is_dir():
            for store in session_dir.glob("*" + ZARR_SUFFIX):
                found.add(store.name[: -len(ZARR_SUFFIX)])
    return sorted(found)


def select_notebooks():
    requested = os.environ.get("NOTEBOOKS", "").strip()
    if requested:
        return [NB_DIR / n.strip() for n in requested.split(",") if n.strip()]
    return sorted(p for p in NB_DIR.glob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def has_error_output(notebook):
    nb = nbformat.read(notebook, as_version=4)
    return any(
        output.get("output_type") == "error"
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


def execute(notebook, session_id, out_dir):
    """Run one notebook for one session. Returns True if no cell errored."""
    out_dir.mkdir(parents=True, exist_ok=True)
    executed = out_dir / notebook.name
    env = dict(os.environ, SESSION_ID=session_id, DATACUBE_ROOT=str(ROOT))

    # --allow-errors so a bad cell still leaves us a notebook to debug from;
    # we detect the failure afterwards by scanning for error outputs.
    completed = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--allow-errors",
            f"--ExecutePreprocessor.timeout={TIMEOUT}",
            "--output", executed.stem,
            "--output-dir", str(out_dir),
            str(notebook),
        ],
        env=env,
    )
    if completed.returncode != 0 or not executed.exists():
        return False

    subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "html",
            "--output", executed.stem,
            "--output-dir", str(out_dir),
            str(executed),
        ],
        env=env,
    )
    return not has_error_output(executed)


def run():
    requested = os.environ.get("SESSION_IDS", "").strip()
    sessions = [s.strip() for s in requested.split(",") if s.strip()] or discover_sessions()
    notebooks = select_notebooks()

    if not sessions:
        print(f"no sessions found under {ROOT}", file=sys.stderr)
        return 1
    missing = [n for n in notebooks if not n.exists()]
    if missing or not notebooks:
        print(f"notebooks not found: {missing or NB_DIR}", file=sys.stderr)
        return 1

    print(f"datacube : {ROOT}")
    print(f"results  : {RESULTS}")
    print(f"{len(notebooks)} notebook(s) x {len(sessions)} session(s)\n", flush=True)

    failures = []
    for session_id in sessions:
        for notebook in notebooks:
            label = f"{session_id} / {notebook.name}"
            print(f"=== start {label} ===", flush=True)
            started = time.time()
            ok = execute(notebook, session_id, RESULTS / session_id)
            elapsed = (time.time() - started) / 60
            print(f"=== {'ok' if ok else 'FAILED'} {label} ({elapsed:.1f} min) ===\n", flush=True)
            if not ok:
                failures.append(label)

    if failures:
        print("failed:", file=sys.stderr)
        for label in failures:
            print(f"  - {label}", file=sys.stderr)
        return 1
    print("all runs completed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
