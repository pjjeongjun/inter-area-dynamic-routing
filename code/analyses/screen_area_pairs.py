#!/usr/bin/env python
"""Run a configuration-driven residual ridge screen across sessions."""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import StratifiedKFold

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.activity import interval_means, interval_rates, resolve_intervals
from src.data import load_session, select_units, task_trials
from src.residualization import residualize_population
from src.ridge import ridge_score_split


def git_commit():
    """Return the current commit without requiring a clean worktree."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def validate_config(config):
    required = {
        "name",
        "sessions",
        "area_pairs",
        "epochs",
        "n_neurons",
        "n_unit_samples",
        "n_folds",
        "nuisance",
        "stratify_by",
        "random_seed",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if config["n_neurons"] < 1 or config["n_unit_samples"] < 1:
        raise ValueError("n_neurons and n_unit_samples must be positive")
    if config["n_folds"] < 2:
        raise ValueError("n_folds must be at least two")


def analyze_session(session_ref, config):
    session, _, session_id = load_session(session_ref)
    units = session.units[:]
    trials = task_trials(session)
    stratify_by = list(config["stratify_by"])
    missing_strata = set(stratify_by).difference(trials.columns)
    if missing_strata:
        raise ValueError(f"Unknown stratification columns: {sorted(missing_strata)}")
    strata = trials[stratify_by].astype(str).agg(" | ".join, axis=1).to_numpy()
    folds = list(
        StratifiedKFold(
            config["n_folds"],
            shuffle=True,
            random_state=config["random_seed"],
        ).split(trials, strata)
    )
    pairs = [tuple(pair) for pair in config["area_pairs"]]
    areas = sorted({area for pair in pairs for area in pair})
    selected = {area: select_units(units, area) for area in areas}
    running = session.processing["behavior"]["running_speed"]
    rows = []

    for epoch_index, epoch in enumerate(config["epochs"]):
        starts, stops = resolve_intervals(trials, epoch)
        covariates = trials.copy()
        covariates["mean_running_speed"] = interval_means(
            running.timestamps[:],
            running.data[:],
            starts,
            stops,
        )
        categorical = list(config["nuisance"]["categorical"])
        continuous = list(config["nuisance"]["continuous"])
        requested_covariates = categorical + continuous
        missing_covariates = set(requested_covariates).difference(covariates.columns)
        if missing_covariates:
            raise ValueError(f"Unknown nuisance covariates: {sorted(missing_covariates)}")
        residual = {}
        available = {}

        for area in areas:
            rates = interval_rates(selected[area]["spike_times"], starts, stops)
            rates = rates[:, np.var(rates, axis=0) > 0]
            available[area] = rates.shape[1]
            if rates.shape[1]:
                residual[area] = residualize_population(
                    rates,
                    covariates[requested_covariates],
                    categorical,
                    continuous,
                )

        for pair_index, (area_a, area_b) in enumerate(pairs):
            if min(available[area_a], available[area_b]) < config["n_neurons"]:
                continue
            rng = np.random.default_rng(
                config["random_seed"] + 100 * pair_index
            )
            samples = [
                {
                    area_a: rng.choice(
                        available[area_a], config["n_neurons"], replace=False
                    ),
                    area_b: rng.choice(
                        available[area_b], config["n_neurons"], replace=False
                    ),
                }
                for _ in range(config["n_unit_samples"])
            ]
            for source_area, target_area in ((area_a, area_b), (area_b, area_a)):
                for sample_index, sample in enumerate(samples):
                    source = residual[source_area][:, sample[source_area]]
                    target = residual[target_area][:, sample[target_area]]
                    for fold_index, (train, test) in enumerate(folds):
                        score, alpha = ridge_score_split(
                            source, target, train, test
                        )
                        rows.append(
                            {
                                "session_id": session_id,
                                "epoch": epoch["name"],
                                "source_area": source_area,
                                "target_area": target_area,
                                "unit_sample": sample_index,
                                "fold": fold_index,
                                "n_neurons": config["n_neurons"],
                                "n_trials": len(trials),
                                "n_source_available": available[source_area],
                                "n_target_available": available[target_area],
                                "selected_alpha": alpha,
                                "ridge_r2": score,
                            }
                        )
        print(f"{session_id}: finished {epoch['name']}", flush=True)
    return pd.DataFrame(rows)


def write_outputs(fold_scores, config, config_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "config.yaml").open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    fold_scores.to_csv(output_dir / "fold_scores.csv", index=False)
    group = ["session_id", "epoch", "source_area", "target_area"]
    session_summary = (
        fold_scores.groupby(group, as_index=False)
        .agg(
            ridge_r2=("ridge_r2", "mean"),
            fold_sample_sd=("ridge_r2", "std"),
            n_fold_sample_scores=("ridge_r2", "size"),
        )
    )
    session_summary.to_csv(output_dir / "session_summary.csv", index=False)

    cross_session = (
        session_summary.groupby(
            ["epoch", "source_area", "target_area"], as_index=False
        )
        .agg(
            n_sessions=("session_id", "nunique"),
            mean_across_sessions=("ridge_r2", "mean"),
            worst_session=("ridge_r2", "min"),
            best_session=("ridge_r2", "max"),
            sd_across_sessions=("ridge_r2", "std"),
        )
    )
    cross_session.to_csv(output_dir / "cross_session_summary.csv", index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": config["name"],
        "config_source": str(config_path.resolve()),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "preprocessing_scope": "whole-session exploratory residualization",
        "stratify_by": config["stratify_by"],
        "fit_intercept": False,
        "scientific_unit": "session/mouse",
    }
    with (output_dir / "run_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--session",
        action="append",
        help="Override config sessions; repeat for multiple sessions.",
    )
    args = parser.parse_args()

    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    if args.session:
        config["sessions"] = args.session

    frames = [analyze_session(ref, config) for ref in config["sessions"]]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No configured session had enough units for any pair")
    write_outputs(
        pd.concat(frames, ignore_index=True),
        config,
        args.config,
        args.output_dir,
    )
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
