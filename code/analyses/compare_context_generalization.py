#!/usr/bin/env python
"""Compare within- and cross-context ridge prediction for one area pair."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.residualization import residualize_population
from src.ridge import ridge_score_split
from src.activity import interval_means, interval_rates
from src.data import TASK_STIMULI, load_session, select_units, task_trials

def balanced_context_folds(trials, n_folds, seed):
    rng = np.random.default_rng(seed)
    context_values = {"visual": "vis", "auditory": "aud"}
    indices = {context: [] for context in context_values}
    for stimulus in TASK_STIMULI:
        candidates = {
            context: np.flatnonzero(
                trials["rewarded_modality"].eq(value)
                & trials["stim_name"].eq(stimulus)
            )
            for context, value in context_values.items()
        }
        count = min(map(len, candidates.values()))
        for context, values in candidates.items():
            indices[context].extend(rng.choice(values, count, replace=False))
    folds = {}
    for context, values in indices.items():
        values = np.asarray(values, dtype=int)
        labels = trials.iloc[values]["stim_name"].to_numpy()
        splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        folds[context] = [
            (values[train], values[test])
            for train, test in splitter.split(values, labels)
        ]
    return indices, folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--session", default="667252")
    parser.add_argument("--areas", nargs=2, default=["MOs", "MOp"])
    parser.add_argument("--n-neurons", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    session, _, session_id = load_session(args.session)
    trials = task_trials(session)
    starts = trials["quiescent_start_time"].to_numpy(float)
    stops = trials["quiescent_stop_time"].to_numpy(float)
    running = session.processing["behavior"]["running_speed"]
    covariates = trials[["rewarded_modality", "trial_index"]].copy()
    covariates["mean_running_speed"] = interval_means(
        running.timestamps[:], running.data[:], starts, stops
    )

    residual = {}
    available = {}
    for area in args.areas:
        units = select_units(session.units[:], area)
        rates = interval_rates(units["spike_times"], starts, stops)
        rates = rates[:, np.var(rates, axis=0) > 0]
        available[area] = rates.shape[1]
        if rates.shape[1] < args.n_neurons:
            raise ValueError(f"{area} has only {rates.shape[1]} active units")
        residual[area] = residualize_population(
            rates,
            covariates,
            categorical=["rewarded_modality"],
            continuous=["mean_running_speed", "trial_index"],
        )

    balanced, folds = balanced_context_folds(
        trials, args.n_folds, args.seed
    )
    comparisons = {
        "visual_to_visual": ("visual", "visual"),
        "visual_to_auditory": ("visual", "auditory"),
        "auditory_to_auditory": ("auditory", "auditory"),
        "auditory_to_visual": ("auditory", "visual"),
    }
    rng = np.random.default_rng(args.seed)
    rows = []
    for sample in range(args.n_samples):
        chosen = {
            area: rng.choice(available[area], args.n_neurons, replace=False)
            for area in args.areas
        }
        for source_area, target_area in (
            tuple(args.areas),
            tuple(reversed(args.areas)),
        ):
            source = residual[source_area][:, chosen[source_area]]
            target = residual[target_area][:, chosen[target_area]]
            for fold in range(args.n_folds):
                for comparison, (train_context, test_context) in comparisons.items():
                    train = folds[train_context][fold][0]
                    test = folds[test_context][fold][1]
                    score, alpha = ridge_score_split(
                        source, target, train, test
                    )
                    rows.append(
                        {
                            "session_id": session_id,
                            "source_area": source_area,
                            "target_area": target_area,
                            "comparison": comparison,
                            "unit_sample": sample,
                            "fold": fold,
                            "selected_alpha": alpha,
                            "ridge_r2": score,
                        }
                    )

    scores = pd.DataFrame(rows)
    scores.to_csv(args.output_dir / "fold_scores.csv", index=False)
    summary = (
        scores.groupby(
            ["session_id", "source_area", "target_area", "comparison"],
            as_index=False,
        )
        .agg(
            ridge_r2=("ridge_r2", "mean"),
            sd=("ridge_r2", "std"),
            n=("ridge_r2", "size"),
        )
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    paired = scores.pivot(
        index=["session_id", "source_area", "target_area", "unit_sample", "fold"],
        columns="comparison",
        values="ridge_r2",
    ).reset_index()
    paired["visual_transfer_loss"] = (
        paired["visual_to_auditory"] - paired["visual_to_visual"]
    )
    paired["auditory_transfer_loss"] = (
        paired["auditory_to_visual"] - paired["auditory_to_auditory"]
    )
    paired.to_csv(args.output_dir / "paired_transfer.csv", index=False)
    transfer_summary = (
        paired.groupby(["source_area", "target_area"], as_index=False)
        .agg(
            visual_within=("visual_to_visual", "mean"),
            visual_to_auditory=("visual_to_auditory", "mean"),
            visual_transfer_loss=("visual_transfer_loss", "mean"),
            auditory_within=("auditory_to_auditory", "mean"),
            auditory_to_visual=("auditory_to_visual", "mean"),
            auditory_transfer_loss=("auditory_transfer_loss", "mean"),
        )
    )
    transfer_summary.to_csv(
        args.output_dir / "transfer_summary.csv", index=False
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "areas": args.areas,
        "epoch": "full_quiescent_interval",
        "n_neurons": args.n_neurons,
        "n_samples": args.n_samples,
        "n_folds": args.n_folds,
        "balanced_trials": {key: len(value) for key, value in balanced.items()},
        "nuisance": [
            "rewarded_modality",
            "mean_running_speed",
            "trial_index",
        ],
        "residualization_scope": "pooled session before context split",
        "fit_intercept": False,
        "seed": args.seed,
        "available_units": available,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(transfer_summary.to_string(index=False))


if __name__ == "__main__":
    main()
