#!/usr/bin/env python
"""Paired sensitivity check for ridge intercept conventions."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.activity import interval_means, interval_rates
from src.data import load_session, select_units, task_trials
from src.residualization import residualize_population
from src.ridge import ridge_score_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--session", default="667252")
    parser.add_argument("--areas", nargs=2, default=["MOs", "MOp"])
    parser.add_argument("--n-neurons", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    session, _, session_id = load_session(args.session)
    trials = task_trials(session)
    starts = trials["quiescent_start_time"].to_numpy(float)
    stops = trials["quiescent_stop_time"].to_numpy(float)
    running = session.processing["behavior"]["running_speed"]
    covariates = trials[["rewarded_modality", "trial_index"]].copy()
    covariates["mean_running_speed"] = interval_means(
        running.timestamps[:], running.data[:], starts, stops
    )
    columns = ["rewarded_modality", "mean_running_speed", "trial_index"]

    residual = {}
    for area in args.areas:
        units = select_units(session.units[:], area)
        rates = interval_rates(units["spike_times"], starts, stops)
        rates = rates[:, np.var(rates, axis=0) > 0]
        residual[area] = residualize_population(
            rates,
            covariates[columns],
            categorical=["rewarded_modality"],
            continuous=["mean_running_speed", "trial_index"],
        )

    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(
            trials, trials["stimulus_context"]
        )
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for sample in range(args.n_samples):
        chosen = {
            area: rng.choice(
                residual[area].shape[1], args.n_neurons, replace=False
            )
            for area in args.areas
        }
        for source_area, target_area in (
            tuple(args.areas),
            tuple(reversed(args.areas)),
        ):
            source = residual[source_area][:, chosen[source_area]]
            target = residual[target_area][:, chosen[target_area]]
            for fold, (train, test) in enumerate(folds):
                for fit_intercept in (False, True):
                    score, alpha = ridge_score_split(
                        source,
                        target,
                        train,
                        test,
                        fit_intercept=fit_intercept,
                    )
                    rows.append(
                        {
                            "session_id": session_id,
                            "source_area": source_area,
                            "target_area": target_area,
                            "unit_sample": sample,
                            "fold": fold,
                            "fit_intercept": fit_intercept,
                            "selected_alpha": alpha,
                            "ridge_r2": score,
                        }
                    )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = (
        result.groupby(
            ["source_area", "target_area", "fit_intercept"], as_index=False
        )
        .agg(ridge_r2=("ridge_r2", "mean"))
    )
    paired = result.pivot(
        index=["source_area", "target_area", "unit_sample", "fold"],
        columns="fit_intercept",
        values="ridge_r2",
    ).reset_index()
    paired["false_minus_true"] = paired[False] - paired[True]
    difference = (
        paired.groupby(["source_area", "target_area"], as_index=False)
        .agg(
            false_minus_true=("false_minus_true", "mean"),
            max_absolute_change=("false_minus_true", lambda x: x.abs().max()),
        )
    )
    print(summary.to_string(index=False))
    print(difference.to_string(index=False))


if __name__ == "__main__":
    main()
