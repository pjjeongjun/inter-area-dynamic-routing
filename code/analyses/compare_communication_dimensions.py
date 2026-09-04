#!/usr/bin/env python
"""Semedo-style comparison of predictive, dominant, and within-area dimensions."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.residualization import residualize_population
from src.ridge import DEFAULT_ALPHAS, fit_ridge, select_ridge_alpha
from src.rrr import evaluate_rrr_split
from src.activity import interval_means, interval_rates
from src.data import load_session, select_units, task_trials

def one_se_dimension(scores):
    """Choose the smallest dimension within one SE of the best mean."""
    scores = np.asarray(scores, float)
    means = np.nanmean(scores, axis=0)
    sems = np.nanstd(scores, axis=0, ddof=1) / np.sqrt(scores.shape[0])
    best = int(np.nanargmax(means))
    eligible = np.flatnonzero(means >= means[best] - sems[best])
    return int(eligible[0] + 1), means, sems


def predictive_curve(source_train, target_train, source_test, target_test):
    result = evaluate_rrr_split(
        source_train,
        target_train,
        source_test,
        target_test,
        alpha_grid=DEFAULT_ALPHAS,
        max_rank=min(source_train.shape[1], target_train.shape[1]),
        center=False,
    )
    return result["rank_r2"], result["selected_alpha"]


def dominant_curve(source_train, target_train, source_test, target_test):
    max_dim = min(source_train.shape[1], target_train.shape[1])
    pca = PCA(n_components=max_dim).fit(source_train)
    train_scores = pca.transform(source_train)
    test_scores = pca.transform(source_test)
    scores = np.empty(max_dim)
    alphas = np.empty(max_dim)
    for dimension in range(1, max_dim + 1):
        alpha = select_ridge_alpha(
            train_scores[:, :dimension],
            target_train,
        )
        model = fit_ridge(
            train_scores[:, :dimension], target_train, alpha
        )
        scores[dimension - 1] = r2_score(
            target_test,
            model.predict(test_scores[:, :dimension]),
            multioutput="variance_weighted",
        )
        alphas[dimension - 1] = alpha
    return scores, alphas


def inner_curves(source, target, labels, n_splits, seed):
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    predictive = []
    dominant = []
    for train, validation in splitter.split(source, labels):
        pred, _ = predictive_curve(
            source[train], target[train], source[validation], target[validation]
        )
        dom, _ = dominant_curve(
            source[train], target[train], source[validation], target[validation]
        )
        predictive.append(pred)
        dominant.append(dom)
    return np.asarray(predictive), np.asarray(dominant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--session", default="667252")
    parser.add_argument("--areas", nargs=2, default=["MOs", "MOp"])
    parser.add_argument("--n-neurons", type=int, default=25)
    parser.add_argument("--n-samples", type=int, default=10)
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
    labels = trials["stimulus_context"].to_numpy()

    activity = {}
    active_units = {}
    for area in args.areas:
        units = select_units(session.units[:], area)
        rates = interval_rates(units["spike_times"], starts, stops)
        active = np.var(rates, axis=0) > 0
        active_units[area] = units.iloc[np.flatnonzero(active)].copy()
        activity[area] = rates[:, active]
        if activity[area].shape[1] < 2 * args.n_neurons:
            raise ValueError(
                f"{area} has {activity[area].shape[1]} active units; "
                f"need {2 * args.n_neurons}"
            )

    residual = {
        area: residualize_population(
            activity[area],
            covariates,
            categorical=["rewarded_modality"],
            continuous=["mean_running_speed", "trial_index"],
        )
        for area in args.areas
    }

    outer = list(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(
            trials, labels
        )
    )
    rng = np.random.default_rng(args.seed)
    curve_rows = []
    selection_rows = []
    sampled_unit_rows = []
    area_a, area_b = args.areas
    mappings = (
        ("between", area_a, area_b, "A", "A"),
        ("between", area_b, area_a, "A", "A"),
        ("within", area_a, area_a, "A", "B"),
        ("within", area_b, area_b, "A", "B"),
    )

    for sample in range(args.n_samples):
        indices = {}
        for area in args.areas:
            chosen = rng.choice(
                activity[area].shape[1], 2 * args.n_neurons, replace=False
            )
            indices[(area, "A")] = chosen[: args.n_neurons]
            indices[(area, "B")] = chosen[args.n_neurons :]
            for group in ("A", "B"):
                for model_column, unit_index in enumerate(indices[(area, group)]):
                    sampled_unit_rows.append(
                        {
                            "sample": sample,
                            "area": area,
                            "group": group,
                            "model_column": model_column,
                            "unit_id": active_units[area].index[unit_index],
                        }
                    )

        for fold, (train, test) in enumerate(outer):
            for mapping_type, source_area, target_area, source_group, target_group in mappings:
                source_train = residual[source_area][train][
                    :, indices[(source_area, source_group)]
                ]
                source_test = residual[source_area][test][
                    :, indices[(source_area, source_group)]
                ]
                target_train = residual[target_area][train][
                    :, indices[(target_area, target_group)]
                ]
                target_test = residual[target_area][test][
                    :, indices[(target_area, target_group)]
                ]

                inner_pred, inner_dom = inner_curves(
                    source_train,
                    target_train,
                    labels[train],
                    n_splits=4,
                    seed=args.seed + fold,
                )
                selected_predictive, pred_mean, pred_sem = one_se_dimension(
                    inner_pred
                )
                selected_dominant, dom_mean, dom_sem = one_se_dimension(inner_dom)
                predictive_target = pred_mean[selected_predictive - 1]
                matching = np.flatnonzero(
                    dom_mean
                    >= predictive_target - pred_sem[selected_predictive - 1]
                )
                dominant_to_match = (
                    int(matching[0] + 1) if matching.size else np.nan
                )

                outer_pred, pred_alpha = predictive_curve(
                    source_train, target_train, source_test, target_test
                )
                outer_dom, dom_alphas = dominant_curve(
                    source_train, target_train, source_test, target_test
                )
                base = {
                    "session_id": session_id,
                    "sample": sample,
                    "fold": fold,
                    "mapping_type": mapping_type,
                    "source_area": source_area,
                    "target_area": target_area,
                }
                for dimension, (pred_score, dom_score) in enumerate(
                    zip(outer_pred, outer_dom), start=1
                ):
                    curve_rows.extend(
                        [
                            {
                                **base,
                                "model": "predictive_rrr",
                                "dimension": dimension,
                                "ridge_r2": pred_score,
                            },
                            {
                                **base,
                                "model": "dominant_pca",
                                "dimension": dimension,
                                "ridge_r2": dom_score,
                            },
                        ]
                    )
                selection_rows.append(
                    {
                        **base,
                        "predictive_dimension": selected_predictive,
                        "dominant_dimension": selected_dominant,
                        "dominant_dimensions_to_match_predictive": dominant_to_match,
                        "predictive_test_r2": outer_pred[
                            selected_predictive - 1
                        ],
                        "dominant_same_dimension_test_r2": outer_dom[
                            selected_predictive - 1
                        ],
                        "dominant_selected_test_r2": outer_dom[
                            selected_dominant - 1
                        ],
                        "predictive_alpha": pred_alpha,
                        "dominant_alpha_at_selected": dom_alphas[
                            selected_dominant - 1
                        ],
                    }
                )
        print(f"finished neuron sample {sample + 1}/{args.n_samples}", flush=True)

    curves = pd.DataFrame(curve_rows)
    selections = pd.DataFrame(selection_rows)
    sampled_units = pd.DataFrame(sampled_unit_rows)
    curves.to_csv(args.output_dir / "dimension_curves.csv", index=False)
    selections.to_csv(args.output_dir / "selected_dimensions.csv", index=False)
    sampled_units.to_csv(args.output_dir / "sampled_units.csv", index=False)
    summary = (
        selections.groupby(
            ["mapping_type", "source_area", "target_area"], as_index=False
        )
        .agg(
            predictive_dimension_mean=("predictive_dimension", "mean"),
            predictive_dimension_median=("predictive_dimension", "median"),
            dominant_to_match_mean=(
                "dominant_dimensions_to_match_predictive",
                "mean",
            ),
            predictive_test_r2=("predictive_test_r2", "mean"),
            dominant_same_dimension_test_r2=(
                "dominant_same_dimension_test_r2",
                "mean",
            ),
        )
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "areas": args.areas,
        "epoch": "full_quiescent_interval",
        "n_neurons_per_population": args.n_neurons,
        "n_neuron_samples": args.n_samples,
        "outer_folds": 5,
        "inner_folds": 4,
        "rank_rule": "smallest rank within one SE of best inner-CV mean",
        "nuisance": [
            "rewarded_modality",
            "mean_running_speed",
            "trial_index",
        ],
        "stratify_by": ["stimulus_context"],
        "preprocessing": (
            "global activity scaling, nuisance regression, and residual "
            "scaling before prediction CV"
        ),
        "fit_intercept": False,
        "sampled_unit_table": "sampled_units.csv",
        "random_seed": args.seed,
        "available_units": {area: matrix.shape[1] for area, matrix in activity.items()},
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
