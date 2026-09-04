#!/usr/bin/env python
"""Refit and save all final models from one dimensionality-selection sample."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.activity import interval_means, interval_rates
from src.data import load_session, select_units, task_trials
from src.model_artifacts import save_rrr_model
from src.residualization import PopulationResidualizer
from src.ridge import select_ridge_alpha
from src.rrr import RegularizedReducedRankRegression


def read_selection(selection_dir, unit_sample):
    metadata = json.loads((selection_dir / "run_metadata.json").read_text())
    units_path = selection_dir / "sampled_units.csv"
    if not units_path.exists():
        raise ValueError("selection run lacks sampled_units.csv; rerun dimensionality")
    expected = {"rewarded_modality", "mean_running_speed", "trial_index"}
    if set(metadata.get("nuisance", ())) != expected:
        raise ValueError("selection run does not use quiescent-primary nuisances")
    if metadata.get("fit_intercept") is not False:
        raise ValueError("selection run does not use the no-intercept convention")
    if metadata.get("epoch") != "full_quiescent_interval":
        raise ValueError("selection run is not a full-quiescent analysis")
    selections = pd.read_csv(selection_dir / "selected_dimensions.csv")
    selections = selections.loc[selections["sample"].eq(unit_sample)].copy()
    expected_rows = 4 * int(metadata["outer_folds"])
    if len(selections) != expected_rows:
        raise ValueError("selection sample does not contain all four mappings and folds")
    sampled_units = pd.read_csv(units_path)
    sampled_units = sampled_units.loc[
        sampled_units["sample"].eq(unit_sample)
    ].copy()
    return metadata, selections, sampled_units


def population_rates(session, area, unit_ids, starts, stops):
    units = select_units(session.units[:], area)
    missing = set(unit_ids).difference(units.index)
    if missing:
        raise ValueError(f"saved unit IDs are missing from {area}: {sorted(missing)}")
    return interval_rates(units.loc[unit_ids, "spike_times"], starts, stops)


def rounded_median(values):
    return int(np.floor(np.median(values) + 0.5))


def fit_mapping(output_dir, selection, source, target, folds):
    output_dir.mkdir()
    fold_dir = output_dir / "fold_models"
    fold_dir.mkdir()
    rows = []
    selection = selection.sort_values("fold")
    for (fold, (train, test)), row in zip(enumerate(folds), selection.itertuples()):
        if fold != int(row.fold):
            raise ValueError("saved selection folds do not match regenerated folds")
        rank = int(row.predictive_dimension)
        alpha = float(row.predictive_alpha)
        model = RegularizedReducedRankRegression(alpha, center=False).fit(
            source[train], target[train]
        )
        score = r2_score(
            target[test], model.predict(source[test], rank=rank),
            multioutput="variance_weighted",
        )
        save_rrr_model(
            fold_dir / f"fold_{fold}.npz", model, rank,
            train_indices=train, test_indices=test, test_r2=score,
        )
        rows.append({"fold": fold, "alpha": alpha, "rank": rank, "test_r2": score})

    final_rank = rounded_median(selection["predictive_dimension"])
    final_alpha = select_ridge_alpha(source, target)
    model = RegularizedReducedRankRegression(final_alpha, center=False).fit(
        source, target
    )
    if final_rank > model.max_rank_:
        raise ValueError(f"all-trial model supports only rank {model.max_rank_}")
    save_rrr_model(output_dir / "final_model.npz", model, final_rank)
    pd.DataFrame(rows).to_csv(output_dir / "cv_scores.csv", index=False)
    return final_rank, final_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--unit-sample", type=int, default=0)
    args = parser.parse_args()
    metadata, selections, sampled_units = read_selection(
        args.selection_dir, args.unit_sample
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    session, _, session_id = load_session(metadata["session_id"])
    trials = task_trials(session)
    starts = trials["quiescent_start_time"].to_numpy(float)
    stops = trials["quiescent_stop_time"].to_numpy(float)
    running = session.processing["behavior"]["running_speed"]
    covariates = trials[["rewarded_modality", "trial_index"]].copy()
    covariates["mean_running_speed"] = interval_means(
        running.timestamps[:], running.data[:], starts, stops
    )
    labels = trials["stimulus_context"].to_numpy()
    nuisance = {
        "categorical": ["rewarded_modality"],
        "continuous": ["mean_running_speed", "trial_index"],
    }

    populations = {}
    preprocessors = {}
    for area in metadata["areas"]:
        for group in ("A", "B"):
            table = sampled_units.loc[
                sampled_units["area"].eq(area)
                & sampled_units["group"].eq(group)
            ].sort_values("model_column")
            unit_ids = table["unit_id"].to_numpy()
            if len(unit_ids) != int(metadata["n_neurons_per_population"]):
                raise ValueError(f"incomplete sampled units for {area} group {group}")
            rates = population_rates(session, area, unit_ids, starts, stops)
            preprocessor = PopulationResidualizer(**nuisance)
            populations[(area, group)] = preprocessor.fit_transform(
                rates, covariates
            )
            preprocessors[f"{area}_{group}"] = preprocessor

    seed = int(metadata["random_seed"])
    folds = list(
        StratifiedKFold(
            int(metadata["outer_folds"]), shuffle=True, random_state=seed
        ).split(trials, labels)
    )
    mapping_rows = []
    mappings = selections[
        ["mapping_type", "source_area", "target_area"]
    ].drop_duplicates()
    for mapping in mappings.itertuples(index=False):
        source_group = "A"
        target_group = "B" if mapping.mapping_type == "within" else "A"
        selected = selections.loc[
            selections["mapping_type"].eq(mapping.mapping_type)
            & selections["source_area"].eq(mapping.source_area)
            & selections["target_area"].eq(mapping.target_area)
        ]
        name = f"{mapping.mapping_type}_{mapping.source_area}_to_{mapping.target_area}"
        rank, alpha = fit_mapping(
            args.output_dir / name,
            selected,
            populations[(mapping.source_area, source_group)],
            populations[(mapping.target_area, target_group)],
            folds,
        )
        mapping_rows.append(
            {
                "mapping": name,
                "source_group": source_group,
                "target_group": target_group,
                "final_rank": rank,
                "final_alpha": alpha,
            }
        )

    joblib.dump(preprocessors, args.output_dir / "preprocessing.joblib")
    sampled_units.to_csv(args.output_dir / "sampled_units.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(args.output_dir / "models.csv", index=False)
    trials.assign(analysis_trial=np.arange(len(trials))).to_csv(
        args.output_dir / "trials.csv", index=False
    )
    output_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_dir": str(args.selection_dir.resolve()),
        "selection_unit_sample": args.unit_sample,
        "session_id": session_id,
        "epoch": "full_quiescent_interval",
        "categorical_nuisance": nuisance["categorical"],
        "continuous_nuisance": nuisance["continuous"],
        "stratify_by": ["stimulus_context"],
        "residualization_scope": "global",
        "fit_intercept": False,
        "random_seed": seed,
        "rank_source": "rounded median of outer-fold selected ranks",
        "alpha_source": "all-trial RidgeCV",
        "model_role": "all-trial descriptive refits; report fold cv_scores",
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
