"""Portable array artifacts for fitted communication-subspace models."""

from pathlib import Path

import numpy as np

from .rrr import RegularizedReducedRankRegression


def save_rrr_model(path, model, rank, **arrays):
    """Save a fitted RRR model and its selected-rank subspace as compressed arrays."""
    rank = int(rank)
    model._validate_rank(rank)
    payload = {
        "alpha": np.asarray(model.alpha),
        "center": np.asarray(model.center),
        "rank": np.asarray(rank),
        "max_rank": np.asarray(model.max_rank_),
        "full_coef": model.full_coef_,
        "rank_coef": model.coef_at_rank(rank),
        "source_axes": model.source_axes_at_rank(rank),
        "target_axes": model.target_axes_[:, :rank],
        "all_target_axes": model.target_axes_,
        "singular_values": model.singular_values_,
        "source_mean": model.source_mean_,
        "target_mean": model.target_mean_,
    }
    payload.update({key: np.asarray(value) for key, value in arrays.items()})
    np.savez_compressed(Path(path), **payload)


def load_rrr_model(path):
    """Restore an RRR estimator and selected rank from a saved array artifact."""
    with np.load(Path(path), allow_pickle=False) as artifact:
        arrays = {key: artifact[key] for key in artifact.files}
    model = RegularizedReducedRankRegression(
        alpha=float(arrays["alpha"]),
        center=bool(arrays["center"]),
    )
    model.full_coef_ = arrays["full_coef"]
    model.target_axes_ = arrays["all_target_axes"]
    model.singular_values_ = arrays["singular_values"]
    model.source_mean_ = arrays["source_mean"]
    model.target_mean_ = arrays["target_mean"]
    model.max_rank_ = int(arrays["max_rank"])
    model.n_source_features_in_ = model.full_coef_.shape[0]
    model.n_target_features_in_ = model.full_coef_.shape[1]
    return model, int(arrays["rank"]), arrays
