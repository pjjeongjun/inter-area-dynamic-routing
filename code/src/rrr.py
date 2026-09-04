"""Ridge-regularized reduced-rank regression."""

import numpy as np
from sklearn.metrics import r2_score

from .ridge import fit_ridge, select_ridge_alpha


class RegularizedReducedRankRegression:
    """Ridge-regularized reduced-rank multivariate linear regression.

    One fit represents the complete rank path for one ridge penalty. Passing
    rank=None to predict or coef_at_rank returns the underlying ridge solution.
    Inputs are not centered by default, which is appropriate for activity that
    has already been centered globally. Set ``center=True`` explicitly to fit
    and apply training-set means.
    """

    def __init__(self, alpha, center=False):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite non-negative scalar")
        if not isinstance(center, (bool, np.bool_)):
            raise ValueError("center must be a boolean")
        self.alpha = float(alpha)
        self.center = bool(center)

    def fit(self, source_activity, target_activity):
        """Fit the regularized rank path from source to target activity."""
        source_activity = np.asarray(source_activity, dtype=float)
        target_activity = np.asarray(target_activity, dtype=float)
        if source_activity.ndim != 2 or target_activity.ndim != 2:
            raise ValueError("source_activity and target_activity must be two-dimensional")
        if source_activity.shape[0] != target_activity.shape[0]:
            raise ValueError("source_activity and target_activity must have equal rows")
        if not np.all(np.isfinite(source_activity)) or not np.all(
            np.isfinite(target_activity)
        ):
            raise ValueError("source_activity and target_activity must be finite")

        if self.center:
            self.source_mean_ = source_activity.mean(axis=0)
            self.target_mean_ = target_activity.mean(axis=0)
        else:
            self.source_mean_ = np.zeros(source_activity.shape[1])
            self.target_mean_ = np.zeros(target_activity.shape[1])
        source_centered = source_activity - self.source_mean_
        target_centered = target_activity - self.target_mean_

        ridge = fit_ridge(
            source_centered,
            target_centered,
            self.alpha,
            fit_intercept=False,
        )
        self.full_coef_ = ridge.coef_.T
        fitted_target = source_centered @ self.full_coef_
        _, singular_values, right_vectors = np.linalg.svd(
            fitted_target, full_matrices=False
        )
        self.singular_values_ = singular_values
        self.target_axes_ = right_vectors.T
        self.n_source_features_in_ = source_activity.shape[1]
        self.n_target_features_in_ = target_activity.shape[1]
        if singular_values.size == 0:
            self.max_rank_ = 0
        else:
            tolerance = (
                max(fitted_target.shape)
                * np.finfo(singular_values.dtype).eps
                * singular_values[0]
            )
            self.max_rank_ = int(np.count_nonzero(singular_values > tolerance))
        return self

    def _validate_rank(self, rank):
        if not isinstance(rank, (int, np.integer)) or not 0 <= rank <= self.max_rank_:
            raise ValueError(
                f"rank must be an integer between 0 and {self.max_rank_}"
            )

    def coef_at_rank(self, rank=None):
        """Return coefficients at rank, or the full ridge coefficients."""
        if not hasattr(self, "full_coef_"):
            raise ValueError("fit must be called before requesting coefficients")
        if rank is None:
            return self.full_coef_.copy()
        self._validate_rank(rank)
        target_axes = self.target_axes_[:, :rank]
        return self.full_coef_ @ target_axes @ target_axes.T

    def source_axes_at_rank(self, rank):
        """Return source-to-latent predictive axes for a requested rank."""
        if not hasattr(self, "full_coef_"):
            raise ValueError("fit must be called before requesting axes")
        self._validate_rank(rank)
        return self.full_coef_ @ self.target_axes_[:, :rank]

    def transform_source(self, source_activity, rank):
        """Project source activity onto the fitted predictive coordinates."""
        source_activity = np.asarray(source_activity, dtype=float)
        if source_activity.ndim != 2:
            raise ValueError("source_activity must be two-dimensional")
        if source_activity.shape[1] != self.n_source_features_in_:
            raise ValueError("source_activity has the wrong number of units")
        return (source_activity - self.source_mean_) @ self.source_axes_at_rank(rank)

    def predict(self, source_activity, rank=None):
        """Predict at rank, or use the underlying ridge model if omitted."""
        if not hasattr(self, "full_coef_"):
            raise ValueError("fit must be called before predict")
        source_activity = np.asarray(source_activity, dtype=float)
        if source_activity.ndim != 2:
            raise ValueError("source_activity must be two-dimensional")
        if source_activity.shape[1] != self.full_coef_.shape[0]:
            raise ValueError("source_activity has the wrong number of units")
        if not np.all(np.isfinite(source_activity)):
            raise ValueError("source_activity must be finite")
        coef = self.coef_at_rank(rank)
        return self.target_mean_ + (source_activity - self.source_mean_) @ coef


def evaluate_rrr_split(
    source_train,
    target_train,
    source_test,
    target_test,
    *,
    alpha_grid,
    max_rank=50,
    center=False,
):
    """Fit and evaluate one regularized RRR train/test comparison.

    Ridge alpha is selected from the training data only. The fitted model is
    then applied unchanged to the supplied test data at every supported rank.
    Rank is an analysis axis and is never selected using test performance.
    """
    arrays = {
        "source_train": np.asarray(source_train, dtype=float),
        "target_train": np.asarray(target_train, dtype=float),
        "source_test": np.asarray(source_test, dtype=float),
        "target_test": np.asarray(target_test, dtype=float),
    }
    for name, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if array.shape[0] < 2:
            raise ValueError(f"{name} must contain at least two trials")
        if array.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one unit")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")

    source_train = arrays["source_train"]
    target_train = arrays["target_train"]
    source_test = arrays["source_test"]
    target_test = arrays["target_test"]
    if source_train.shape[0] != target_train.shape[0]:
        raise ValueError("source_train and target_train must have equal rows")
    if source_test.shape[0] != target_test.shape[0]:
        raise ValueError("source_test and target_test must have equal rows")
    if source_train.shape[1] != source_test.shape[1]:
        raise ValueError("source train/test matrices must have equal columns")
    if target_train.shape[1] != target_test.shape[1]:
        raise ValueError("target train/test matrices must have equal columns")

    alpha_grid = np.asarray(alpha_grid, dtype=float)
    if (
        alpha_grid.ndim != 1
        or alpha_grid.size == 0
        or not np.all(np.isfinite(alpha_grid))
        or np.any(alpha_grid < 0)
    ):
        raise ValueError("alpha_grid must contain finite non-negative values")
    if not isinstance(max_rank, (int, np.integer)) or max_rank < 1:
        raise ValueError("max_rank must be a positive integer")

    selected_alpha = select_ridge_alpha(
        source_train,
        target_train,
        alphas=alpha_grid,
        fit_intercept=center,
    )
    model = RegularizedReducedRankRegression(
        alpha=selected_alpha,
        center=center,
    ).fit(source_train, target_train)

    supported_rank = min(
        int(max_rank),
        model.max_rank_,
        source_train.shape[1],
        target_train.shape[1],
    )
    if supported_rank < 1:
        raise ValueError("training data do not support a positive RRR rank")
    ranks = np.arange(1, supported_rank + 1)
    rank_r2 = np.array(
        [
            r2_score(
                target_test,
                model.predict(source_test, rank=rank),
                multioutput="variance_weighted",
            )
            for rank in ranks
        ]
    )
    full_rank_r2 = r2_score(
        target_test,
        model.predict(source_test),
        multioutput="variance_weighted",
    )
    return {
        "ranks": ranks,
        "rank_r2": rank_r2,
        "full_rank_r2": float(full_rank_r2),
        "selected_alpha": selected_alpha,
        "supported_rank": supported_rank,
        "model": model,
    }
