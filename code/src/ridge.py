"""Full-rank ridge regression for population prediction."""

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

DEFAULT_ALPHAS = np.logspace(-3, 5, 33)


def select_ridge_alpha(
    source,
    target,
    train=None,
    alphas=DEFAULT_ALPHAS,
    *,
    fit_intercept=False,
):
    """Select a shared multivariate Ridge penalty on the requested rows."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if train is None:
        train = np.arange(source.shape[0])
    selector = RidgeCV(
        alphas=np.asarray(alphas, dtype=float),
        alpha_per_target=False,
        fit_intercept=fit_intercept,
    ).fit(source[train], target[train])
    return float(selector.alpha_)


def fit_ridge(source, target, alpha, train=None, *, fit_intercept=False):
    """Fit a multivariate Ridge model on all rows or a supplied row index."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if train is None:
        train = np.arange(source.shape[0])
    return Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(
        source[train], target[train]
    )


def ridge_score_split(
    source, target, train, test, alphas=DEFAULT_ALPHAS, *, fit_intercept=False
):
    """Select ridge alpha on training trials and score one held-out split.

    Residual activity is expected to be nuisance-regressed and z-scored before
    this function is called. Models therefore use the pooled residual origin by
    default, matching the no-intercept Semedo-style formulation. Alpha selection
    and the final fit always use the same intercept convention.
    """
    alpha = select_ridge_alpha(
        source,
        target,
        train,
        alphas,
        fit_intercept=fit_intercept,
    )
    model = fit_ridge(
        source, target, alpha, train, fit_intercept=fit_intercept
    )
    score = r2_score(
        target[test],
        model.predict(source[test]),
        multioutput="variance_weighted",
    )
    return float(score), alpha
