"""Covariate modeling and neural-activity residualization."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class PopulationResidualizer:
    """Fitted global preprocessing for one neural population.

    The object retains both activity scalers and the fitted nuisance model so
    the exact transformation used to fit a saved communication model can be
    reapplied later.
    """

    def __init__(self, categorical=(), continuous=()):
        self.categorical = tuple(categorical)
        self.continuous = tuple(continuous)
        if not self.categorical and not self.continuous:
            raise ValueError("at least one covariate column is required")

    def fit(self, activity, covariates):
        activity = np.asarray(activity, dtype=float)
        if activity.ndim != 2 or activity.shape[1] == 0:
            raise ValueError("activity must be a non-empty two-dimensional array")
        if activity.shape[0] != len(covariates):
            raise ValueError("activity and covariates must contain equal rows")
        if not np.all(np.isfinite(activity)):
            raise ValueError("activity must contain only finite values")

        self.activity_scaler_ = StandardScaler().fit(activity)
        standardized = self.activity_scaler_.transform(activity)
        self.nuisance_model_ = fit_covariate_model(
            covariates,
            standardized,
            categorical_columns=self.categorical,
            continuous_columns=self.continuous,
        )
        residual = residualize_activity(
            self.nuisance_model_, covariates, standardized
        )
        self.residual_scaler_ = StandardScaler().fit(residual)
        self.n_features_in_ = activity.shape[1]
        return self

    def transform(self, activity, covariates):
        if not hasattr(self, "activity_scaler_"):
            raise ValueError("fit must be called before transform")
        activity = np.asarray(activity, dtype=float)
        if activity.ndim != 2 or activity.shape[1] != self.n_features_in_:
            raise ValueError("activity has the wrong number of units")
        if activity.shape[0] != len(covariates):
            raise ValueError("activity and covariates must contain equal rows")
        standardized = self.activity_scaler_.transform(activity)
        residual = residualize_activity(
            self.nuisance_model_, covariates, standardized
        )
        return self.residual_scaler_.transform(residual)

    def fit_transform(self, activity, covariates):
        return self.fit(activity, covariates).transform(activity, covariates)


def fit_covariate_model(
    covariates,
    activity,
    categorical_columns=(),
    continuous_columns=(),
):
    """Fit a multivariate OLS model from trial covariates to neural activity.

    The encoded design uses reference coding for categorical variables. Before
    fitting, this function verifies that there are more observations than OLS
    parameters (including the intercept) and that the augmented design matrix
    has full column rank.
    """
    categorical_columns = list(categorical_columns)
    continuous_columns = list(continuous_columns)
    if not categorical_columns and not continuous_columns:
        raise ValueError("at least one covariate column is required")

    transformers = []
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if continuous_columns:
        continuous_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("continuous", continuous_pipeline, continuous_columns))

    preprocessor = ColumnTransformer(transformers=transformers)
    design = np.asarray(preprocessor.fit_transform(covariates), dtype=float)
    augmented_design = np.column_stack([np.ones(design.shape[0]), design])
    n_trials, n_parameters = augmented_design.shape
    if n_trials <= n_parameters:
        raise ValueError(
            "OLS covariate model requires more training trials than parameters: "
            f"got {n_trials} trials and {n_parameters} parameters "
            "(including the intercept)"
        )
    design_rank = np.linalg.matrix_rank(augmented_design)
    if design_rank < n_parameters:
        raise ValueError(
            "OLS covariate design matrix is rank deficient: "
            f"rank {design_rank} for {n_parameters} parameters"
        )

    model = Pipeline(
        [
            ("covariates", preprocessor),
            ("ols", LinearRegression()),
        ]
    )
    model.fit(covariates, np.asarray(activity, dtype=float))
    return model


def residualize_activity(model, covariates, activity):
    """Subtract activity predicted by a fitted covariate model."""
    activity = np.asarray(activity, dtype=float)
    prediction = np.asarray(model.predict(covariates), dtype=float)
    if activity.ndim == 2 and prediction.ndim == 1:
        prediction = prediction[:, np.newaxis]
    if prediction.shape != activity.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match activity {activity.shape}"
        )
    return activity - prediction


def residualize_population(activity, covariates, categorical, continuous):
    """Globally standardize, residualize, and restandardize one population.

    All transforms are fit once using the complete supplied trial set. This is
    the project's intentional preprocessing convention: downstream prediction
    models cross-validate only the neural mapping, not nuisance removal.
    """
    return PopulationResidualizer(categorical, continuous).fit_transform(
        activity, covariates
    )
