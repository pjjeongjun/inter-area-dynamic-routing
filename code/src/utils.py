"""Backward-compatible imports for historical notebooks and scripts.

New code should import from activity, residualization, or rrr directly.
"""

from .activity import (
    bin_spikes_around_events,
    get_event_spike_count_matrix,
    get_event_window_mean,
)
from .residualization import (
    PopulationResidualizer,
    fit_covariate_model,
    residualize_activity,
    residualize_population,
)
from .rrr import (
    RegularizedReducedRankRegression,
    evaluate_rrr_split,
)
from .ridge import fit_ridge, ridge_score_split, select_ridge_alpha

__all__ = [
    "bin_spikes_around_events",
    "get_event_spike_count_matrix",
    "get_event_window_mean",
    "fit_covariate_model",
    "PopulationResidualizer",
    "residualize_activity",
    "residualize_population",
    "RegularizedReducedRankRegression",
    "evaluate_rrr_split",
    "select_ridge_alpha",
    "fit_ridge",
    "ridge_score_split",
]
