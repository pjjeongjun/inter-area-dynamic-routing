"""Neural and behavioral activity extraction helpers."""

import numpy as np


def interval_rates(spike_series, starts, stops):
    """Compute trial-by-unit firing rates for arbitrary trial intervals."""
    starts = np.asarray(starts, dtype=float)
    stops = np.asarray(stops, dtype=float)
    durations = stops - starts
    if starts.shape != stops.shape or starts.ndim != 1:
        raise ValueError("starts and stops must be matching one-dimensional arrays")
    if np.any(~np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("all intervals must have finite positive duration")
    rates = np.empty((starts.size, len(spike_series)), dtype=float)
    for unit_index, spike_times in enumerate(spike_series):
        spikes = np.sort(np.asarray(spike_times, dtype=float))
        counts = np.searchsorted(spikes, stops, side="left") - np.searchsorted(
            spikes, starts, side="left"
        )
        rates[:, unit_index] = counts / durations
    return rates


def interval_means(timestamps, values, starts, stops):
    """Average a sampled signal within arbitrary trial intervals."""
    timestamps = np.asarray(timestamps, dtype=float)
    values = np.asarray(values, dtype=float)
    starts = np.asarray(starts, dtype=float)
    stops = np.asarray(stops, dtype=float)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    values = values[order]
    means = np.full(starts.size, np.nan, dtype=float)
    for index, (start, stop) in enumerate(zip(starts, stops)):
        left = np.searchsorted(timestamps, start, side="left")
        right = np.searchsorted(timestamps, stop, side="left")
        if right > left:
            means[index] = np.nanmean(values[left:right])
    return means


def resolve_intervals(trials, epoch):
    """Convert an epoch configuration into absolute per-trial intervals."""
    reference = epoch["reference"]
    if reference == "quiescent_interval":
        return (
            trials["quiescent_start_time"].to_numpy(float),
            trials["quiescent_stop_time"].to_numpy(float),
        )
    if reference == "stimulus_onset":
        start, stop = map(float, epoch["window"])
        onset = trials["stim_start_time"].to_numpy(float)
        return onset + start, onset + stop
    raise ValueError(f"Unknown epoch reference: {reference}")


def bin_spikes_around_events(
    spike_times,
    event_times,
    window=(-0.5, 1.0),
    bin_size=0.02,
):
    """Bin one unit's spike times relative to a collection of events.

    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds on the session clock.
    event_times : array-like
        Event times in seconds on the same clock (for example, stimulus onsets).
    window : tuple of float, default (-0.5, 1.0)
        Start and stop times relative to each event. The stop is exclusive.
    bin_size : float, default 0.02
        Width of each time bin in seconds.

    Returns
    -------
    counts : numpy.ndarray
        Integer spike counts with shape (n_events, n_bins).
    bin_edges : numpy.ndarray
        Relative-time bin edges with shape (n_bins + 1,).
    bin_centers : numpy.ndarray
        Relative-time bin centers with shape (n_bins,).
    """
    spike_times = np.asarray(spike_times, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    window_start, window_stop = map(float, window)

    if spike_times.ndim != 1 or event_times.ndim != 1:
        raise ValueError("spike_times and event_times must be one-dimensional")
    if not np.all(np.isfinite(event_times)):
        raise ValueError("event_times must contain only finite values")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    if window_stop <= window_start:
        raise ValueError("window stop must be greater than window start")

    n_bins_float = (window_stop - window_start) / bin_size
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins_float, n_bins):
        raise ValueError("window duration must be an integer multiple of bin_size")

    # searchsorted requires sorted input; NWB spike times normally already are.
    if np.any(np.diff(spike_times) < 0):
        spike_times = np.sort(spike_times)

    bin_edges = window_start + np.arange(n_bins + 1) * bin_size
    bin_centers = bin_edges[:-1] + bin_size / 2
    counts = np.empty((event_times.size, n_bins), dtype=np.int64)

    for event_index, event_time in enumerate(event_times):
        edge_indices = np.searchsorted(
            spike_times,
            event_time + bin_edges,
            side="left",
        )
        counts[event_index] = np.diff(edge_indices)

    return counts, bin_edges, bin_centers


def get_event_spike_count_matrix(units, event_times, window=(0.1, 0.3)):
    """Count spikes in one event-aligned window for every trial and unit."""
    event_times = np.asarray(event_times, dtype=float)
    window_start, window_stop = map(float, window)
    if window_stop <= window_start:
        raise ValueError("window stop must be greater than window start")
    if "spike_times" not in units:
        raise ValueError("units must contain a spike_times column")

    activity = np.empty((event_times.size, len(units)), dtype=np.int64)
    for unit_index, spike_times in enumerate(units["spike_times"]):
        counts, _, _ = bin_spikes_around_events(
            spike_times,
            event_times,
            window=(window_start, window_stop),
            bin_size=window_stop - window_start,
        )
        activity[:, unit_index] = counts[:, 0]
    return activity


def get_event_window_mean(timestamps, values, event_times, window=(0.1, 0.3)):
    """Average a sampled time series within a window around each event."""
    timestamps = np.asarray(timestamps, dtype=float)
    values = np.asarray(values, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    window_start, window_stop = map(float, window)

    if timestamps.ndim != 1 or values.ndim != 1 or event_times.ndim != 1:
        raise ValueError("timestamps, values, and event_times must be one-dimensional")
    if timestamps.size != values.size:
        raise ValueError("timestamps and values must have the same length")
    if window_stop <= window_start:
        raise ValueError("window stop must be greater than window start")

    valid = np.isfinite(timestamps)
    timestamps = timestamps[valid]
    values = values[valid]
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    values = values[order]

    means = np.full(event_times.size, np.nan, dtype=float)
    for event_index, event_time in enumerate(event_times):
        left = np.searchsorted(timestamps, event_time + window_start, side="left")
        right = np.searchsorted(timestamps, event_time + window_stop, side="left")
        if right > left:
            means[event_index] = np.nanmean(values[left:right])
    return means


