"""Ridge nuisance regression: running speed, keypoint position, prev_stim, prev_response, lick_history, spline."""
import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge

BIN = 1
N_BINS = 1
N_KNOTS = 2
N_FOLDS = 10
ALPHAS = np.logspace(-2, 6, 17)
KP_CAM = "side"
KP_PARTS = ("jaw", "nose_tip", "ear_base_l", "whisker_pad_l_top")
LIK_MIN = 0.99


def _sample_at(ts, data, edges):
    """Mean of each stream within each bin."""
    idx = np.searchsorted(ts, edges)
    csum = np.concatenate([np.zeros((1,) + data.shape[1:]), np.cumsum(data, axis=0, dtype=np.float64)])
    tot = csum[idx[:, 1:]] - csum[idx[:, :-1]]
    n = np.diff(idx, axis=1)
    return tot / n if data.ndim == 1 else tot / n[..., None]


def _timeseries(beh, name, ncols=None):
    o = beh[name]
    ts = np.asarray(o.timestamps[:])
    ok = np.isfinite(ts)
    d = np.asarray(o.data[:, :ncols] if ncols else o.data[:])
    return ts[ok], d[ok]


def _keypoints(beh):
    tbl = beh[f"lp_{KP_CAM}_camera"]
    ts = np.asarray(tbl["timestamps"][:])
    ok = np.isfinite(ts)
    parts = sorted(c[:-2] for c in tbl.colnames if c.endswith("_x"))
    lik = {p: np.median(np.asarray(tbl[p + "_likelihood"][:])[ok]) for p in parts}
    keep = [p for p in parts if p in KP_PARTS and lik[p] >= LIK_MIN]
    xy = np.stack([np.asarray(tbl[f"{p}_{a}"][:])[ok] for p in keep for a in "xy"], axis=1)
    return ts[ok], xy, keep


def _design_matrix(nwb, trials, bin_edges):
    beh = nwb.processing["behavior"]
    n_trials = len(trials)

    run_ts, run = _timeseries(beh, "running_speed")
    running_b = _sample_at(run_ts, run, bin_edges)

    kp_ts, kp_xy, kp_keep = _keypoints(beh)
    kp_pos_b = _sample_at(kp_ts, kp_xy, bin_edges)
    kp_pos_names = [f"kppos_{KP_CAM}_{p}_{a}" for p in kp_keep for a in "xy"]

    trials_all = nwb.intervals["trials"].to_dataframe()
    pos = trials.trial_index.to_numpy().astype(int)
    prev_stim = trials_all.stim_name.shift(1).to_numpy()[pos]
    prev_response = trials_all.is_response.shift(1).to_numpy()[pos].astype(float)

    lick_ts = np.asarray(beh["licks"]["timestamps"][:])
    q0 = trials.quiescent_start_time.to_numpy()
    j = np.searchsorted(lick_ts, q0)
    time_since_lick = q0 - lick_ts[j - 1]
    licks_2s = (j - np.searchsorted(lick_ts, q0 - 2.0)).astype(float)

    sess_time = trials.start_time.to_numpy()[:, None]
    spline = SplineTransformer(n_knots=N_KNOTS, degree=3, extrapolation="constant").fit_transform(sess_time)[:, 1:]

    stim_levels = sorted(pd.unique(trials_all.stim_name.dropna()))
    prev_stim_oh = np.stack([(prev_stim == s).astype(float) for s in stim_levels[1:]], axis=1)

    per_bin = {"running": running_b}
    per_bin |= {n: kp_pos_b[..., j] for j, n in enumerate(kp_pos_names)}

    per_trial = {"prev_response": prev_response, "time_since_lick": time_since_lick, "licks_2s": licks_2s}
    per_trial |= {f"spline{j}": spline[:, j] for j in range(spline.shape[1])}
    per_trial |= {f"prev_{s}": prev_stim_oh[:, j] for j, s in enumerate(stim_levels[1:])}

    reg_names = list(per_bin) + list(per_trial)
    X = np.concatenate([
        np.stack(list(per_bin.values()), axis=-1),
        np.broadcast_to(np.stack(list(per_trial.values()), axis=-1)[:, None, :], (n_trials, N_BINS, len(per_trial))),
    ], axis=-1)
    return X.reshape(n_trials * N_BINS, -1), reg_names


def _block_folds(trials, seed=0):
    rng = npr.default_rng(seed)
    fold = np.empty(len(trials), dtype=int)
    for _, idx in trials.groupby("block_index").indices.items():
        fold[idx] = rng.permutation(np.arange(len(idx)) % N_FOLDS)
    return np.repeat(fold, N_BINS)


def get_residuals(nwb, region, require_qc=True):
    """Out-of-fold Ridge residuals (sqrt(counts) scale) for all neurons in `region`, plus their unit ids."""
    trials = nwb.intervals["trials"].to_dataframe()
    trials = trials[~trials.is_instruction].reset_index(drop=True)
    bin_edges = trials.quiescent_stop_time.to_numpy()[:, None] + np.arange(-N_BINS, 1) * BIN

    units = nwb.units[:]
    if require_qc:
        units = units[units.default_qc]
    units = units[units.structure == region]
    unit_ids = units.index.to_numpy()
    units = units.reset_index(drop=True)

    counts = np.empty((len(trials), N_BINS, len(units)), dtype=np.int16)
    for u, st in enumerate(units.spike_times):
        counts[:, :, u] = np.diff(np.searchsorted(np.asarray(st), bin_edges), axis=1)

    X, reg_names = _design_matrix(nwb, trials, bin_edges)
    Y = np.sqrt(counts.reshape(len(trials) * N_BINS, len(units)).astype(float))

    bin_fold = _block_folds(trials)
    splits = [(np.flatnonzero(bin_fold != f), np.flatnonzero(bin_fold == f)) for f in range(N_FOLDS)]

    Y_hat = np.empty_like(Y)
    for f, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte, Ytr = sc.transform(X[tr]), sc.transform(X[te]), Y[tr]
        inner_sse = np.zeros((len(ALPHAS), len(units)))
        for g in set(range(N_FOLDS)) - {f}:
            val = bin_fold[tr] == g
            for a, alpha in enumerate(ALPHAS):
                m = Ridge(alpha=alpha).fit(Xtr[~val], Ytr[~val])
                inner_sse[a] += ((Ytr[val] - m.predict(Xtr[val])) ** 2).sum(0)
        best = inner_sse.argmin(axis=0)
        for a in np.unique(best):
            cols = np.flatnonzero(best == a)
            Y_hat[np.ix_(te, cols)] = Ridge(alpha=ALPHAS[a]).fit(Xtr, Ytr[:, cols]).predict(Xte).reshape(len(te), -1)

    return Y - Y_hat, unit_ids
