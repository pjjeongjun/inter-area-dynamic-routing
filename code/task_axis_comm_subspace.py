"""Task (context) axis vs. communication subspace, all ordered area pairs, one session.

Run as a script to analyse one session and write ``alignment_results.pkl`` to
``RESULTS_DIR/<session_id>/``::

    DATACUBE_ROOT=... SESSION_ID=743199_2024-12-05 RESULTS_DIR=../results \
        python task_axis_comm_subspace.py

Pipeline (see ``analyse_session`` for the exact order):

1. Stimulus trials only (catch trials dropped). Per-trial, per-unit firing rate over
   the quiescent window before stimulus onset, z-scored per unit over trials.
   Nuisance regression (``NUISANCE_REGRESSORS``, default previous-trial stimulus
   identity, response and reward plus running speed and pupil area): each unit's
   z-scored rate is regressed (OLS, with intercept) on the previous trial's stimulus
   (one-hot), whether the animal responded and whether it was rewarded, and on the
   z-scored mean running speed and pupil area in the quiescent window (missing pupil
   values filled with the session mean); the residual is re-z-scored. Trials with no
   previous trial are dropped. Set ``NUISANCE_REGRESSORS=none`` to skip. No condition mean is
   subtracted: the residual still carries the across-context mean difference.
2. Qualified cells: QC-pass single units (``is_qc_pass`` and ``decoder_label == 'sua'``).
   Areas with at least ``MIN_UNITS`` of them are included and every area is
   subsampled to exactly ``MIN_UNITS`` units, ``N_SUBSAMPLES`` times with different
   seeds (seed 0 reproduces the earlier notebooks' subsample). Every quantity below is
   computed per subsample and then averaged, so areas with many units and areas with
   barely enough are compared on equal footing.
3. Task axis: one LDA axis per area (context = rewarded modality, visual vs auditory
   block), unit-normalised in the area's z-scored unit space. Its predictive accuracy
   is leave-one-block-out cross-validated decoding accuracy, compared with a
   label-shuffle null.
4. Communication subspace: for every ordered pair (source -> target), ridge regression
   (alpha by efficient leave-one-out ``RidgeCV``) followed by reduced-rank regression
   (SVD of the fitted prediction). Predictive performance is the pooled 10-fold
   cross-validated R^2 as a function of rank; the dimensionality is the smallest rank
   whose CV R^2 is within one SEM of the best rank (Semedo et al., 2019). The
   subspace is the orthonormalised span of the first ``d`` predictive source
   directions (and, on the target side, the first ``d`` predicted target directions).
   Significance of the full-rank R^2 comes from a trial-shuffle null.
5. Alignment: cosine similarity = norm of the source area's unit task axis projected
   onto the pair's source-side communication subspace (1 = axis lies in the
   subspace, 0 = orthogonal). Chance is the random-axis null: a uniformly random unit
   vector in the same ``MIN_UNITS``-dimensional space projected onto the same
   subspace, which only depends on the subspace dimensionality. This is the only null
   used for alignment. Nulls are drawn per subsample and averaged across subsamples
   draw-wise, so they are nulls for the subsample-averaged statistic that the figures
   show.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, LeaveOneGroupOut

# ----------------------------------------------------------------------------- settings
MIN_UNITS = 30
NUISANCE_REGRESSORS = os.environ.get('NUISANCE_REGRESSORS', 'prev_stim,prev_response,prev_reward,running_speed,pupil_area')
N_SUBSAMPLES = 10
ALPHA_GRID = np.logspace(-1, 4, 25)
N_FOLDS = 10
N_R2_PERMUTATIONS = 1000
N_ACC_PERMUTATIONS = 200
N_RANDOM_AXES = 1000
FDR_ALPHA = 0.05

# frontal / prefrontal structures (Allen CCF acronyms) vs. everything else
FRONTAL_AREAS = {'MOs', 'PL', 'ILA', 'ACAd', 'ACAv', 'ORBl', 'ORBm', 'ORBvl', 'FRP', 'AId', 'AIv', 'AIp'}


def area_group(area: str) -> str:
    return 'frontal' if area in FRONTAL_AREAS else 'other'


def pair_type(source: str, target: str) -> str:
    return f'{area_group(source)}->{area_group(target)}'


# ----------------------------------------------------------------------------- data
def find_nwb(root: Path, session_id: str) -> Path:
    for d in sorted(root.iterdir()):
        candidate = d / f'{session_id}.nwb.zarr'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'no {session_id}.nwb.zarr under {root}')


def load_session(root: Path, session_id: str):
    import pynwb
    session = pynwb.read_nwb(find_nwb(root, session_id))
    trials = session.trials[:]
    units = session.units[:]
    return trials, units


def interval_counts(spike_times, starts, stops):
    st = np.sort(np.asarray(spike_times))
    return np.searchsorted(st, stops) - np.searchsorted(st, starts)


def quiescent_rates(trials: pd.DataFrame, units: pd.DataFrame):
    """Stimulus trials and the trials x units quiescent-window firing-rate matrix."""
    reg = trials[trials['stim_name'] != 'catch'].reset_index(drop=True)
    t0, t1 = reg['quiescent_start_time'].values, reg['quiescent_stop_time'].values
    rates = np.column_stack([interval_counts(st, t0, t1) / (t1 - t0) for st in units['spike_times']])
    return reg, pd.DataFrame(rates, columns=units.index, index=reg.index)


def window_means(ts, data, starts, stops, good=None):
    """Mean of a sampled signal within each [start, stop) window (NaN if no sample)."""
    ts = np.asarray(ts)
    data = np.asarray(data, dtype=float)
    ok = np.isfinite(ts) & np.isfinite(data)
    if good is not None:
        ok &= np.asarray(good, dtype=bool)
    ts, data = ts[ok], data[ok]
    order = np.argsort(ts)
    ts, data = ts[order], data[order]
    csum = np.concatenate([[0.0], np.cumsum(data)])
    i0, i1 = np.searchsorted(ts, starts), np.searchsorted(ts, stops)
    n = i1 - i0
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (csum[i1] - csum[i0]) / n
    out[n == 0] = np.nan
    return out


def behaviour_variables(session, reg: pd.DataFrame):
    """Per-trial quiescent-window means of running speed and pupil area (and trial index)."""
    beh = session.processing['behavior']
    t0, t1 = reg['quiescent_start_time'].values, reg['quiescent_stop_time'].values
    out = {'trial_index': reg['trial_index'].values.astype(float)}
    run = beh['running_speed']
    out['running_speed'] = window_means(run.timestamps[:], run.data[:], t0, t1)
    if 'eye_tracking' in beh.data_interfaces:           # some sessions have no usable eye tracking
        eye = beh['eye_tracking']
        out['pupil_area'] = window_means(eye['timestamps'][:], eye['pupil_area'][:], t0, t1,
                                         good=~np.asarray(eye['pupil_is_bad_frame'][:], dtype=bool))
    else:
        out['pupil_area'] = np.full(len(reg), np.nan)
    return pd.DataFrame(out, index=reg.index)


def nuisance_design(trials: pd.DataFrame, reg: pd.DataFrame, regressors, behaviour: pd.DataFrame | None = None):
    """Design matrix (one row per reg trial) of previous-trial covariates, and a mask of
    reg trials that have a previous trial. Previous trial = the trial with
    trial_index - 1 in the full trial table (catch and instruction trials included)."""
    by_index = trials.set_index('trial_index')
    prev_idx = reg['trial_index'].values - 1
    has_prev = np.isin(prev_idx, by_index.index.values)
    prev = by_index.reindex(prev_idx)
    cols = {}
    if 'prev_stim' in regressors:
        levels = sorted(trials['stim_name'].dropna().unique())
        for lev in levels[1:]:                       # first level is the reference
            cols[f'prev_stim_{lev}'] = (prev['stim_name'].values == lev).astype(float)
    if 'prev_response' in regressors:
        cols['prev_response'] = prev['is_response'].values.astype(float)
    if 'prev_reward' in regressors:
        cols['prev_reward'] = prev['is_rewarded'].values.astype(float)
    for v in ('running_speed', 'pupil_area', 'trial_index'):
        if v in regressors:
            x = behaviour[v].values.astype(float)
            if np.isfinite(x).sum() < 2 or np.nanstd(x) == 0:          # signal absent in this session: skip it
                print(f'nuisance regressor {v} unavailable in this session; skipped')
                continue
            x = np.where(np.isfinite(x), x, np.nanmean(x))          # fill missing with the session mean
            cols[v] = (x - x.mean()) / x.std()
    X = pd.DataFrame(cols, index=reg.index)
    return X, has_prev


def residualize(Y, X):
    """OLS residual of each column of Y on X (with intercept); also the per-column R^2."""
    Xd = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    resid = Y - Xd @ beta
    r2 = 1 - resid.var(axis=0) / Y.var(axis=0)
    return resid, r2


def zscore(a):
    a = np.asarray(a, dtype=float)
    sd = a.std(axis=0)
    sd[sd == 0] = 1.0
    return (a - a.mean(axis=0)) / sd


# ----------------------------------------------------------------------------- task axis
def lda_axis(X, labels):
    lda = LinearDiscriminantAnalysis(solver='svd').fit(X, labels)
    w = lda.coef_.ravel()
    return w / np.linalg.norm(w), lda.score(X, labels)


def axis_r2(X, w, labels):
    """In-sample R² of the task axis: fraction of the variance of the projection X @ w
    that is explained by the context label (between-context / total variance)."""
    proj = X @ w
    total = np.sum((proj - proj.mean()) ** 2)
    within = sum(np.sum((proj[labels == c] - proj[labels == c].mean()) ** 2) for c in np.unique(labels))
    return 1 - within / total


def lda_lobo_accuracy(X, labels, blocks):
    correct = 0
    for tr, te in LeaveOneGroupOut().split(X, labels, blocks):
        lda = LinearDiscriminantAnalysis(solver='svd').fit(X[tr], labels[tr])
        correct += (lda.predict(X[te]) == labels[te]).sum()
    return correct / len(labels)


# ----------------------------------------------------------------------------- comm subspace
def ridge_coef(X, Y, alpha):
    n = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(n), X.T @ Y)


def select_alpha(X, Y):
    cv = RidgeCV(alphas=ALPHA_GRID, alpha_per_target=False, scoring='r2', fit_intercept=False).fit(X, Y)
    return float(cv.alpha_)


def cv_rrr_r2(X, Y, alpha, folds, max_rank=None):
    """Pooled K-fold CV R^2 of reduced-rank ridge regression for ranks 1..max_rank.

    Returns (r2_by_rank[K, R], r2_full[K]) per fold, where rank R = full ridge.
    The reduced-rank model at rank r keeps the first r predictive dimensions of the
    training-set prediction (SVD of X_train @ B)."""
    n_src = X.shape[1]
    R = n_src if max_rank is None else max_rank
    r2_rank = np.empty((len(folds), R))
    r2_full = np.empty(len(folds))
    for k, (tr, te) in enumerate(folds):
        xm, ym = X[tr].mean(axis=0), Y[tr].mean(axis=0)
        Xtr, Ytr, Xte, Yte = X[tr] - xm, Y[tr] - ym, X[te] - xm, Y[te] - ym
        B = ridge_coef(Xtr, Ytr, alpha)
        _, _, Vt = np.linalg.svd(Xtr @ B, full_matrices=False)
        sst = np.sum(Yte ** 2)
        pred_full = Xte @ B
        r2_full[k] = 1 - np.sum((Yte - pred_full) ** 2) / sst
        # cumulative rank-r predictions: pred_r = pred_full @ V_r V_r^T
        proj = pred_full @ Vt.T                       # coordinates on predictive dims
        pred = np.zeros_like(Yte)
        for r in range(R):
            pred += np.outer(proj[:, r], Vt[r])
            r2_rank[k, r] = 1 - np.sum((Yte - pred) ** 2) / sst
    return r2_rank, r2_full


def dimensionality_1sem(r2_rank):
    """Smallest rank whose mean CV R^2 is within 1 SEM (over folds) of the best rank."""
    mean = r2_rank.mean(axis=0)
    best = int(np.argmax(mean))
    sem = r2_rank[:, best].std(ddof=1) / np.sqrt(r2_rank.shape[0])
    return int(np.argmax(mean >= mean[best] - sem)) + 1


def insample_rank_r2(X, Y, alpha, d):
    """In-sample R² (pooled over target units) of the rank-d reduced-rank ridge fit on all trials."""
    B = ridge_coef(X, Y, alpha)
    F = X @ B
    _, _, Vt = np.linalg.svd(F, full_matrices=False)
    pred = F @ Vt[:d].T @ Vt[:d]
    return 1 - np.sum((Y - pred) ** 2) / np.sum(Y ** 2)


def comm_subspace(X, Y, alpha, d):
    """Orthonormal source-side (n_src x d) and target-side (n_tgt x d) bases of the
    rank-d communication subspace fit on all trials."""
    B = ridge_coef(X, Y, alpha)
    _, _, Vt = np.linalg.svd(X @ B, full_matrices=False)
    Q_src, _ = np.linalg.qr(B @ Vt[:d].T)
    return Q_src[:, :d], Vt[:d].T


def cv_full_r2(X, Y, alpha, folds):
    """Pooled K-fold CV R^2 of the full ridge model."""
    sse = sst = 0.0
    for tr, te in folds:
        xm, ym = X[tr].mean(axis=0), Y[tr].mean(axis=0)
        B = ridge_coef(X[tr] - xm, Y[tr] - ym, alpha)
        Yte = Y[te] - ym
        sse += np.sum((Yte - (X[te] - xm) @ B) ** 2)
        sst += np.sum(Yte ** 2)
    return 1 - sse / sst


# ----------------------------------------------------------------------------- stats helpers
def fdr_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.clip(np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def p_two_sided(null, observed):
    n = len(null)
    hi = (np.sum(null >= observed) + 1) / (n + 1)
    lo = (np.sum(null <= observed) + 1) / (n + 1)
    return min(1.0, 2 * min(hi, lo))


def p_one_sided(null, observed):
    return (np.sum(null >= observed) + 1) / (len(null) + 1)


# ----------------------------------------------------------------------------- main analysis
def preprocess(session, trials, units, log=print):
    """Stimulus trials, quiescent-window rates, nuisance regression. Returns
    (reg_trials, rate_matrix [z-scored residuals], regressor names, per-unit nuisance R^2 or None)."""
    reg, fr = quiescent_rates(trials, units)
    regressors = [r for r in NUISANCE_REGRESSORS.split(',') if r and r != 'none']
    nuisance_r2_unit = None
    if regressors:
        behaviour = behaviour_variables(session, reg) if session is not None else None
        X_nuis, has_prev = nuisance_design(trials, reg, regressors, behaviour)
        reg, fr, X_nuis = reg[has_prev].reset_index(drop=True), fr[has_prev].reset_index(drop=True), X_nuis[has_prev]
        resid, nuisance_r2_unit = residualize(zscore(fr.values), X_nuis.values)
        fr = pd.DataFrame(zscore(resid), columns=fr.columns, index=fr.index)
        n_missing = int(behaviour.loc[has_prev, 'pupil_area'].isna().sum()) if behaviour is not None else 0
        log(f'nuisance regression: {list(X_nuis.columns)}; {int((~has_prev).sum())} trial(s) without a previous trial '
            f'dropped; {n_missing} trial(s) with missing pupil filled with the mean')
    return reg, fr, regressors, nuisance_r2_unit


def analyse_session(trials, units, session_id, min_units=MIN_UNITS, n_subsamples=N_SUBSAMPLES, verbose=True, session=None):
    t_start = time.time()
    log = print if verbose else (lambda *a, **k: None)

    reg, fr, regressors, nuisance_r2_unit = preprocess(session, trials, units, log)
    context = reg['rewarded_modality'].values
    blocks = reg['block_index'].values
    log(f'{session_id}: {len(reg)} stimulus trials, {len(np.unique(blocks))} blocks, '
        f'quiescent window {np.median(reg["quiescent_stop_time"] - reg["quiescent_start_time"]):.2f} s (median)')

    sua = units[units['is_qc_pass'] & (units['decoder_label'] == 'sua')]
    counts = sua['structure'].value_counts()
    areas = counts[counts >= min_units].index.tolist()
    if len(areas) < 2:
        raise RuntimeError(f'{session_id}: only {len(areas)} area(s) with >= {min_units} units')
    pairs = [(s, t) for s in areas for t in areas if s != t]
    n_pairs = len(pairs)
    log(f'{len(areas)} areas with >= {min_units} QC-pass SUA units: '
        + ', '.join(f'{a} ({counts[a]})' for a in areas) + f'; {n_pairs} ordered pairs')

    # ---- subsamples (seed k = k; k = 0 reproduces the earlier notebooks' subsample)
    subsamples = []
    for k in range(n_subsamples):
        rng = npr.default_rng(k)
        subsamples.append({a: rng.choice(sua[sua['structure'] == a].index.values, size=min_units, replace=False)
                           for a in areas})

    folds = list(KFold(N_FOLDS, shuffle=True, random_state=0).split(np.arange(len(reg))))
    n = min_units

    # containers: per subsample
    K = n_subsamples
    acc_train = np.empty((K, len(areas)))
    task_r2 = np.empty((K, len(areas)))
    acc_cv = np.empty((K, len(areas)))
    acc_null = np.empty((K, len(areas), N_ACC_PERMUTATIONS))
    axes_ = np.empty((K, len(areas), n))
    proj_example = {}                                      # (k=0) per-trial projections on the axis per area
    alpha = np.empty((K, n_pairs))
    r2_curve = np.empty((K, n_pairs, n))                   # mean over folds
    r2_curve_sem = np.empty((K, n_pairs, n))
    r2_full = np.empty((K, n_pairs))
    r2_train_dim = np.empty((K, n_pairs))
    r2_cv_dim = np.empty((K, n_pairs))
    r2_null = np.empty((K, n_pairs, N_R2_PERMUTATIONS))
    dims = np.empty((K, n_pairs), dtype=int)
    cos_src = np.empty((K, n_pairs))
    cos_tgt = np.empty((K, n_pairs))
    cos_null_random = np.empty((K, n_pairs, N_RANDOM_AXES))
    cos_tgt_null_random = np.empty((K, n_pairs, N_RANDOM_AXES))
    Q_src_all = [[None] * n_pairs for _ in range(K)]

    for k in range(K):
        tk = time.time()
        act = {a: zscore(fr[subsamples[k][a]].values) for a in areas}

        # ---- task axis per area
        lda_rng = npr.default_rng(1000 + k)
        for i, a in enumerate(areas):
            axes_[k, i], acc_train[k, i] = lda_axis(act[a], context)
            task_r2[k, i] = axis_r2(act[a], axes_[k, i], context)
            acc_cv[k, i] = lda_lobo_accuracy(act[a], context, blocks)
            for j in range(N_ACC_PERMUTATIONS):
                acc_null[k, i, j] = lda_lobo_accuracy(act[a], lda_rng.permutation(context), blocks)
            if k == 0:
                proj_example[a] = act[a] @ axes_[k, i]
        rand_rng = npr.default_rng(2000 + k)
        random_axes = rand_rng.standard_normal((N_RANDOM_AXES, n))
        random_axes /= np.linalg.norm(random_axes, axis=1, keepdims=True)

        # ---- communication subspaces
        perm_rng = npr.default_rng(3000 + k)
        for p, (s, t) in enumerate(pairs):
            X, Y = act[s], act[t]
            alpha[k, p] = select_alpha(X, Y)
            r2_rank, _ = cv_rrr_r2(X, Y, alpha[k, p], folds)
            r2_curve[k, p] = r2_rank.mean(axis=0)
            r2_curve_sem[k, p] = r2_rank.std(axis=0, ddof=1) / np.sqrt(N_FOLDS)
            r2_full[k, p] = cv_full_r2(X, Y, alpha[k, p], folds)
            dims[k, p] = dimensionality_1sem(r2_rank)
            r2_cv_dim[k, p] = r2_rank[:, dims[k, p] - 1].mean()
            r2_train_dim[k, p] = insample_rank_r2(X, Y, alpha[k, p], dims[k, p])
            Q_src, Q_tgt = comm_subspace(X, Y, alpha[k, p], dims[k, p])
            Q_src_all[k][p] = Q_src
            for j in range(N_R2_PERMUTATIONS):
                r2_null[k, p, j] = cv_full_r2(X, Y[perm_rng.permutation(len(Y))], alpha[k, p], folds)
            ws, wt = axes_[k, areas.index(s)], axes_[k, areas.index(t)]
            cos_src[k, p] = np.linalg.norm(Q_src.T @ ws)
            cos_tgt[k, p] = np.linalg.norm(Q_tgt.T @ wt)
            cos_null_random[k, p] = np.linalg.norm(random_axes @ Q_src, axis=1)
            cos_tgt_null_random[k, p] = np.linalg.norm(random_axes @ Q_tgt, axis=1)
        log(f'  subsample {k + 1}/{K} done in {time.time() - tk:.0f} s')

    # ---- aggregate over subsamples (nulls averaged draw-wise)
    def agg(x):
        return x.mean(axis=0)

    acc_null_m = agg(acc_null)
    r2_null_m = agg(r2_null)
    cos_rand_m = agg(cos_null_random)
    cos_tgt_rand_m = agg(cos_tgt_null_random)

    area_rows = []
    for i, a in enumerate(areas):
        unit_pos = [fr.columns.get_loc(u) for u in sua[sua['structure'] == a].index]
        area_rows.append({
            'nuisance_r2': float(np.nanmean(nuisance_r2_unit[unit_pos])) if nuisance_r2_unit is not None else 0.0,
            'session': session_id, 'area': a, 'group': area_group(a), 'n_units_available': int(counts[a]),
            'acc_train': acc_train[:, i].mean(), 'acc_cv': acc_cv[:, i].mean(), 'acc_cv_sd': acc_cv[:, i].std(ddof=1),
            'task_r2': task_r2[:, i].mean(), 'task_r2_sd': task_r2[:, i].std(ddof=1),
            'acc_null_mean': acc_null_m[i].mean(), 'acc_null_p': p_one_sided(acc_null_m[i], acc_cv[:, i].mean()),
            'acc_null_q975': np.quantile(acc_null_m[i], 0.975),
        })
    area_table = pd.DataFrame(area_rows)

    pair_rows = []
    for p, (s, t) in enumerate(pairs):
        obs_src, obs_tgt, obs_r2 = cos_src[:, p].mean(), cos_tgt[:, p].mean(), r2_full[:, p].mean()
        pair_rows.append({
            'session': session_id, 'source': s, 'target': t, 'pair_type': pair_type(s, t),
            'alpha': np.median(alpha[:, p]),
            'r2': obs_r2, 'r2_sd': r2_full[:, p].std(ddof=1),
            'r2_train_dim': r2_train_dim[:, p].mean(), 'r2_train_dim_sd': r2_train_dim[:, p].std(ddof=1),
            'r2_cv_dim': r2_cv_dim[:, p].mean(), 'r2_cv_dim_sd': r2_cv_dim[:, p].std(ddof=1),
            'r2_null_mean': r2_null_m[p].mean(), 'r2_p': p_one_sided(r2_null_m[p], obs_r2),
            'dim': dims[:, p].mean(), 'dim_sd': dims[:, p].std(ddof=1),
            'cos': obs_src, 'cos_sd': cos_src[:, p].std(ddof=1),
            'cos_chance': cos_rand_m[p].mean(),
            'cos_chance_lo': np.quantile(cos_rand_m[p], 0.025), 'cos_chance_hi': np.quantile(cos_rand_m[p], 0.975),
            'cos_z_random': (obs_src - cos_rand_m[p].mean()) / cos_rand_m[p].std(ddof=1),
            'cos_p_random': p_two_sided(cos_rand_m[p], obs_src),
            'cos_tgt': obs_tgt, 'cos_tgt_sd': cos_tgt[:, p].std(ddof=1),
            'cos_tgt_chance': cos_tgt_rand_m[p].mean(),
            'cos_tgt_z_random': (obs_tgt - cos_tgt_rand_m[p].mean()) / cos_tgt_rand_m[p].std(ddof=1),
            'cos_tgt_p_random': p_two_sided(cos_tgt_rand_m[p], obs_tgt),
        })
    pair_table = pd.DataFrame(pair_rows)
    for col in ['r2_p', 'cos_p_random', 'cos_tgt_p_random']:
        pair_table[col.replace('_p', '_q')] = fdr_bh(pair_table[col].values)
    pair_table['angle_deg'] = np.degrees(np.arccos(np.clip(pair_table['cos'], 0, 1)))
    pair_table['angle_chance_deg'] = np.degrees(np.arccos(np.clip(pair_table['cos_chance'], 0, 1)))

    log(f'{session_id}: finished in {(time.time() - t_start) / 60:.1f} min')
    return {
        'session_id': session_id, 'areas': areas, 'pairs': pairs, 'min_units': min_units,
        'n_subsamples': n_subsamples, 'n_trials': len(reg), 'n_blocks': int(len(np.unique(blocks))),
        'nuisance_regressors': regressors,
        'context': context, 'blocks': blocks, 'trial_index': reg['trial_index'].values,
        'unit_counts': {a: int(counts[a]) for a in areas},
        'area_table': area_table, 'pair_table': pair_table,
        'acc_cv': acc_cv, 'acc_train': acc_train, 'acc_null_mean': acc_null_m, 'task_r2': task_r2,
        'r2_train_dim': r2_train_dim, 'r2_cv_dim': r2_cv_dim,
        'axes': axes_, 'proj_example': proj_example,
        'r2_curve': r2_curve, 'r2_curve_sem': r2_curve_sem, 'r2_full': r2_full, 'dims': dims,
        'cos_src': cos_src, 'cos_tgt': cos_tgt,
        'cos_null_random_mean': cos_rand_m,
        'Q_src_subsample0': Q_src_all[0],
    }


def main():
    code_dir = Path(__file__).resolve().parent
    capsule_root = Path('/root/capsule/data/dynamicrouting_datacube')
    local_root = code_dir.parent / 'data' / 'dynamicrouting_datacube'
    root = Path(os.environ.get('DATACUBE_ROOT', capsule_root if capsule_root.exists() else local_root))
    session_id = os.environ.get('SESSION_ID', '743199_2024-12-05')
    results = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results')) / session_id
    results.mkdir(parents=True, exist_ok=True)
    import pynwb
    session = pynwb.read_nwb(find_nwb(root, session_id))
    out = analyse_session(session.trials[:], session.units[:], session_id, session=session)
    with open(results / 'alignment_results.pkl', 'wb') as f:
        pickle.dump(out, f)
    out['area_table'].to_csv(results / 'task_axis_by_area.csv', index=False)
    out['pair_table'].to_csv(results / 'alignment_by_pair.csv', index=False)
    print(f'wrote {results / "alignment_results.pkl"}')


if __name__ == '__main__':
    sys.exit(main())
