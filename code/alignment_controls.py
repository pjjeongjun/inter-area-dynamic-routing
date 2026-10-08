"""Cross-block control for the task-axis / communication-subspace alignment, one session.

    DATACUBE_ROOT=... SESSION_ID=... RESULTS_DIR=... python alignment_controls.py

Same preprocessing as ``task_axis_comm_subspace.py`` (quiescent rates, nuisance
regression, QC-pass single units, 30-unit subsamples). Then, per subsample, the 6 blocks
are split into every 3-vs-3 partition in which both halves contain both contexts (18
ordered splits). The task axis (LDA) is fit on half A and the communication subspace
(ridge + RRR, 1-SEM dimensionality) on half B; the cosine between them is compared with
(a) the shuffled-label axis null for the same half-B subspace (an LDA axis fit on half A
with trial-shuffled labels, ``N_SHUF_CROSS`` draws; the significance null), (b) the
random-axis chance for the half-B subspace (reference) and (c) the "same-blocks" cosine,
i.e. the half-B axis in the half-B subspace. The half-A axis's decoding accuracy on half
B is also recorded. If the alignment comes from block-specific shared state rather than
a context direction that generalises, the cross-block cosine falls to the null while the
same-blocks cosine stays high. The full-data cosine is recomputed for reference.

As in the main analysis, every null draw (label permutation) is generated once per
session and applied to every unit subsample; nulls are averaged over subsamples draw-wise
and the random-axis chance is pooled.

Qualification (source task axis decodes context above the block-permutation null, leave-
one-block-out, as in ``task_axis_comm_subspace.py``) is recomputed here; the R² criterion
was met by every pair in the main analysis and is not repeated.
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import numpy.random as npr
import pandas as pd
import pynwb
from sklearn.model_selection import KFold

import task_axis_comm_subspace as m

N_RANDOM_AXES = 1000
N_ACC_PERMUTATIONS = 200
N_SHUF_CROSS = 200          # shuffled-label half-A axes per split


# ----------------------------------------------------------------------------- helpers
def fit_subspace(X, Y, folds):
    alpha = m.select_alpha(X, Y)
    r2_rank, _ = m.cv_rrr_r2(X, Y, alpha, folds)
    d = m.dimensionality_1sem(r2_rank)
    Q_src, _ = m.comm_subspace(X, Y, alpha, d)
    return Q_src, d


def block_splits(blocks, context):
    """Ordered (half_A_mask, half_B_mask) for every 3-vs-3 block partition with both
    contexts in both halves."""
    ids = np.unique(blocks)
    splits = []
    for a in itertools.combinations(ids, len(ids) // 2):
        mask_a = np.isin(blocks, a)
        mask_b = ~mask_a
        if len(np.unique(context[mask_a])) == 2 and len(np.unique(context[mask_b])) == 2:
            splits.append((mask_a, mask_b))
    return splits


# ----------------------------------------------------------------------------- main
def analyse_controls(session, trials, units, session_id, n_subsamples=m.N_SUBSAMPLES, verbose=True):
    t_start = time.time()
    log = print if verbose else (lambda *a, **k: None)

    reg, fr, regressors, _ = m.preprocess(session, trials, units, log)
    context = reg['rewarded_modality'].values
    blocks = reg['block_index'].values
    log(f'{session_id}: {len(reg)} trials')

    sua = units[units['is_qc_pass'] & (units['decoder_label'] == 'sua')]
    counts = sua['structure'].value_counts()
    areas = counts[counts >= m.MIN_UNITS].index.tolist()
    pairs = [(s, t) for s in areas for t in areas if s != t]
    n, K, n_pairs = m.MIN_UNITS, n_subsamples, len(pairs)
    subsamples = [{a: npr.default_rng(k).choice(sua[sua['structure'] == a].index.values, size=n, replace=False)
                   for a in areas} for k in range(K)]
    folds = list(KFold(m.N_FOLDS, shuffle=True, random_state=0).split(np.arange(len(reg))))
    splits = block_splits(blocks, context)
    log(f'{len(areas)} areas, {n_pairs} ordered pairs, {len(splits)} cross-block splits')

    # null draws, generated once and applied to every subsample
    label_perms = m.label_permutations(context, N_ACC_PERMUTATIONS, seed=1000)
    block_perms = m.block_label_permutations(context, blocks)
    n_block = len(block_perms)
    folds_ctx = m.context_block_folds(context, blocks)
    block_folds = [m.context_block_folds(lab, blocks) for lab in block_perms]
    # one trial-wise permutation of the full label vector per draw, restricted to each split's half A: a trial keeps the
    # same shuffled label in every split, so the null averaged over splits has the same across-split dependence as the
    # observed cross-block cosine (independent permutations per split would shrink the null, as per-subsample ones did)
    axis_perms = m.label_permutations(context, N_SHUF_CROSS, seed=7000)
    split_perms = [axis_perms[:, ma] for ma, mb in splits]

    A = len(areas)
    acc_cv = np.empty((K, A))
    acc_null = np.empty((K, A, N_ACC_PERMUTATIONS))
    acc_block_null = np.empty((K, A, n_block))
    cos_task = np.empty((K, n_pairs))
    null_full = np.empty((K, n_pairs, N_RANDOM_AXES))
    dims_full = np.empty((K, n_pairs))
    cos_cross = np.empty((K, n_pairs))
    cos_same = np.empty((K, n_pairs))
    null_cross = np.empty((K, n_pairs, N_RANDOM_AXES))
    shuf_cross = np.empty((K, n_pairs, N_SHUF_CROSS))
    dims_cross = np.empty((K, n_pairs))
    acc_cross = np.empty((K, A))

    for k in range(K):
        tk = time.time()
        act = {a: m.zscore(fr[subsamples[k][a]].values) for a in areas}
        rng = npr.default_rng(5000 + k)
        random_axes = rng.standard_normal((N_RANDOM_AXES, n))
        random_axes /= np.linalg.norm(random_axes, axis=1, keepdims=True)

        # task axis and qualification per area
        task_axes = {}
        for i, a in enumerate(areas):
            task_axes[a], _ = m.lda_axis(act[a], context)
            acc_cv[k, i] = m.lda_cv_accuracy(act[a], context, folds_ctx)
            for j in range(N_ACC_PERMUTATIONS):
                acc_null[k, i, j] = m.lda_cv_accuracy(act[a], label_perms[j], folds_ctx)
            for j in range(n_block):
                acc_block_null[k, i, j] = m.lda_cv_accuracy(act[a], block_perms[j], block_folds[j])

        # cross-block axes per area per split (axis on A, tested on B), and shuffled-label half-A axes
        axes_half, shuf_half = {}, {}
        acc_split = np.zeros(A)
        for si, (ma, mb) in enumerate(splits):
            for i, a in enumerate(areas):
                w, _ = m.lda_axis(act[a][ma], context[ma])
                axes_half[(si, a)] = w
                shuf_half[(si, a)] = m.lda_axes(act[a][ma], split_perms[si])
                lda = m.LinearDiscriminantAnalysis(solver='svd', priors=[0.5, 0.5]).fit(act[a][ma], context[ma])
                acc_split[i] += (lda.predict(act[a][mb]) == context[mb]).mean() / len(splits)
        acc_cross[k] = acc_split

        for p, (s, t) in enumerate(pairs):
            # full-data subspace: task axis and nuisance axes
            Q, d = fit_subspace(act[s], act[t], folds)
            dims_full[k, p] = d
            cos_task[k, p] = np.linalg.norm(Q.T @ task_axes[s])
            null_full[k, p] = np.linalg.norm(random_axes @ Q, axis=1)
            # cross-block: subspace on half B, axis from half A (and from half B for reference)
            cc = cs = dd = 0.0
            nn = np.zeros(N_RANDOM_AXES)
            ss = np.zeros(N_SHUF_CROSS)
            for si, (ma, mb) in enumerate(splits):
                idx_b = np.flatnonzero(mb)
                folds_b = list(KFold(m.N_FOLDS, shuffle=True, random_state=0).split(idx_b))
                Qb, db = fit_subspace(act[s][mb], act[t][mb], folds_b)
                cc += np.linalg.norm(Qb.T @ axes_half[(si, s)])
                # axis on the same half B: the complementary split has A = this B
                w_b, _ = m.lda_axis(act[s][mb], context[mb])
                cs += np.linalg.norm(Qb.T @ w_b)
                nn += np.linalg.norm(random_axes @ Qb, axis=1)
                ss += np.linalg.norm(shuf_half[(si, s)] @ Qb, axis=1)
                dd += db
            cos_cross[k, p] = cc / len(splits)
            cos_same[k, p] = cs / len(splits)
            null_cross[k, p] = nn / len(splits)
            shuf_cross[k, p] = ss / len(splits)
            dims_cross[k, p] = dd / len(splits)
        log(f'  subsample {k + 1}/{K} done in {time.time() - tk:.0f} s')

    # ---- aggregate: shuffled-label nulls averaged draw-wise over subsamples, random-axis chance pooled
    def pool(x):
        return np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)

    nf, nc, sc = pool(null_full), pool(null_cross), shuf_cross.mean(axis=0)
    acc_null_m, acc_block_m = acc_null.mean(axis=0), acc_block_null.mean(axis=0)
    area_rows = []
    for i, a in enumerate(areas):
        obs = acc_cv[:, i].mean()
        row = {'session': session_id, 'area': a, 'group': m.area_group(a), 'acc_cv': obs,
               'acc_null_q975': np.quantile(acc_null_m[i], 0.975),
               'acc_block_null_mean': acc_block_m[i].mean(), 'acc_block_null_hi': acc_block_m[i].max(),
               'acc_block_p': m.p_one_sided(acc_block_m[i], obs), 'acc_cross_block': acc_cross[:, i].mean()}
        row['axis_predictive'] = bool(obs > row['acc_block_null_hi'])
        area_rows.append(row)
    area_table = pd.DataFrame(area_rows)
    pred = dict(zip(area_table['area'], area_table['axis_predictive']))

    pair_rows = []
    for p, (s, t) in enumerate(pairs):
        obs_cross, obs_same = cos_cross[:, p].mean(), cos_same[:, p].mean()
        row = {'session': session_id, 'source': s, 'target': t, 'pair_type': m.pair_type(s, t),
               'qualified': bool(pred[s]), 'dim': dims_full[:, p].mean(),
               'cos_task': cos_task[:, p].mean(), 'cos_chance': nf[p].mean(),
               'cos_task_z_random': (cos_task[:, p].mean() - nf[p].mean()) / nf[p].std(ddof=1),
               'dim_cross': dims_cross[:, p].mean(),
               'cos_cross': obs_cross, 'cos_same': obs_same,
               # random-axis chance of the half-B subspace (reference)
               'cos_cross_chance': nc[p].mean(),
               'cos_cross_z_random': (obs_cross - nc[p].mean()) / nc[p].std(ddof=1),
               # shuffled-label axis null of the half-B subspace (significance)
               'cos_cross_null': sc[p].mean(), 'cos_cross_null_sd': sc[p].std(ddof=1),
               'cos_cross_null_lo': np.quantile(sc[p], 0.025), 'cos_cross_null_hi': np.quantile(sc[p], 0.975),
               'cos_cross_z': (obs_cross - sc[p].mean()) / sc[p].std(ddof=1),
               'cos_cross_p': m.p_two_sided(sc[p], obs_cross),
               'cos_same_z': (obs_same - sc[p].mean()) / sc[p].std(ddof=1)}
        pair_rows.append(row)
    pair_table = pd.DataFrame(pair_rows)
    q = pair_table['qualified'].values
    for col in ['cos_cross_p']:
        qcol = col[:-2] + '_q'
        pair_table[qcol] = np.nan
        if q.any():
            pair_table.loc[q, qcol] = m.fdr_bh(pair_table.loc[q, col].values)

    log(f'{session_id}: controls finished in {(time.time() - t_start) / 60:.1f} min')
    return {'session_id': session_id, 'areas': areas, 'pairs': pairs, 'n_splits': len(splits),
            'n_block_permutations': n_block, 'n_shuf_cross': N_SHUF_CROSS,
            'regressors': regressors, 'area_table': area_table, 'pair_table': pair_table}


def main():
    code_dir = Path(__file__).resolve().parent
    capsule_root = Path('/root/capsule/data/dynamicrouting_datacube')
    local_root = code_dir.parent / 'data' / 'dynamicrouting_datacube'
    root = Path(os.environ.get('DATACUBE_ROOT', capsule_root if capsule_root.exists() else local_root))
    session_id = os.environ.get('SESSION_ID', '743199_2024-12-05')
    results = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results_3sessions')) / session_id
    results.mkdir(parents=True, exist_ok=True)
    session = pynwb.read_nwb(m.find_nwb(root, session_id))
    out = analyse_controls(session, session.trials[:], session.units[:], session_id)
    with open(results / 'controls_results.pkl', 'wb') as f:
        pickle.dump(out, f)
    out['area_table'].to_csv(results / 'controls_by_area.csv', index=False)
    out['pair_table'].to_csv(results / 'controls_by_pair.csv', index=False)
    print(f'wrote {results / "controls_results.pkl"}')


if __name__ == '__main__':
    sys.exit(main())
