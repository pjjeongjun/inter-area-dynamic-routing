"""Add context decoding from the source area's *private* subspace (the directions of the MIN_UNITS-unit source
space orthogonal to the communication subspace) to existing alignment_results.pkl files, per pair, with the
nearest-centroid reader, folds and shared permutation draws of patch_cv_quality.py's sub_acc_nc_*.

In each fold the rank-d communication subspace Q (MIN_UNITS x d) is fit on the training trials (as for
sub_acc_nc_cv); its orthogonal complement C (MIN_UNITS x (MIN_UNITS - d)) is the private subspace.

  priv_full_acc_nc_* per pair: nearest centroid on all MIN_UNITS - d private coordinates.
  priv_d_acc_nc_*    per pair: nearest centroid on the d private directions of largest training-set variance
                     (top d principal components of the activity projected onto C), so that the comparison with
                     sub_acc_nc_cv is d against d dimensions.

Each has the trial-shuffle null (`_null_mean`, `_p`) and the block-permutation null (`_block_null_*`,
`_block_p`, `_predictive` = above every block-permutation draw), computed as for sub_acc_nc_*.

    DATACUBE_ROOT=... RESULTS_DIR=... python patch_private_nc.py [session ...]
"""
import os, pickle, sys, time
from pathlib import Path
import numpy as np, numpy.random as npr, pynwb
import task_axis_comm_subspace as m
from patch_cv_quality import cv_accuracy_from

VARIANTS = ('priv_full', 'priv_d')


def private_projections(X, Y, alpha, d, folds):
    """Per fold, the coordinates of the communication subspace and of the two private-subspace variants:
    {'sub': [...], 'priv_full': [...], 'priv_d': [...]}, each a list of (train idx, test idx, train, test)."""
    out = {'sub': [], 'priv_full': [], 'priv_d': []}
    for tr, te in folds:
        xm, ym = X[tr].mean(axis=0), Y[tr].mean(axis=0)
        Xtr, Xte = X[tr] - xm, X[te] - xm
        Q, _ = m.comm_subspace(Xtr, Y[tr] - ym, alpha, d)
        C = np.linalg.qr(Q, mode='complete')[0][:, d:]                 # orthonormal complement of Q
        Ptr, Pte = Xtr @ C, Xte @ C
        V = np.linalg.svd(Ptr, full_matrices=False)[2][:d].T            # top-d variance directions within C
        out['sub'].append((tr, te, Xtr @ Q, Xte @ Q))
        out['priv_full'].append((tr, te, Ptr, Pte))
        out['priv_d'].append((tr, te, Ptr @ V, Pte @ V))
    return out


def patch(session_dir: Path, root: Path):
    t0 = time.time()
    with open(session_dir / 'alignment_results.pkl', 'rb') as f:
        res = pickle.load(f)
    sid, areas, pairs, K = res['session_id'], res['areas'], res['pairs'], res['n_subsamples']
    m.NUISANCE_REGRESSORS = ','.join(res.get('nuisance_regressors', [])) or 'none'
    if res.get('context_cv'):
        m.CONTEXT_CV = res['context_cv']                                # same folds as the stored sub_acc_nc_cv
    session = pynwb.read_nwb(m.find_nwb(root, sid))
    trials, units = session.trials[:], session.units[:]
    reg, fr, _, _ = m.preprocess(session, trials, units, log=lambda *a, **k: None)
    assert len(reg) == res['n_trials'], (len(reg), res['n_trials'])
    context, blocks = reg['rewarded_modality'].values, reg['block_index'].values
    assert (context == res['context']).all() and (blocks == res['blocks']).all()
    sua = units[units['is_qc_pass'] & (units['decoder_label'] == 'sua')]
    P = len(pairs)
    label_perms = m.label_permutations(context, m.N_ACC_PERMUTATIONS, seed=4000)   # same draws as patch_cv_quality
    block_perms = m.block_label_permutations(context, blocks)
    n_block = len(block_perms)
    folds_ctx = m.context_folds(context, blocks)
    block_folds = [m.context_folds(lab, blocks) for lab in block_perms]
    acc = {v: np.empty((K, P)) for v in VARIANTS}
    null = {v: np.empty((K, P, m.N_ACC_PERMUTATIONS)) for v in VARIANTS}
    block = {v: np.empty((K, P, n_block)) for v in VARIANTS}
    for k in range(K):
        tk = time.time()
        rng = npr.default_rng(k)
        sub = {a: rng.choice(sua[sua['structure'] == a].index.values, size=m.MIN_UNITS, replace=False) for a in areas}
        act = {a: m.zscore(fr[sub[a]].values) for a in areas}
        for p, (s, t) in enumerate(pairs):
            X, Y = act[s], act[t]
            alpha, d = m.select_alpha(X, Y), int(res['dims'][k, p])
            proj = private_projections(X, Y, alpha, d, folds_ctx)
            if p == 0 and 'sub_acc_nc_cv' in res:     # the rebuilt subspace must reproduce the stored accuracy
                chk = cv_accuracy_from(proj['sub'], context, 'nc')
                assert abs(chk - res['sub_acc_nc_cv'][k, 0]) < 1e-9, (k, chk, res['sub_acc_nc_cv'][k, 0])
            for v in VARIANTS:
                acc[v][k, p] = cv_accuracy_from(proj[v], context, 'nc')
                for j in range(m.N_ACC_PERMUTATIONS):
                    null[v][k, p, j] = cv_accuracy_from(proj[v], label_perms[j], 'nc')
            for j in range(n_block):     # the block-permuted labels define their own balanced folds
                proj_b = private_projections(X, Y, alpha, d, block_folds[j])
                for v in VARIANTS:
                    block[v][k, p, j] = cv_accuracy_from(proj_b[v], block_perms[j], 'nc')
        print(f'{sid}: subsample {k + 1}/{K} in {time.time() - tk:.0f} s', flush=True)
    pt = res['pair_table']
    for v in VARIANTS:
        null_m, block_m = null[v].mean(axis=0), block[v].mean(axis=0)              # draw-wise over subsamples
        res[f'{v}_acc_nc_cv'], res[f'{v}_acc_nc_null_mean'], res[f'{v}_acc_nc_block_null_mean'] = acc[v], null_m, block_m
        obs = acc[v].mean(0)
        pt[f'{v}_acc_nc_cv'], pt[f'{v}_acc_nc_cv_sd'] = obs, acc[v].std(0, ddof=1)
        pt[f'{v}_acc_nc_null_mean'] = null_m.mean(axis=1)
        pt[f'{v}_acc_nc_p'] = [m.p_one_sided(null_m[p], obs[p]) for p in range(P)]
        pt[f'{v}_acc_nc_block_null_mean'] = block_m.mean(axis=1)
        pt[f'{v}_acc_nc_block_null_lo'], pt[f'{v}_acc_nc_block_null_hi'] = block_m.min(axis=1), block_m.max(axis=1)
        pt[f'{v}_acc_nc_block_p'] = [m.p_one_sided(block_m[p], obs[p]) for p in range(P)]
        pt[f'{v}_acc_nc_predictive'] = obs > pt[f'{v}_acc_nc_block_null_hi'].values
    with open(session_dir / 'alignment_results.pkl', 'wb') as f:
        pickle.dump(res, f)
    pt.to_csv(session_dir / 'alignment_by_pair.csv', index=False)
    print(f'{session_dir.parent.name}/{sid}: nearest centroid, comm. subspace {pt["sub_acc_nc_cv"].mean():.3f}, '
          f'private d-matched {pt["priv_d_acc_nc_cv"].mean():.3f} '
          f'({int(pt["priv_d_acc_nc_predictive"].sum())}/{P} above the block null), '
          f'private full {pt["priv_full_acc_nc_cv"].mean():.3f} '
          f'({int(pt["priv_full_acc_nc_predictive"].sum())}/{P}); {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    code_dir = Path(__file__).resolve().parent
    root = Path(os.environ.get('DATACUBE_ROOT', code_dir.parent / 'data' / 'dynamicrouting_datacube'))
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results_3sessions'))
    sessions = sys.argv[1:] or sorted(p.parent.name for p in results_dir.glob('*/alignment_results.pkl'))
    for sid in sessions:
        patch(results_dir / sid, root)
