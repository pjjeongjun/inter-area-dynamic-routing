"""Add the nearest-centroid reader on the full population of each area (``acc_nc_*``) to existing
alignment_results.pkl files, without recomputing the communication-subspace quantities (which take hours).
Same preprocessing, unit subsamples, balanced block folds and shared permutation draws as patch_cv_quality.py,
whose per-area block is reproduced here; running patch_cv_quality.py gives identical acc_nc_* values.

    DATACUBE_ROOT=... RESULTS_DIR=... python patch_area_nc.py [session ...]
"""
import os, pickle, sys, time
from pathlib import Path
import numpy as np, numpy.random as npr, pynwb
import task_axis_comm_subspace as m
from patch_cv_quality import cv_accuracy_from


def patch(session_dir: Path, root: Path):
    t0 = time.time()
    with open(session_dir / 'alignment_results.pkl', 'rb') as f:
        res = pickle.load(f)
    sid, areas, K = res['session_id'], res['areas'], res['n_subsamples']
    m.NUISANCE_REGRESSORS = ','.join(res.get('nuisance_regressors', [])) or 'none'
    session = pynwb.read_nwb(m.find_nwb(root, sid))
    trials, units = session.trials[:], session.units[:]
    reg, fr, _, _ = m.preprocess(session, trials, units, log=lambda *a, **k: None)
    assert len(reg) == res['n_trials'], (len(reg), res['n_trials'])
    context, blocks = reg['rewarded_modality'].values, reg['block_index'].values
    assert (context == res['context']).all() and (blocks == res['blocks']).all()
    sua = units[units['is_qc_pass'] & (units['decoder_label'] == 'sua')]
    label_perms = m.label_permutations(context, m.N_ACC_PERMUTATIONS, seed=4000)   # same draws as patch_cv_quality
    block_perms = m.block_label_permutations(context, blocks)
    n_block = len(block_perms)
    folds_ctx = m.context_folds(context, blocks)
    block_folds = [m.context_folds(lab, blocks) for lab in block_perms]
    A = len(areas)
    a_nc, a_nc_null, a_nc_block = np.empty((K, A)), np.empty((K, A, m.N_ACC_PERMUTATIONS)), np.empty((K, A, n_block))
    for k in range(K):
        rng = npr.default_rng(k)
        sub = {a: rng.choice(sua[sua['structure'] == a].index.values, size=m.MIN_UNITS, replace=False) for a in areas}
        act = {a: m.zscore(fr[sub[a]].values) for a in areas}
        chk = m.lda_cv_accuracy(act[areas[0]], context, folds_ctx)
        assert abs(chk - res['acc_cv'][k, 0]) < 1e-9, (k, chk, res['acc_cv'][k, 0])
        for i, a in enumerate(areas):
            X = act[a]
            full = [(tr, te, X[tr], X[te]) for tr, te in folds_ctx]
            a_nc[k, i] = cv_accuracy_from(full, context, 'nc')
            for j in range(m.N_ACC_PERMUTATIONS):
                a_nc_null[k, i, j] = cv_accuracy_from(full, label_perms[j], 'nc')
            for j in range(n_block):
                a_nc_block[k, i, j] = cv_accuracy_from([(tr, te, X[tr], X[te]) for tr, te in block_folds[j]], block_perms[j], 'nc')
    a_null_m, a_block_m = a_nc_null.mean(axis=0), a_nc_block.mean(axis=0)
    res['acc_nc_cv'], res['acc_nc_null_mean'], res['acc_nc_block_null_mean'] = a_nc, a_null_m, a_block_m
    at = res['area_table']
    obs_a = a_nc.mean(0)
    at['acc_nc_cv'], at['acc_nc_cv_sd'] = obs_a, a_nc.std(0, ddof=1)
    at['acc_nc_null_mean'] = a_null_m.mean(axis=1)
    at['acc_nc_null_p'] = [m.p_one_sided(a_null_m[i], obs_a[i]) for i in range(A)]
    at['acc_nc_block_null_mean'] = a_block_m.mean(axis=1)
    at['acc_nc_block_null_lo'], at['acc_nc_block_null_hi'] = a_block_m.min(axis=1), a_block_m.max(axis=1)
    at['acc_nc_block_p'] = [m.p_one_sided(a_block_m[i], obs_a[i]) for i in range(A)]
    at['acc_nc_predictive'] = obs_a > at['acc_nc_block_null_hi'].values
    with open(session_dir / 'alignment_results.pkl', 'wb') as f:
        pickle.dump(res, f)
    at.to_csv(session_dir / 'task_axis_by_area.csv', index=False)
    print(f'{session_dir.parent.name}/{sid}: population nearest-centroid {obs_a.mean():.3f} '
          f'({int(at["acc_nc_predictive"].sum())}/{A} areas above the block null; LDA {int(at["acc_predictive"].sum())}/{A}); '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    code_dir = Path(__file__).resolve().parent
    root = Path(os.environ.get('DATACUBE_ROOT', code_dir.parent / 'data' / 'dynamicrouting_datacube'))
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results_11sessions'))
    sessions = sys.argv[1:] or sorted(p.parent.name for p in results_dir.glob('*/alignment_results.pkl'))
    for sid in sessions:
        patch(results_dir / sid, root)
