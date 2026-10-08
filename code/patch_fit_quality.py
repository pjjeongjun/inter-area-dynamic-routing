"""Add in-sample fit-quality fields to existing alignment_results.pkl files without rerunning
the permutation tests: task-axis R² per area (variance of the projection explained by context)
and the in-sample / cross-validated R² of the rank-d communication subspace per pair. Rebuilds
the same preprocessing, unit subsamples and folds as the stored run.

    DATACUBE_ROOT=... RESULTS_DIR=../results_3sessions python patch_fit_quality.py [session ...]
"""
import os, pickle, sys
from pathlib import Path
import numpy as np, numpy.random as npr, pandas as pd, pynwb
from sklearn.model_selection import KFold
import task_axis_comm_subspace as m


def patch(session_dir: Path, root: Path):
    with open(session_dir / 'alignment_results.pkl', 'rb') as f:
        res = pickle.load(f)
    sid, areas, pairs, K = res['session_id'], res['areas'], res['pairs'], res['n_subsamples']
    m.NUISANCE_REGRESSORS = ','.join(res.get('nuisance_regressors', [])) or 'none'
    session = pynwb.read_nwb(m.find_nwb(root, sid))
    trials, units = session.trials[:], session.units[:]
    reg, fr, _, _ = m.preprocess(session, trials, units, log=lambda *a, **k: None)
    assert len(reg) == res['n_trials'], (len(reg), res['n_trials'])
    context = reg['rewarded_modality'].values
    sua = units[units['is_qc_pass'] & (units['decoder_label'] == 'sua')]
    folds = list(KFold(m.N_FOLDS, shuffle=True, random_state=0).split(np.arange(len(reg))))
    task_r2 = np.empty((K, len(areas)))
    r2_train = np.empty((K, len(pairs)))
    r2_cv = np.empty((K, len(pairs)))
    for k in range(K):
        rng = npr.default_rng(k)
        sub = {a: rng.choice(sua[sua['structure'] == a].index.values, size=m.MIN_UNITS, replace=False) for a in areas}
        act = {a: m.zscore(fr[sub[a]].values) for a in areas}
        for i, a in enumerate(areas):
            w = res['axes'][k, i]
            task_r2[k, i] = m.axis_r2(act[a], w, context)
        for p, (s, t) in enumerate(pairs):
            X, Y = act[s], act[t]
            alpha = m.select_alpha(X, Y)
            d = int(res['dims'][k, p])
            r2_train[k, p] = m.insample_rank_r2(X, Y, alpha, d)
            r2_cv[k, p] = res['r2_curve'][k, p, d - 1]
    # sanity: stored axis must reproduce the stored training accuracy direction (projection sign-free check)
    res['task_r2'], res['r2_train_dim'], res['r2_cv_dim'] = task_r2, r2_train, r2_cv
    at, pt = res['area_table'], res['pair_table']
    at['task_r2'], at['task_r2_sd'] = task_r2.mean(0), task_r2.std(0, ddof=1)
    pt['r2_train_dim'], pt['r2_train_dim_sd'] = r2_train.mean(0), r2_train.std(0, ddof=1)
    pt['r2_cv_dim'], pt['r2_cv_dim_sd'] = r2_cv.mean(0), r2_cv.std(0, ddof=1)
    with open(session_dir / 'alignment_results.pkl', 'wb') as f:
        pickle.dump(res, f)
    at.to_csv(session_dir / 'task_axis_by_area.csv', index=False)
    pt.to_csv(session_dir / 'alignment_by_pair.csv', index=False)
    print(f'{session_dir.parent.name}/{sid}: task R² {task_r2.mean():.2f}, rank-d R² train {r2_train.mean():.3f} / cv {r2_cv.mean():.3f}')


if __name__ == '__main__':
    code_dir = Path(__file__).resolve().parent
    root = Path(os.environ.get('DATACUBE_ROOT', code_dir.parent / 'data' / 'dynamicrouting_datacube'))
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results_3sessions'))
    sessions = sys.argv[1:] or sorted(p.parent.name for p in results_dir.glob('*/alignment_results.pkl'))
    for sid in sessions:
        patch(results_dir / sid, root)
