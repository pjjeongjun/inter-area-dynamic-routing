"""Add the leave-one-block-out R² of the rank-d communication subspace to existing alignment_results.pkl files,
rebuilding the same preprocessing and unit subsamples as the stored run:

  r2_lobo_dim        per pair: in each fold the rank-d reduced-rank ridge model is fit on five blocks (training-mean
                     centred) and predicts the held-out block's target activity; R² = 1 − SSE/SST pooled over the six
                     folds and over target units, SST about the training mean.
  r2_lobo_null_mean  per pair: the same after permuting target trials relative to source trials (N_ACC_PERMUTATIONS
                     draws). The permutations are generated once per session and applied to every subsample; the
                     null is averaged over subsamples draw by draw.

    RESULTS_DIR=../results_regress_full python patch_lobo_r2.py [session ...]
"""
import os, pickle, sys, time
from pathlib import Path
import numpy as np, numpy.random as npr, pandas as pd, pynwb
from sklearn.model_selection import LeaveOneGroupOut
import task_axis_comm_subspace as m


def lobo_rank_r2(X, Y, alpha, d, folds):
    sse = sst = 0.0
    for tr, te in folds:
        xm, ym = X[tr].mean(axis=0), Y[tr].mean(axis=0)
        Xtr, Ytr = X[tr] - xm, Y[tr] - ym
        B = m.ridge_coef(Xtr, Ytr, alpha)
        _, _, Vt = np.linalg.svd(Xtr @ B, full_matrices=False)
        V = Vt[:d].T
        Yte = Y[te] - ym
        pred = (X[te] - xm) @ B @ V @ V.T
        sse += np.sum((Yte - pred) ** 2)
        sst += np.sum(Yte ** 2)
    return 1 - sse / sst


def patch(session_dir: Path, root: Path):
    t0 = time.time()
    with open(session_dir / 'alignment_results.pkl', 'rb') as f:
        res = pickle.load(f)
    sid, areas, pairs, K = res['session_id'], res['areas'], res['pairs'], res['n_subsamples']
    m.NUISANCE_REGRESSORS = ','.join(res.get('nuisance_regressors', [])) or 'none'
    session = pynwb.read_nwb(m.find_nwb(root, sid))
    trials, units = session.trials[:], session.units[:]
    reg, fr, _, _ = m.preprocess(session, trials, units, log=lambda *a, **k: None)
    assert len(reg) == res['n_trials'], (len(reg), res['n_trials'])
    context, blocks = reg['rewarded_modality'].values, reg['block_index'].values
    assert (blocks == res['blocks']).all()
    sua = units[units['is_qc_pass'] & (units['decoder_label'] == 'sua')]
    folds = list(LeaveOneGroupOut().split(np.arange(len(reg)), groups=blocks))
    P = len(pairs)
    trial_perms = m.trial_permutations(len(reg), m.N_ACC_PERMUTATIONS, seed=5000)   # shared by every subsample
    r2 = np.empty((K, P))
    null = np.empty((K, P, m.N_ACC_PERMUTATIONS))
    for k in range(K):
        tk = time.time()
        rng = npr.default_rng(k)
        sub = {a: rng.choice(sua[sua['structure'] == a].index.values, size=m.MIN_UNITS, replace=False) for a in areas}
        act = {a: m.zscore(fr[sub[a]].values) for a in areas}
        chk = m.lda_cv_accuracy(act[areas[0]], context, m.context_block_folds(context, blocks))
        assert abs(chk - res['acc_cv'][k, 0]) < 1e-9, (k, chk, res['acc_cv'][k, 0])
        for p, (s, t) in enumerate(pairs):
            X, Y = act[s], act[t]
            alpha, d = m.select_alpha(X, Y), int(res['dims'][k, p])
            r2[k, p] = lobo_rank_r2(X, Y, alpha, d, folds)
            for j in range(m.N_ACC_PERMUTATIONS):
                null[k, p, j] = lobo_rank_r2(X, Y[trial_perms[j]], alpha, d, folds)
        print(f'{sid}: subsample {k + 1}/{K} in {time.time() - tk:.0f} s', flush=True)
    null_m = null.mean(axis=0)
    res['r2_lobo_dim'], res['r2_lobo_null_mean'] = r2, null_m
    pt = res['pair_table']
    pt['r2_lobo_dim'], pt['r2_lobo_dim_sd'] = r2.mean(0), r2.std(0, ddof=1)
    pt['r2_lobo_null_mean'] = null_m.mean(axis=1)
    pt['r2_lobo_null_q975'] = np.quantile(null_m, 0.975, axis=1)
    pt['r2_lobo_p'] = [m.p_one_sided(null_m[p], r2[:, p].mean()) for p in range(P)]
    with open(session_dir / 'alignment_results.pkl', 'wb') as f:
        pickle.dump(res, f)
    pt.to_csv(session_dir / 'alignment_by_pair.csv', index=False)
    n_up = int((pt['r2_lobo_dim'] > pt['r2_lobo_null_q975']).sum())
    print(f'{session_dir.parent.name}/{sid}: LOBO rank-d R² mean {r2.mean():.3f} (range {r2.mean(0).min():.3f}–{r2.mean(0).max():.3f}), '
          f'{n_up}/{P} pairs above null; {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    code_dir = Path(__file__).resolve().parent
    root = Path(os.environ.get('DATACUBE_ROOT', code_dir.parent / 'data' / 'dynamicrouting_datacube'))
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results'))
    sessions = sys.argv[1:] or sorted(p.parent.name for p in results_dir.glob('*/alignment_results.pkl'))
    for sid in sessions:
        patch(results_dir / sid, root)
