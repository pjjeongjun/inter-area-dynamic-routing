"""Trial-shuffle significance of the communication subspace's R² at its own dimensionality.

    DATACUBE_ROOT=... RESULTS_DIR=../results python r2_dim_null.py [session_id ...]

``task_axis_comm_subspace.py`` tests the full ridge model's R² against a trial-shuffle
null. The figures show the R² of the reduced-rank model at the pair's 1-SEM
dimensionality d instead, so this script tests that quantity directly: for every
subsample and ordered pair it rebuilds the same activity (same preprocessing, same unit
subsample, same folds), keeps that pair's ridge penalty and d fixed, permutes the
target trials relative to the source trials ``N_R2_PERMUTATIONS`` times and recomputes
the 10-fold CV R² at rank d (mean over folds, as in ``r2_curve``). As in the main
pipeline, the null is averaged over subsamples draw-wise, the p-value is one-sided and
FDR (Benjamini-Hochberg) runs over the ordered pairs of the session.

The preprocessing must match the run that produced ``alignment_results.pkl``; it is
read from the pickle (``nuisance_regressors``; absent = none). The script checks that
it reproduces the stored rank-d R² before computing the null.

Writes ``RESULTS_DIR/<session>/r2_dim_significance.csv`` (one row per ordered pair).
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


def rank_d_cv_r2_batch(X, Y_batch, alpha, folds, d):
    """Mean-over-folds CV R² of rank-d reduced-rank ridge regression, for a batch of
    target matrices Y_batch (P x trials x targets) sharing the source X. Same model as
    ``cv_rrr_r2``: ridge on the centred training fold, rank-d projection onto the top d
    right singular vectors of the training-set prediction."""
    P = Y_batch.shape[0]
    r2 = np.zeros(P)
    n_src = X.shape[1]
    for tr, te in folds:
        xm = X[tr].mean(axis=0)
        Xtr, Xte = X[tr] - xm, X[te] - xm
        M = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(n_src), Xtr.T)         # B = M @ Ytr
        ym = Y_batch[:, tr].mean(axis=1, keepdims=True)
        Ytr, Yte = Y_batch[:, tr] - ym, Y_batch[:, te] - ym
        B = M @ Ytr                                                              # P x src x tgt
        F = Xtr @ B                                                              # training prediction
        _, V = np.linalg.eigh(np.swapaxes(F, 1, 2) @ F)                          # right singular vectors (ascending)
        Vd = V[:, :, -d:]
        pred = (Xte @ B) @ Vd @ np.swapaxes(Vd, 1, 2)
        r2 += 1 - np.sum((Yte - pred) ** 2, axis=(1, 2)) / np.sum(Yte ** 2, axis=(1, 2))
    return r2 / len(folds)


def main(argv):
    code_dir = Path(__file__).resolve().parent
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results'))
    root = Path(os.environ.get('DATACUBE_ROOT', code_dir.parent / 'data' / 'dynamicrouting_datacube'))
    sessions = argv or sorted(p.parent.name for p in results_dir.glob('*/alignment_results.pkl'))
    for sid in sessions:
        with open(results_dir / sid / 'alignment_results.pkl', 'rb') as f:
            res = pickle.load(f)
        # the pipeline reads NUISANCE_REGRESSORS at import: match the stored run before importing it
        os.environ['NUISANCE_REGRESSORS'] = ','.join(res.get('nuisance_regressors') or []) or 'none'
        sys.modules.pop('task_axis_comm_subspace', None)
        import task_axis_comm_subspace as T
        from sklearn.model_selection import KFold

        t0 = time.time()
        session = None
        if os.environ['NUISANCE_REGRESSORS'] != 'none':
            import pynwb
            session = pynwb.read_nwb(T.find_nwb(root, sid))
            trials, units = session.trials[:], session.units[:]
        else:
            trials, units = T.load_session(root, sid)
        reg, fr, _, _ = T.preprocess(session, trials, units)

        # same unit selection and subsamples as analyse_session
        n, K = res['min_units'], res['n_subsamples']
        sua = units[units['is_qc_pass'] & (units['decoder_label'] == 'sua')]
        counts = sua['structure'].value_counts()
        areas = counts[counts >= n].index.tolist()
        assert areas == res['areas'], f'{sid}: areas differ from the stored run ({areas} vs {res["areas"]})'
        pairs = res['pairs']
        subsamples = []
        for k in range(K):
            rng = npr.default_rng(k)
            subsamples.append({a: rng.choice(sua[sua['structure'] == a].index.values, size=n, replace=False)
                               for a in areas})
        folds = list(KFold(T.N_FOLDS, shuffle=True, random_state=0).split(np.arange(len(reg))))

        dims = res["dims"]
        n_perm = T.N_R2_PERMUTATIONS
        obs = np.empty((K, len(pairs)))
        null = np.empty((K, len(pairs), n_perm))
        max_err = 0.0
        for k in range(K):
            act = {a: T.zscore(fr[subsamples[k][a]].values) for a in areas}
            perm_rng = npr.default_rng(5000 + k)
            for p, (s, t) in enumerate(pairs):
                X, Y = act[s], act[t]
                a_kp = T.select_alpha(X, Y)
                d = int(dims[k, p])
                obs[k, p] = rank_d_cv_r2_batch(X, Y[None], a_kp, folds, d)[0]
                max_err = max(max_err, abs(obs[k, p] - res['r2_curve'][k, p, d - 1]))
                perms = np.stack([perm_rng.permutation(len(Y)) for _ in range(n_perm)])
                for c in range(0, n_perm, 250):
                    null[k, p, c:c + 250] = rank_d_cv_r2_batch(X, Y[perms[c:c + 250]], a_kp, folds, d)
            print(f'  {sid}: subsample {k + 1}/{K} done ({time.time() - t0:.0f} s)', flush=True)
        if max_err > 1e-6:
            raise RuntimeError(f'{sid}: rank-d R² does not reproduce the stored run (max |diff| = {max_err:.2e})')

        obs_m, null_m = obs.mean(axis=0), null.mean(axis=0)                    # null averaged draw-wise
        out = pd.DataFrame({
            'session': sid, 'source': [s for s, _ in pairs], 'target': [t for _, t in pairs],
            'r2_dim': obs_m, 'r2_dim_null_mean': null_m.mean(axis=1),
            'r2_dim_null_q95': np.quantile(null_m, 0.95, axis=1),
            'r2_dim_p': [T.p_one_sided(null_m[p], obs_m[p]) for p in range(len(pairs))],
            'n_permutations': n_perm,
        })
        out['r2_dim_q'] = T.fdr_bh(out['r2_dim_p'].values)
        out.to_csv(results_dir / sid / 'r2_dim_significance.csv', index=False)
        print(f'{sid}: reproduced stored rank-d R² (max |diff| {max_err:.1e}); '
              f'{int((out["r2_dim_q"] <= T.FDR_ALPHA).sum())}/{len(out)} pairs significant at q < {T.FDR_ALPHA}; '
              f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
