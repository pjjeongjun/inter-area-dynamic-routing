"""Add cross-validated fit-quality fields to existing alignment_results.pkl files without rerunning the
permutation tests, rebuilding the same preprocessing and unit subsamples as the stored run:

  Cross-validation is the balanced block-wise design of the main analysis: one block of each context is held
  out (9 folds), models are fit on the remaining 2 + 2 blocks, and the LDA uses equal class priors.

  task_r2_cv   per area: cross-validated predictive R² of the context axis. In each fold the axis and the
               per-context means of the projection are fit on the training blocks; the held-out blocks' projection
               is predicted by its context mean, and R² = 1 − SSE / SST is pooled over folds.
  sub_acc_cv   per pair: cross-validated context decoding accuracy from the source activity projected onto the
               rank-d communication subspace. In each fold the subspace (ridge + SVD, rank d) and the LDA on its
               d coordinates are fit on the training blocks and tested on the held-out blocks.
  sub_acc_null per pair: the same with context labels permuted across trials (N_ACC_PERMUTATIONS draws), reusing
               the per-fold subspace projections. As in the main analysis the permutations are generated once per
               session and applied to every subsample, and the null is averaged over subsamples draw by draw.
  sub_acc_block_null per pair: the same with labels permuted across whole blocks (block-permutation null, 18 draws);
               `sub_acc_predictive` = accuracy above every block-permutation draw.

    DATACUBE_ROOT=... RESULTS_DIR=../results_regress_full python patch_cv_quality.py [session ...]
"""
import os, pickle, sys, time
from pathlib import Path
import numpy as np, numpy.random as npr, pandas as pd, pynwb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import task_axis_comm_subspace as m


def axis_r2_cv(X, labels, folds):
    sse = sst = 0.0
    for tr, te in folds:
        w, _ = m.lda_axis(X[tr], labels[tr])
        ptr, pte = X[tr] @ w, X[te] @ w
        means = {c: ptr[labels[tr] == c].mean() for c in np.unique(labels[tr])}
        pred = np.array([means[c] for c in labels[te]])
        sse += np.sum((pte - pred) ** 2)
        sst += np.sum((pte - ptr.mean()) ** 2)
    return 1 - sse / sst


def subspace_projections(X, Y, alpha, d, folds):
    """Per fold: (train idx, test idx, train coords, test coords) on the subspace fit on the training trials."""
    out = []
    for tr, te in folds:
        xm, ym = X[tr].mean(axis=0), Y[tr].mean(axis=0)
        Q_src, _ = m.comm_subspace(X[tr] - xm, Y[tr] - ym, alpha, d)
        out.append((tr, te, (X[tr] - xm) @ Q_src, (X[te] - xm) @ Q_src))
    return out


def cv_accuracy_from(proj, labels):
    """LDA (equal priors) on the subspace coordinates, pooled over held-out trials."""
    correct = total = 0
    for tr, te, Ztr, Zte in proj:
        lda = LinearDiscriminantAnalysis(solver='svd', priors=[0.5, 0.5]).fit(Ztr, labels[tr])
        correct += (lda.predict(Zte) == labels[te]).sum()
        total += len(te)
    return correct / total


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
    assert (context == res['context']).all() and (blocks == res['blocks']).all()
    sua = units[units['is_qc_pass'] & (units['decoder_label'] == 'sua')]
    P = len(pairs)
    label_perms = m.label_permutations(context, m.N_ACC_PERMUTATIONS, seed=4000)   # shared by every subsample
    block_perms = m.block_label_permutations(context, blocks)
    n_block = len(block_perms)
    folds_ctx = m.context_block_folds(context, blocks)                              # hold out one block per context
    block_folds = [m.context_block_folds(lab, blocks) for lab in block_perms]        # same design under permuted labels
    task_r2_cv = np.empty((K, len(areas)))
    sub_acc = np.empty((K, P))
    sub_acc_null = np.empty((K, P, m.N_ACC_PERMUTATIONS))
    sub_acc_block = np.empty((K, P, n_block))
    for k in range(K):
        tk = time.time()
        rng = npr.default_rng(k)
        sub = {a: rng.choice(sua[sua['structure'] == a].index.values, size=m.MIN_UNITS, replace=False) for a in areas}
        act = {a: m.zscore(fr[sub[a]].values) for a in areas}
        # the rebuilt activity must reproduce the stored cross-validated accuracy of the first area
        chk = m.lda_cv_accuracy(act[areas[0]], context, folds_ctx)
        assert abs(chk - res['acc_cv'][k, 0]) < 1e-9, (k, chk, res['acc_cv'][k, 0])
        for i, a in enumerate(areas):
            task_r2_cv[k, i] = axis_r2_cv(act[a], context, folds_ctx)
        for p, (s, t) in enumerate(pairs):
            X, Y = act[s], act[t]
            alpha, d = m.select_alpha(X, Y), int(res['dims'][k, p])
            proj = subspace_projections(X, Y, alpha, d, folds_ctx)
            sub_acc[k, p] = cv_accuracy_from(proj, context)
            for j in range(m.N_ACC_PERMUTATIONS):
                sub_acc_null[k, p, j] = cv_accuracy_from(proj, label_perms[j])
            for j in range(n_block):     # the block-permuted labels define their own balanced folds
                sub_acc_block[k, p, j] = cv_accuracy_from(subspace_projections(X, Y, alpha, d, block_folds[j]), block_perms[j])
        print(f'{sid}: subsample {k + 1}/{K} in {time.time() - tk:.0f} s', flush=True)
    null_m, block_m = sub_acc_null.mean(axis=0), sub_acc_block.mean(axis=0)      # draw-wise over subsamples
    res['task_r2_cv'], res['sub_acc_cv'] = task_r2_cv, sub_acc
    res['sub_acc_null_mean'], res['sub_acc_block_null_mean'] = null_m, block_m
    at, pt = res['area_table'], res['pair_table']
    at['task_r2_cv'], at['task_r2_cv_sd'] = task_r2_cv.mean(0), task_r2_cv.std(0, ddof=1)
    obs = sub_acc.mean(0)
    pt['sub_acc_cv'], pt['sub_acc_cv_sd'] = obs, sub_acc.std(0, ddof=1)
    pt['sub_acc_null_mean'] = null_m.mean(axis=1)                                 # trial-shuffle null (reference)
    pt['sub_acc_null_q975'] = np.quantile(null_m, 0.975, axis=1)
    pt['sub_acc_p'] = [m.p_one_sided(null_m[p], obs[p]) for p in range(P)]
    pt['sub_acc_block_null_mean'] = block_m.mean(axis=1)                          # block-permutation null (qualification)
    pt['sub_acc_block_null_lo'], pt['sub_acc_block_null_hi'] = block_m.min(axis=1), block_m.max(axis=1)
    pt['sub_acc_block_p'] = [m.p_one_sided(block_m[p], obs[p]) for p in range(P)]
    pt['sub_acc_predictive'] = obs > pt['sub_acc_block_null_hi'].values
    with open(session_dir / 'alignment_results.pkl', 'wb') as f:
        pickle.dump(res, f)
    at.to_csv(session_dir / 'task_axis_by_area.csv', index=False)
    pt.to_csv(session_dir / 'alignment_by_pair.csv', index=False)
    n_up = int(pt['sub_acc_predictive'].sum())
    print(f'{session_dir.parent.name}/{sid}: task CV R² {task_r2_cv.mean():.3f} (min {task_r2_cv.mean(0).min():.3f}), '
          f'subspace accuracy {sub_acc.mean():.3f}, {n_up}/{P} pairs above the block-permutation null; '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    code_dir = Path(__file__).resolve().parent
    root = Path(os.environ.get('DATACUBE_ROOT', code_dir.parent / 'data' / 'dynamicrouting_datacube'))
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results'))
    sessions = sys.argv[1:] or sorted(p.parent.name for p in results_dir.glob('*/alignment_results.pkl'))
    for sid in sessions:
        patch(results_dir / sid, root)
