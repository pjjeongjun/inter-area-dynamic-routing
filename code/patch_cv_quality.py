"""Add cross-validated fit-quality fields to existing alignment_results.pkl files without rerunning the
permutation tests, rebuilding the same preprocessing and unit subsamples as the stored run:

  Cross-validation follows m.CONTEXT_CV: one block of each context held out (CONTEXT_CV=block, default; 9 folds) or
  stratified 10-fold held-out trials drawn from every block (CONTEXT_CV=trial); the LDA uses equal class priors.

  acc_*        per area: the main analysis's LDA context decoding (acc_cv, its trial-shuffle and block-permutation
               nulls, acc_predictive), recomputed with these folds so that a change of CONTEXT_CV does not need the
               full analysis rerun (the axes, subspaces and alignment do not depend on the decoding folds).

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
  sub_acc_nc_* per pair: the same three quantities with a nearest-centroid reader instead of the LDA: the per-context
               means of the d subspace coordinates are taken from the training blocks and a held-out trial is assigned
               to the nearer mean (Euclidean distance; no covariance estimate, no fitted weights).
  acc_nc_*     per area: the nearest-centroid reader on all MIN_UNITS units of the area (the per-context means of
               the z-scored population vector), with the same folds, trial-shuffle and block-permutation nulls as
               the LDA's acc_cv; `acc_nc_predictive` = accuracy above every block-permutation draw.

    DATACUBE_ROOT=... RESULTS_DIR=../results_3sessions python patch_cv_quality.py [session ...]
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


def cv_accuracy_from(proj, labels, decoder='lda'):
    """Context decoding on the subspace coordinates, pooled over held-out trials. decoder='lda': LDA with equal
    priors; 'nc': nearest centroid (held-out trial -> the nearer of the two training-set context means)."""
    correct = total = 0
    for tr, te, Ztr, Zte in proj:
        if decoder == 'lda':
            lda = LinearDiscriminantAnalysis(solver='svd', priors=[0.5, 0.5]).fit(Ztr, labels[tr])
            pred = lda.predict(Zte)
        else:
            classes = np.unique(labels[tr])
            mu = np.stack([Ztr[labels[tr] == c].mean(axis=0) for c in classes])          # (2, d)
            dist = ((Zte[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)                 # (n_te, 2)
            pred = classes[dist.argmin(axis=1)]
        correct += (pred == labels[te]).sum()
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
    lda_perms = m.label_permutations(context, m.N_ACC_PERMUTATIONS, seed=1000)     # the main analysis's decoding null
    block_perms = m.block_label_permutations(context, blocks)
    n_block = len(block_perms)
    folds_ctx = m.context_folds(context, blocks)                              # block-wise by default (CONTEXT_CV)
    block_folds = [m.context_folds(lab, blocks) for lab in block_perms]        # same design under permuted labels
    task_r2_cv = np.empty((K, len(areas)))
    sub_acc = np.empty((K, P))
    sub_acc_null = np.empty((K, P, m.N_ACC_PERMUTATIONS))
    sub_acc_block = np.empty((K, P, n_block))
    nc_acc, nc_null, nc_block = np.empty((K, P)), np.empty((K, P, m.N_ACC_PERMUTATIONS)), np.empty((K, P, n_block))
    A = len(areas)
    a_nc, a_nc_null, a_nc_block = np.empty((K, A)), np.empty((K, A, m.N_ACC_PERMUTATIONS)), np.empty((K, A, n_block))
    acc_cv, acc_null, acc_block = np.empty((K, A)), np.empty((K, A, m.N_ACC_PERMUTATIONS)), np.empty((K, A, n_block))
    for k in range(K):
        tk = time.time()
        rng = npr.default_rng(k)
        sub = {a: rng.choice(sua[sua['structure'] == a].index.values, size=m.MIN_UNITS, replace=False) for a in areas}
        act = {a: m.zscore(fr[sub[a]].values) for a in areas}
        # the rebuilt activity must reproduce the stored in-sample LDA accuracy of the first area (fold-independent)
        chk = m.lda_axis(act[areas[0]], context)[1]
        assert abs(chk - res['acc_train'][k, 0]) < 1e-9, (k, chk, res['acc_train'][k, 0])
        for i, a in enumerate(areas):
            acc_cv[k, i] = m.lda_cv_accuracy(act[a], context, folds_ctx)
            for j in range(m.N_ACC_PERMUTATIONS):
                acc_null[k, i, j] = m.lda_cv_accuracy(act[a], lda_perms[j], folds_ctx)
            for j in range(n_block):
                acc_block[k, i, j] = m.lda_cv_accuracy(act[a], block_perms[j], block_folds[j])
            task_r2_cv[k, i] = axis_r2_cv(act[a], context, folds_ctx)
            X = act[a]                         # nearest centroid on the full population vector
            full = [(tr, te, X[tr], X[te]) for tr, te in folds_ctx]
            a_nc[k, i] = cv_accuracy_from(full, context, 'nc')
            for j in range(m.N_ACC_PERMUTATIONS):
                a_nc_null[k, i, j] = cv_accuracy_from(full, label_perms[j], 'nc')
            for j in range(n_block):
                a_nc_block[k, i, j] = cv_accuracy_from([(tr, te, X[tr], X[te]) for tr, te in block_folds[j]], block_perms[j], 'nc')
        for p, (s, t) in enumerate(pairs):
            X, Y = act[s], act[t]
            alpha, d = m.select_alpha(X, Y), int(res['dims'][k, p])
            proj = subspace_projections(X, Y, alpha, d, folds_ctx)
            sub_acc[k, p] = cv_accuracy_from(proj, context)
            nc_acc[k, p] = cv_accuracy_from(proj, context, 'nc')
            for j in range(m.N_ACC_PERMUTATIONS):
                sub_acc_null[k, p, j] = cv_accuracy_from(proj, label_perms[j])
                nc_null[k, p, j] = cv_accuracy_from(proj, label_perms[j], 'nc')
            for j in range(n_block):     # the block-permuted labels define their own balanced folds
                proj_b = subspace_projections(X, Y, alpha, d, block_folds[j])
                sub_acc_block[k, p, j] = cv_accuracy_from(proj_b, block_perms[j])
                nc_block[k, p, j] = cv_accuracy_from(proj_b, block_perms[j], 'nc')
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
    # nearest-centroid reader
    nc_null_m, nc_block_m = nc_null.mean(axis=0), nc_block.mean(axis=0)
    res['sub_acc_nc_cv'], res['sub_acc_nc_null_mean'], res['sub_acc_nc_block_null_mean'] = nc_acc, nc_null_m, nc_block_m
    obs_nc = nc_acc.mean(0)
    pt['sub_acc_nc_cv'], pt['sub_acc_nc_cv_sd'] = obs_nc, nc_acc.std(0, ddof=1)
    pt['sub_acc_nc_null_mean'] = nc_null_m.mean(axis=1)
    pt['sub_acc_nc_p'] = [m.p_one_sided(nc_null_m[p], obs_nc[p]) for p in range(P)]
    pt['sub_acc_nc_block_null_mean'] = nc_block_m.mean(axis=1)
    pt['sub_acc_nc_block_null_lo'], pt['sub_acc_nc_block_null_hi'] = nc_block_m.min(axis=1), nc_block_m.max(axis=1)
    pt['sub_acc_nc_block_p'] = [m.p_one_sided(nc_block_m[p], obs_nc[p]) for p in range(P)]
    pt['sub_acc_nc_predictive'] = obs_nc > pt['sub_acc_nc_block_null_hi'].values
    # LDA context decoding per area (all units), as in the main analysis
    l_null_m, l_block_m = acc_null.mean(axis=0), acc_block.mean(axis=0)
    res['acc_cv'], res['acc_null_mean'], res['acc_block_null_mean'] = acc_cv, l_null_m, l_block_m
    res['context_cv'] = m.CONTEXT_CV
    obs_l = acc_cv.mean(0)
    at['acc_cv'], at['acc_cv_sd'] = obs_l, acc_cv.std(0, ddof=1)
    at['acc_null_mean'] = l_null_m.mean(axis=1)
    at['acc_null_p'] = [m.p_one_sided(l_null_m[i], obs_l[i]) for i in range(A)]
    at['acc_null_q975'] = np.quantile(l_null_m, 0.975, axis=1)
    for i in range(A):
        for key, val in m.null_summary(l_block_m[i], obs_l[i], 'acc_block_null', one_sided=True, extreme=True).items():
            at.loc[i, key] = val
    at['acc_block_p'] = at.pop('acc_block_null_p')
    at['acc_predictive'] = obs_l > at['acc_block_null_hi'].values
    # nearest-centroid reader per area (all units)
    a_null_m, a_block_m = a_nc_null.mean(axis=0), a_nc_block.mean(axis=0)
    res['acc_nc_cv'], res['acc_nc_null_mean'], res['acc_nc_block_null_mean'] = a_nc, a_null_m, a_block_m
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
    pt.to_csv(session_dir / 'alignment_by_pair.csv', index=False)
    n_up = int(pt['sub_acc_predictive'].sum())
    print(f'{session_dir.parent.name}/{sid}: task CV R² {task_r2_cv.mean():.3f} (min {task_r2_cv.mean(0).min():.3f}), '
          f'subspace accuracy LDA {sub_acc.mean():.3f} ({n_up}/{P} above the block null), '
          f'nearest-centroid {nc_acc.mean():.3f} ({int(pt["sub_acc_nc_predictive"].sum())}/{P}); '
          f'area nearest-centroid {a_nc.mean():.3f} ({int(at["acc_nc_predictive"].sum())}/{A} above the block null); '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    code_dir = Path(__file__).resolve().parent
    root = Path(os.environ.get('DATACUBE_ROOT', code_dir.parent / 'data' / 'dynamicrouting_datacube'))
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results_3sessions'))
    sessions = sys.argv[1:] or sorted(p.parent.name for p in results_dir.glob('*/alignment_results.pkl'))
    for sid in sessions:
        patch(results_dir / sid, root)
