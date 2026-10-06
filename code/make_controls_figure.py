"""Pooled figure for the two alignment controls (cross-block axis, nuisance axes).

    RESULTS_DIR=../results_regress_prev python make_controls_figure.py

Reads every ``<session>/controls_results.pkl`` under RESULTS_DIR and writes
``pooled/figure_controls_pooled.svg`` plus ``pooled/controls_by_pair_pooled.csv``.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from make_alignment_figures import (FDR_ALPHA, GROUP_COLOR, MARKERS, PAIR_COLOR, PAIR_LABEL, PAIR_TYPES, area_group,
                                    area_order, group_legend, place_letters, save, session_legend,
                                    strip_by_type_sessions)



def figure_controls(results, out_dir: Path):
    sessions = [r['session_id'] for r in results]
    tables_all = [r['pair_table'] for r in results]
    pooled_all = pd.concat(tables_all, ignore_index=True)
    pooled_all.to_csv(out_dir / 'controls_by_pair_pooled.csv', index=False)
    tables = [t[t['qualified']] for t in tables_all]
    pooled = pooled_all[pooled_all['qualified']]
    areas_tab = pd.concat([r['area_table'] for r in results], ignore_index=True)
    areas_q = areas_tab[areas_tab['axis_predictive']]
    n_splits = results[0]['n_splits']

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.45, left=0.07, right=0.98, top=0.8, bottom=0.2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # A: same-blocks vs cross-blocks vs chance, one line per qualified pair
    cols = ['cos_task', 'cos_same', 'cos_cross', 'cos_cross_chance']
    xlab = ['all\ntrials', 'same\n3 blocks', 'other\n3 blocks', 'chance']
    rng = np.random.default_rng(0)
    for k, t in enumerate(tables):
        for _, r in t.iterrows():
            jitter = rng.uniform(-0.08, 0.08)
            y = [r[c] for c in cols]
            axA.plot(np.arange(4) + jitter, y, color=PAIR_COLOR[r['pair_type']], lw=0.5, alpha=0.5)
            axA.scatter(np.arange(4) + jitter, y, s=10, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[r['pair_type']],
                        edgecolor='k', linewidth=0.2, zorder=3)
    means = [pooled[c].mean() for c in cols]
    axA.plot(np.arange(4) + 0.25, means, 'k_', ms=12, mew=1.6, zorder=4)
    axA.set_xticks(range(4))
    axA.set_xticklabels(xlab, fontsize=6.8)
    axA.set_ylim(0, 1)
    axA.set_ylabel('cosine (task axis, comm. subspace)')
    n_up = int(((pooled['cos_cross_q'] <= FDR_ALPHA) & (pooled['cos_cross_z'] > 0)).sum())
    axA.set_title(f'Cross-block control, {len(pooled)} qualified pairs\n'
                  f'axis fit on other blocks: {n_up}/{len(pooled)} above chance (q < {FDR_ALPHA})', loc='left')
    axA.set_xlabel('blocks used for the task axis relative to the subspace')
    session_legend(axA, sessions, loc='lower left')

    # B: cross-block excess vs same-block excess
    for k, t in enumerate(tables):
        for typ in PAIR_TYPES:
            sub = t[t['pair_type'] == typ]
            axB.scatter(sub['cos_same'] - sub['cos_cross_chance'], sub['cos_cross'] - sub['cos_cross_chance'], s=18,
                        marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[typ], edgecolor='k', linewidth=0.3)
    lim = max(0.7, (pooled['cos_same'] - pooled['cos_cross_chance']).max() + 0.05)
    axB.plot([0, lim], [0, lim], 'k--', lw=0.7)
    axB.axhline(0, color='k', ls=':', lw=0.7)
    axB.set_xlim(-0.05, lim)
    axB.set_ylim(min(-0.1, (pooled['cos_cross'] - pooled['cos_cross_chance']).min() - 0.03), lim)
    axB.set_aspect('equal')
    axB.set_xlabel('same-blocks cosine − chance')
    axB.set_ylabel('cross-blocks cosine − chance')
    frac = ((pooled['cos_cross'] - pooled['cos_cross_chance']) / (pooled['cos_same'] - pooled['cos_cross_chance'])).median()
    axB.set_title(f'Fraction that generalises across blocks\nmedian (cross \u2212 chance) / (same \u2212 chance) = {frac:.2f}', loc='left')
    h = [Line2D([], [], marker='o', color=PAIR_COLOR[t], ls='', ms=5, label=PAIR_LABEL[t]) for t in PAIR_TYPES]
    axB.legend(handles=h, loc='upper left', frameon=False, title='source → target')

    # C: cross-block z by pair type
    strip_by_type_sessions(axC, tables, 'cos_cross_z', 'cross-block cosine: z vs. random-axis null', zero_line=0,
                           show_n=False)
    axC.set_title('Cross-block alignment by pair type\nz vs. random-axis null', loc='left')

    fig.suptitle(f'Cross-block control (pooled over {len(results)} sessions, {n_splits} splits; regressed out: '
                 f'{", ".join(results[0]["regressors"])})', fontsize=9, x=0.07, ha='left', y=0.97)
    place_letters(fig, [axA, axB, axC], 'ABC', dx=-0.05)
    save(fig, out_dir / 'figure_controls_pooled.svg')
    return pooled_all


def main(argv):
    code_dir = Path(__file__).resolve().parent
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results'))
    sessions = argv or sorted(p.parent.name for p in results_dir.glob('*/controls_results.pkl'))
    results = []
    for sid in sessions:
        with open(results_dir / sid / 'controls_results.pkl', 'rb') as f:
            results.append(pickle.load(f))
    out = results_dir / 'pooled'
    out.mkdir(parents=True, exist_ok=True)
    figure_controls(results, out)
    print(f'controls figure written to {out}')


if __name__ == '__main__':
    main(sys.argv[1:])
