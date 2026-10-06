"""Portrait-format (6.5 in wide) figures for the short report, drawn from the pooled outputs of one
regression setting (by default the full nuisance set in results_regress_full/).

    python make_report_figures.py        # local: ../results_regress_full -> ../results/report/figures/*.svg
    RESULTS_DIR=/root/capsule/results python make_report_figures.py   # capsule: -> $RESULTS_DIR/report/figures

Sessions are every ``<session>/alignment_results.pkl`` found under RESULTS_DIR (the three
local sessions, or all sessions of a capsule run). ``REPORT_DIR`` overrides the output folder.
Panel R2-D needs ``RESULTS_DIR/pooled/controls_by_pair_pooled.csv`` (make_controls_figure.py).

Figure R1  communication subspace (cross-validated R², context decoding) and per-area context decoding (default run)
Figure S1  communication-subspace dimensionality (supplementary)
Figure R2  alignment by area (source, target), vs. context decoding, and the cross-block control (default run)
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
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy import stats

import make_alignment_figures as m
from make_alignment_figures import (FDR_ALPHA, GROUP_COLOR, MARKERS, PAIR_COLOR, PAIR_LABEL, PAIR_TYPES, area_group,
                                    area_order, place_letters, qualify, r2_at_dimensionality, save)

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'legend.fontsize': 6.2, 'legend.title_fontsize': 6.5})

BUNDLE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', BUNDLE / 'results_regress_full'))
SESSIONS = sorted(p.parent.name for p in RESULTS_DIR.glob('*/alignment_results.pkl'))
SHORT = {'frontal->frontal': 'Frontal\n→ Frontal', 'frontal->other': 'Frontal\n→ Others', 'other->frontal': 'Others\n→ Frontal',
         'other->other': 'Others\n→ Others'}
W = 6.5


def load_results(d=None):
    base = RESULTS_DIR if d is None else BUNDLE / d
    return [pickle.load(open(base / s / 'alignment_results.pkl', 'rb')) for s in SESSIONS]


def short_type_ticks(ax):
    ax.set_xticklabels([SHORT[t] for t in PAIR_TYPES])


def strip(ax, tables, ycol, ylabel, zero_line=None, log=False, show_n=False, text_loc='top'):
    m.strip_by_type_sessions(ax, tables, ycol, ylabel, zero_line=zero_line, log=log, show_n=show_n, text_loc=text_loc)
    short_type_ticks(ax)


def session_legend(ax, loc, mouse_only=False, **kw):
    h = [Line2D([], [], marker=MARKERS[k % len(MARKERS)], color='0.5', ls='', ms=4, label=s.split('_')[0] if mouse_only else s)
         for k, s in enumerate(SESSIONS)]
    ax.legend(handles=h, loc=loc, frameon=False, title='mouse' if mouse_only else 'session', **kw)


def pair_legend(ax, loc, **kw):
    h = [Line2D([], [], color=PAIR_COLOR[t], marker='o', ls='', ms=4, label=PAIR_LABEL[t]) for t in PAIR_TYPES]
    ax.legend(handles=h, loc=loc, frameon=False, title='source → target', **kw)


def scatter_types(ax, tables, xcol, ycol, s=14):
    for k, t in enumerate(tables):
        for typ in PAIR_TYPES:
            sub = t[t['pair_type'] == typ]
            ax.scatter(sub[xcol], sub[ycol], s=s, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[typ], edgecolor='k', linewidth=0.3)


# ----------------------------------------------------------------------------- R1
ACC_YLIM = (0.1, 0.9)   # shared by every decoding-accuracy panel


def acc_area_panel(ax, areas_tab, order):
    """Context-axis (LDA) leave-one-block-out decoding accuracy per area instance; filled = above the label-shuffle null."""
    for i, a in enumerate(order):
        for _, r in areas_tab[areas_tab['area'] == a].iterrows():
            k = SESSIONS.index(r['session'])
            pred = r['acc_cv'] > r['acc_null_q975']
            c = GROUP_COLOR[area_group(a)]
            ax.errorbar(i, r['acc_cv'], yerr=r['acc_cv_sd'], fmt=MARKERS[k % len(MARKERS)], ms=4.2, color=c, mfc=c if pred else 'white',
                        mec=c, mew=0.8, elinewidth=0.6, capsize=0, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    for lab, a in zip(ax.get_xticklabels(), order):
        lab.set_color(GROUP_COLOR[area_group(a)])
    ax.set_xlim(-0.7, len(order) - 0.3)
    null_hi = areas_tab['acc_null_q975'].mean()
    ax.fill_between([-0.7, len(order) - 0.3], 1 - null_hi, null_hi, color='0.88', lw=0, zorder=0)
    ax.axhline(0.5, color='k', ls='--', lw=0.7)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('Area (blue: frontal cortex, orange: others)')
    ax.set_ylabel('context decoding accuracy\n(held-out block)')


def figure_r1(results, out_dir):
    """Communication subspace: cross-validated R² per pair (A) and context decoded from it per pair (B); context axis
    (LDA): context decoding accuracy per area instance (C)."""
    tables = [qualify(r) for r in results]
    pooled = pd.concat(tables, ignore_index=True)
    areas_tab = pd.concat([r['area_table'] for r in results], ignore_index=True)

    fig = plt.figure(figsize=(W, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.62, wspace=0.42, left=0.12, right=0.985, top=0.93, bottom=0.1)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    strip(axA, tables, 'r2_cv_dim', 'cross-validated R²\n(rank-d fit, held-out trials)')
    axA.set_ylim(0, pooled['r2_cv_dim'].max() * 1.15)
    axA.axhline(0, color='k', ls='--', lw=0.7)
    axA.set_xlabel('pair type (source → target)')
    counts = pooled['pair_type'].value_counts()
    axA.text(0.03, 0.98, '\n'.join(f'{SHORT[t].replace(chr(10), " ")}: n = {counts.get(t, 0)}' for t in PAIR_TYPES), transform=axA.transAxes,
             va='top', fontsize=6)
    n_sig = int(pooled['r2_significant'].sum())
    axA.set_title(f'Comm. subspace: cross-validated R²\n{n_sig}/{len(pooled)} pairs > trial-shuffle null', loc='left')

    subspace_decoding_panel(axB, tables, pooled)

    acc_area_panel(axC, areas_tab, area_order(areas_tab['area'].unique()))
    axC.set_title('Context axis (LDA): context decoding', loc='left')
    h = [Line2D([], [], marker='o', color='0.4', ls='', ms=4, label='Above chance'),
         Line2D([], [], marker='o', mfc='white', mec='0.4', ls='', ms=4, label='Not above chance')]
    axC.legend(handles=h, loc='lower left', frameon=False, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)
    axC.add_artist(axC.get_legend())
    session_legend(axC, 'upper right', handletextpad=0.3, labelspacing=0.25, borderaxespad=0.2)

    place_letters(fig, [axA, axB, axC], 'ABC', dx=-0.075)
    save(fig, out_dir / 'figure_R1_subspace_and_decoding.svg')


def figure_rs1(results, out_dir):
    """Supplementary Figure 1: how the communication-subspace dimensionality is chosen."""
    tables = [qualify(r) for r in results]
    pooled = pd.concat(tables, ignore_index=True)
    n = results[0]['min_units']
    fig = plt.figure(figsize=(W, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.45, left=0.11, right=0.985, top=0.84, bottom=0.2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    ranks = np.arange(1, n + 1)
    for res, tab in zip(results, tables):
        curves = res['r2_curve'].mean(axis=0)
        for p in range(len(res['pairs'])):
            typ = tab.iloc[p]['pair_type']
            axA.plot(ranks, curves[p], color=PAIR_COLOR[typ], lw=0.45, alpha=0.8)
            d = tab.iloc[p]['dim']
            axA.plot(d, np.interp(d, ranks, curves[p]), marker='o', ms=2.2, color=PAIR_COLOR[typ], mec='k', mew=0.3, ls='')
    axA.axhline(0, color='k', lw=0.6, ls=':')
    axA.set_xlim(0.5, n + 0.5)
    axA.set_xlabel('Rank')
    axA.set_ylabel('R² (cross-validated)')
    axA.set_title('R² vs. rank', loc='left')
    strip(axB, tables, 'dim', 'Dimensionality')
    axB.set_xlabel('Pair type')
    axB.set_ylim(0, 12)
    axB.set_title('Dimensionality', loc='left')
    place_letters(fig, [axA, axB], 'AB', dx=-0.075)
    save(fig, out_dir / 'figure_S1_dimensionality.svg')


# ----------------------------------------------------------------------------- R2
def excess_strip(ax, tables, col, chance_col, ylabel):
    """Per-pair excess (observed − chance cosine) by pair type, one marker shape per session, black = mean ± SEM."""
    tabs = [t.assign(_ex=t[col] - t[chance_col]) for t in tables]
    strip(ax, tabs, '_ex', ylabel, zero_line=0)
    return pd.concat(tabs, ignore_index=True)


def alignment_by_area_panel(ax, pooled, area_col, excess, order, frontal_mask):
    """Alignment excess (observed − chance cosine) per area, one point per qualified pair; coloured bars = mean over all
    pairs whose plotted area is frontal / not frontal, compared by the bracket (Mann–Whitney)."""
    rng = np.random.default_rng(0)
    for i, a in enumerate(order):
        vals = excess[pooled[area_col] == a].values
        if len(vals):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=11, color=GROUP_COLOR[area_group(a)], edgecolor='k',
                       linewidth=0.3, zorder=3)
    ax.axhline(0, color='k', ls='--', lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    for lab, a in zip(ax.get_xticklabels(), order):
        lab.set_color(GROUP_COLOR[area_group(a)])
    ax.set_xlim(-0.6, len(order) - 0.4)
    p_fr = stats.mannwhitneyu(excess[frontal_mask], excess[~frontal_mask]).pvalue
    i_fr = [i for i, a in enumerate(order) if area_group(a) == 'frontal']
    i_ot = [i for i, a in enumerate(order) if area_group(a) != 'frontal']
    for g, mask, grp in ((i_fr, frontal_mask, 'frontal'), (i_ot, ~frontal_mask, 'other')):
        ax.plot([min(g) - 0.35, max(g) + 0.35], [excess[mask].mean()] * 2, color=GROUP_COLOR[grp], lw=2.0, zorder=4)
    yb = 0.64
    for g in (i_fr, i_ot):
        ax.plot([min(g), min(g), max(g), max(g)], [yb - 0.025, yb, yb, yb - 0.025], color='k', lw=0.8)
    c_fr, c_ot = np.mean([min(i_fr), max(i_fr)]), np.mean([min(i_ot), max(i_ot)])
    ax.plot([c_fr, c_fr, c_ot, c_ot], [yb + 0.01, yb + 0.05, yb + 0.05, yb + 0.01], color='k', lw=0.8)
    ax.text((c_fr + c_ot) / 2, yb + 0.05, '*' if p_fr < 0.05 else 'n.s.', ha='center', va='bottom',
            fontsize=8 if p_fr < 0.05 else 6.5)
    return p_fr


def figure_r2(results, out_dir):
    tables_all = [qualify(r) for r in results]
    pooled_all = pd.concat(tables_all, ignore_index=True)
    tables = [t[t['qualified']] for t in tables_all]
    pooled = pooled_all[pooled_all['qualified']]
    pooled_t = pooled_all[pooled_all['qualified_tgt']]
    ylab = 'Alignment\n(cosine − chance)'

    fig = plt.figure(figsize=(W, 5.9))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.32, left=0.1, right=0.985, top=0.93, bottom=0.11)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axC = fig.add_subplot(gs[1, 0], sharey=axA)

    # A: source side, by source area; bracket = frontal vs. non-frontal sources
    order = area_order(pooled['source'].unique())
    alignment_by_area_panel(axA, pooled, 'source', pooled['excess'], order, pooled['pair_type'].str.startswith('frontal'))
    axA.set_ylim(-0.25, 0.8)
    axA.set_ylabel(ylab)
    axA.set_xlabel('Source area')
    n_up = int(((pooled['cos_q_random'] <= FDR_ALPHA) & (pooled['cos_z_random'] > 0)).sum())
    axA.set_title(f'Source side: alignment by area\n{n_up}/{len(pooled)} pairs above chance', loc='left')

    # B: target side, by target area; bracket = frontal vs. non-frontal targets
    ex_t = pooled_t['cos_tgt'] - pooled_t['cos_tgt_chance']
    order_t = area_order(pooled_t['target'].unique())
    alignment_by_area_panel(axB, pooled_t, 'target', ex_t, order_t, pooled_t['pair_type'].str.endswith('frontal'))
    axB.set_ylabel(ylab)
    axB.set_xlabel('Target area')
    n_up_t = int(((pooled_t['cos_tgt_q_random'] <= FDR_ALPHA) & (pooled_t['cos_tgt_z_random'] > 0)).sum())
    axB.set_title(f'Target side: alignment by area\n{n_up_t}/{len(pooled_t)} pairs above chance', loc='left')

    # C: source-side excess against the source axis's decoding accuracy, qualified pairs only
    for k, t in enumerate(tables):
        for g in GROUP_COLOR:
            sub = t[[area_group(a) == g for a in t['source']]]
            axC.scatter(sub['source_acc_cv'], sub['excess'], s=14, marker=MARKERS[k % len(MARKERS)], color=GROUP_COLOR[g], edgecolor='k',
                        linewidth=0.3)
    axC.axhline(0, color='k', ls='--', lw=0.7)
    rho, pval = stats.spearmanr(pooled['source_acc_cv'], pooled['excess'])
    fit = stats.linregress(pooled['source_acc_cv'], pooled['excess'])
    xfit = np.array([pooled['source_acc_cv'].min(), pooled['source_acc_cv'].max()])
    axC.plot(xfit, fit.intercept + fit.slope * xfit, color='k', lw=1.1, zorder=4)
    axC.set_xlabel('Decoding accuracy (source axis)')
    axC.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axC.xaxis.set_major_locator(MultipleLocator(0.05))
    axC.set_ylabel(ylab)
    axC.set_title('Alignment vs. context decoding', loc='left')
    axC.text(0.03, 0.97, f'ρ = {rho:.2f}, p = {pval:.2f}', transform=axC.transAxes, va='top', fontsize=6.5)
    m.group_legend(axC, loc='lower right', handletextpad=0.3)

    axD = fig.add_subplot(gs[1, 1])
    cross_block_panel(axD)

    place_letters(fig, [axA, axB, axC, axD], 'ABCD', dx=-0.075)
    save(fig, out_dir / 'figure_R2_alignment.svg')


# ----------------------------------------------------------------------------- R1, panel B
def subspace_decoding_panel(ax, tables, pooled):
    """Context decoded from the communication subspace (held-out block), by pair type; filled = the pair's decoding accuracy
    is above its label-shuffle null (97.5th percentile), open = not. Asterisks: each pair type vs. its label-shuffle null; bracket: frontal → vs. other → sources."""
    rng = np.random.default_rng(0)
    for i, typ in enumerate(PAIR_TYPES):
        for k, t in enumerate(tables):
            sub = t[t['pair_type'] == typ]
            x = i + rng.uniform(-0.2, 0.2, len(sub))
            q = (sub['sub_acc_cv'] > sub['sub_acc_null_q975']).values
            ax.scatter(x[q], sub['sub_acc_cv'].values[q], s=16, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[typ], edgecolor='k',
                        linewidth=0.3, zorder=3)
            ax.scatter(x[~q], sub['sub_acc_cv'].values[~q], s=16, marker=MARKERS[k % len(MARKERS)], color='white', edgecolor=PAIR_COLOR[typ],
                        linewidth=0.7, zorder=3)
        sub = pooled[pooled['pair_type'] == typ]
        ax.errorbar(i + 0.34, sub['sub_acc_cv'].mean(), yerr=sub['sub_acc_cv'].std(ddof=1) / np.sqrt(len(sub)), fmt='_',
                     color='k', ms=9, mew=1.4, elinewidth=1.0, capsize=0, zorder=4)
    ax.set_xticks(range(len(PAIR_TYPES)))
    short_type_ticks(ax)
    ax.set_xlim(-0.6, len(PAIR_TYPES) - 0.3)
    ax.set_ylabel('context decoding accuracy\n(held-out block)')
    null_hi = pooled['sub_acc_null_q975'].mean()
    ax.fill_between([-0.6, len(PAIR_TYPES) - 0.3], 1 - null_hi, null_hi, color='0.88', lw=0, zorder=0)
    ax.axhline(0.5, color='k', ls='--', lw=0.7)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('pair type (source → target)')
    # asterisks: each pair type vs. its label-shuffle null (two-sided Wilcoxon signed-rank on accuracy − null mean);
    # bracket: frontal → vs. other → sources (Mann–Whitney), all pairs
    def stars(p):
        return '*' if p < 0.05 else 'n.s.'
    for i, typ in enumerate(PAIR_TYPES):
        sub = pooled[pooled['pair_type'] == typ]
        p_typ = stats.wilcoxon(sub['sub_acc_cv'] - sub['sub_acc_null_mean']).pvalue
        ax.text(i, sub['sub_acc_cv'].max() + 0.015, stars(p_typ), ha='center', va='bottom',
                 fontsize=8 if p_typ < 0.05 else 6.5)
    fr = pooled['pair_type'].str.startswith('frontal')
    p_fr = stats.mannwhitneyu(pooled['sub_acc_cv'][fr], pooled['sub_acc_cv'][~fr]).pvalue
    yb = 0.87
    ax.plot([0, 0, 1, 1], [yb - 0.02, yb, yb, yb - 0.02], color='k', lw=0.8)
    ax.plot([2, 2, 3, 3], [yb - 0.02, yb, yb, yb - 0.02], color='k', lw=0.8)
    ax.plot([0.5, 0.5, 2.5, 2.5], [yb + 0.01, yb + 0.04, yb + 0.04, yb + 0.01], color='k', lw=0.8)
    ax.text(1.5, yb + 0.04, stars(p_fr), ha='center', va='bottom', fontsize=8)
    ax.set_title('Comm. subspace: context decoding', loc='left')
    h = [Line2D([], [], marker='o', color='0.4', ls='', ms=4, label='Above chance'),
         Line2D([], [], marker='o', mfc='white', mec='0.4', ls='', ms=4, label='Not above chance')]
    ax.legend(handles=h, loc='lower left', frameon=False, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)



# ----------------------------------------------------------------------------- R2, panel D
def cross_block_panel(ax, d=None):
    """Cross-block control, qualified pairs: cosine between the task axis fit on the other three blocks and the
    communication subspace fit on the remaining three, vs. the random-axis chance of that half-data subspace; lines join
    the same pair, black bars = means; bracket = paired two-sided Wilcoxon signed-rank test across pairs."""
    base = RESULTS_DIR if d is None else BUNDLE / d
    c = pd.read_csv(base / 'pooled' / 'controls_by_pair_pooled.csv')
    tables = [c[(c['session'] == s) & c['qualified']] for s in SESSIONS]
    pooled = pd.concat(tables, ignore_index=True)
    cols = ['cos_cross', 'cos_cross_chance']
    rng = np.random.default_rng(0)
    for k, t in enumerate(tables):
        for _, r in t.iterrows():
            j = rng.uniform(-0.08, 0.08)
            y = [r[cc] for cc in cols]
            ax.plot(np.arange(2) + j, y, color=PAIR_COLOR[r['pair_type']], lw=0.45, alpha=0.5)
            ax.scatter(np.arange(2) + j, y, s=7, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[r['pair_type']], edgecolor='k', linewidth=0.2, zorder=3)
    ax.plot(np.arange(2) + 0.22, [pooled[cc].mean() for cc in cols], 'k_', ms=10, mew=1.5, zorder=4)
    p = stats.wilcoxon(pooled['cos_cross'], pooled['cos_cross_chance']).pvalue
    yb = pooled[cols].values.max() + 0.06
    ax.plot([0, 0, 1, 1], [yb - 0.03, yb, yb, yb - 0.03], color='k', lw=0.8)
    ax.text(0.5, yb, '*' if p < 0.05 else 'n.s.', ha='center', va='bottom', fontsize=8 if p < 0.05 else 6.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(['axis from\nother blocks', 'chance'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Cosine (axis, subspace)')
    n_up = int(((pooled['cos_cross_q'] <= FDR_ALPHA) & (pooled['cos_cross_z'] > 0)).sum())
    ax.set_title(f'Cross-block control\n{n_up}/{len(pooled)} pairs above chance', loc='left')
    return p


def main(argv):
    default_out = RESULTS_DIR / 'report' / 'figures' if 'RESULTS_DIR' in os.environ else BUNDLE / 'results' / 'report' / 'figures'
    out_dir = Path(os.environ.get('REPORT_DIR', default_out))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not SESSIONS:
        sys.exit(f'no alignment_results.pkl under {RESULTS_DIR}')
    print(f'{len(SESSIONS)} session(s) from {RESULTS_DIR}')
    full = load_results()
    figure_r1(full, out_dir)
    figure_rs1(full, out_dir)
    figure_r2(full, out_dir)
    print(f'report figures written to {out_dir}')


if __name__ == '__main__':
    main(sys.argv[1:])
