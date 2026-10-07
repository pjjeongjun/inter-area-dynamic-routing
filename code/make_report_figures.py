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
from make_alignment_figures import (FDR_ALPHA, GROUP_COLOR, MARKERS, PAIR_COLOR, PAIR_LABEL, PAIR_TYPES, REGION_COLOR,
                                    area_group, area_region, region_legend,
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


def swarm_x(y, center, dy=0.025, dx=0.09, half_width=0.27):
    """Deterministic beeswarm offsets: points whose y values fall within ``dy`` of one another are spread
    sideways by ``dx`` around ``center`` (alternating sides), so markers do not overlap."""
    y = np.asarray(y, dtype=float)
    order = np.argsort(y)
    x = np.full(len(y), float(center))
    placed = []                                   # (y, x) of points already placed
    for i in order:
        slot = 0
        while True:
            off = (slot + 1) // 2 * dx * (1 if slot % 2 else -1)
            if all(abs(y[i] - py) >= dy or abs(center + off - px) >= dx * 0.95 for py, px in placed) or abs(off) > half_width:
                break
            slot += 1
        x[i] = center + off
        placed.append((y[i], x[i]))
    return x


def stars(p):
    return '*' if p < 0.05 else 'n.s.'


def p_text(p):
    return 'p < 0.001' if p < 0.001 else (f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}')


OTHERS_BAR = '0.3'   # mean bar over all non-frontal areas (grey: the non-frontal regions have their own colours)


def group_bars_and_bracket(ax, x_fr, x_ot, mean_fr, mean_ot, p, yb):
    """Thick coloured bars at the two group means (spanning the groups' x ranges) and a bracket between the groups
    annotated with the test result."""

    for xs, mval, col in ((x_fr, mean_fr, GROUP_COLOR['frontal']), (x_ot, mean_ot, OTHERS_BAR)):
        ax.plot([min(xs) - 0.35, max(xs) + 0.35], [mval] * 2, color=col, lw=2.0, alpha=0.45, zorder=2, solid_capstyle='butt')
    for xs in (x_fr, x_ot):
        ax.plot([min(xs), min(xs), max(xs), max(xs)], [yb - 0.02, yb, yb, yb - 0.02], color='k', lw=0.8)
    c_fr, c_ot = np.mean([min(x_fr), max(x_fr)]), np.mean([min(x_ot), max(x_ot)])
    ax.plot([c_fr, c_fr, c_ot, c_ot], [yb + 0.01, yb + 0.04, yb + 0.04, yb + 0.01], color='k', lw=0.8)
    ax.text((c_fr + c_ot) / 2, yb + 0.04, stars(p), ha='center', va='bottom', fontsize=8 if p < 0.05 else 6.5)


def acc_area_panel(ax, areas_tab, order):
    """Context-axis (LDA) decoding accuracy (one held-out block per context, 9 folds) per area instance; filled = above every
    draw of the block-permutation null (grey bar: that instance's null range, min to max over the 18 block permutations).
    Instances of the same area (different sessions) are offset sideways. Coloured bars: mean over the frontal and the
    non-frontal area instances; bracket: Mann–Whitney test across area instances."""
    for i, a in enumerate(order):
        rows = areas_tab[areas_tab['area'] == a]
        offs = (np.arange(len(rows)) - (len(rows) - 1) / 2) * 0.26
        for (_, r), dx in zip(rows.iterrows(), offs):
            k = SESSIONS.index(r['session'])
            pred = bool(r['acc_predictive'])
            c = REGION_COLOR[area_region(a)]
            ax.plot([i + dx, i + dx], [r['acc_block_null_lo'], r['acc_block_null_hi']], color='0.8', lw=3.2 if len(rows) > 1 else 4,
                    solid_capstyle='butt', alpha=0.6, zorder=1)
            ax.errorbar(i + dx, r['acc_cv'], yerr=r['acc_cv_sd'], fmt=MARKERS[k % len(MARKERS)], ms=4.2, color=c, mfc=c if pred else 'white',
                        mec=c, mew=0.8, elinewidth=0.6, capsize=0, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    for lab, a in zip(ax.get_xticklabels(), order):
        lab.set_color(REGION_COLOR[area_region(a)])
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.axhline(0.5, color='k', ls='--', lw=0.7)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('Area (colour: region)')
    ax.set_ylabel('context decoding accuracy\n(held-out blocks)')
    # group means over area instances and the frontal vs. non-frontal comparison (Mann–Whitney across instances)
    grp = areas_tab['area'].map(area_group)
    fr, ot = areas_tab.loc[grp == 'frontal', 'acc_cv'], areas_tab.loc[grp != 'frontal', 'acc_cv']
    p_fr = stats.mannwhitneyu(fr, ot).pvalue
    i_fr = [i for i, a in enumerate(order) if area_group(a) == 'frontal']
    i_ot = [i for i, a in enumerate(order) if area_group(a) != 'frontal']
    group_bars_and_bracket(ax, i_fr, i_ot, fr.mean(), ot.mean(), p_fr, yb=0.9)
    ax.set_title(f'Context axis (LDA): context decoding\nfrontal {fr.mean():.0%} vs. others {ot.mean():.0%} over area instances, {p_text(p_fr)}',
                 loc='left')
    return p_fr


def figure_r1(results, out_dir):
    """Communication subspace: cross-validated R² per pair (A) and context decoded from it per pair (B); context axis
    (LDA): context decoding accuracy per area instance (C)."""
    tables = [qualify(r) for r in results]
    pooled = pd.concat(tables, ignore_index=True)
    areas_tab = pd.concat([r['area_table'] for r in results], ignore_index=True)

    fig = plt.figure(figsize=(W, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.62, wspace=0.42, left=0.12, right=0.985, top=0.93, bottom=0.1)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    sub = gs[1, :].subgridspec(1, 2, width_ratios=[1, 0.6], wspace=0.04)   # room for the region and session legends beside C
    axC = fig.add_subplot(sub[0, 0])
    axL = fig.add_subplot(sub[0, 1])
    axL.axis('off')

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
    h = [Line2D([], [], marker='o', color='0.4', ls='', ms=4, label='Above null'),
         Line2D([], [], marker='o', mfc='white', mec='0.4', ls='', ms=4, label='Not above null')]
    axC.legend(handles=h, loc='lower left', frameon=False, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)
    axC.add_artist(axC.get_legend())
    region_legend(axL, [area_region(a) for a in areas_tab['area'].unique()], loc='upper left', bbox_to_anchor=(0.0, 1.0),
                  handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0, fontsize=5.5, title_fontsize=6)
    session_legend(axL, 'upper right', bbox_to_anchor=(1.0, 1.0), handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0,
                   fontsize=5.5)

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
        sel = (pooled[area_col] == a).values
        for sid in SESSIONS:                              # one marker shape per session, as in the other panels
            m_ = sel & (pooled['session'] == sid).values
            if m_.any():
                ax.scatter(i + rng.uniform(-0.15, 0.15, int(m_.sum())), excess[m_].values, s=12,
                           marker=MARKERS[SESSIONS.index(sid) % len(MARKERS)], color=REGION_COLOR[area_region(a)],
                           edgecolor='k', linewidth=0.3, zorder=3)
    ax.axhline(0, color='k', ls='--', lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    for lab, a in zip(ax.get_xticklabels(), order):
        lab.set_color(REGION_COLOR[area_region(a)])
    ax.set_xlim(-0.6, len(order) - 0.4)
    frontal_mask = np.asarray(frontal_mask, dtype=bool)
    i_fr = [i for i, a in enumerate(order) if area_group(a) == 'frontal']
    i_ot = [i for i, a in enumerate(order) if area_group(a) != 'frontal']
    for g, mask, col in ((i_fr, frontal_mask, GROUP_COLOR['frontal']), (i_ot, ~frontal_mask, OTHERS_BAR)):
        if g and mask.any():        # group mean bar only when the group has pairs
            ax.plot([min(g) - 0.35, max(g) + 0.35], [excess[mask].mean()] * 2, color=col, lw=2.0, alpha=0.45,
                    zorder=2, solid_capstyle='butt')
    # frontal-vs-other comparison needs both groups (one session alone may have only one)
    if not i_fr or not i_ot or frontal_mask.sum() < 2 or (~frontal_mask).sum() < 2:
        return np.nan
    p_fr = stats.mannwhitneyu(excess[frontal_mask], excess[~frontal_mask]).pvalue
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
    ylab = 'Alignment\n(cosine − shuffled-label null)'

    fig = plt.figure(figsize=(W, 5.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.45], hspace=0.62, wspace=0.32, left=0.1, right=0.985, top=0.93, bottom=0.11)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axC = fig.add_subplot(gs[1, 0], sharey=axA)
    axL = fig.add_subplot(gs[0, 2])                      # session legend shared by A and B
    axL.axis('off')
    region_legend(axL, [area_region(a) for a in set(pooled['source']) | set(pooled_t['target'])], loc='center right',
                  bbox_to_anchor=(1.0, 0.5), handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0, fontsize=5.5, title_fontsize=6)
    axL2 = fig.add_subplot(gs[1, 2])                     # session legend (marker shapes used in every panel)
    axL2.axis('off')
    session_legend(axL2, 'center right', bbox_to_anchor=(1.0, 0.5), handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0,
                   fontsize=5.5)

    # A: source side, by source area; bracket = frontal vs. non-frontal sources
    order = area_order(pooled['source'].unique())
    alignment_by_area_panel(axA, pooled, 'source', pooled['excess'], order, pooled['pair_type'].str.startswith('frontal'))
    axA.set_ylim(-0.25, 0.8)
    axA.set_ylabel(ylab)
    axA.set_xlabel('Source area')
    n_up = int(((pooled['cos_shuf_q'] <= FDR_ALPHA) & (pooled['cos_shuf_z'] > 0)).sum())
    axA.set_title(f'Source side: alignment by area\n{n_up}/{len(pooled)} pairs above the null', loc='left')

    # B: target side, by target area; bracket = frontal vs. non-frontal targets
    ex_t = pooled_t['excess_tgt']
    order_t = area_order(pooled_t['target'].unique())
    alignment_by_area_panel(axB, pooled_t, 'target', ex_t, order_t, pooled_t['pair_type'].str.endswith('frontal'))
    axB.set_ylabel(ylab)
    axB.set_xlabel('Target area')
    n_up_t = int(((pooled_t['cos_tgt_shuf_q'] <= FDR_ALPHA) & (pooled_t['cos_tgt_shuf_z'] > 0)).sum())
    axB.set_title(f'Target side: alignment by area\n{n_up_t}/{len(pooled_t)} pairs above the null', loc='left')

    # C: source-side excess against the source axis's decoding accuracy, qualified pairs only
    for k, t in enumerate(tables):
        for g in GROUP_COLOR:
            sub = t[t['source'].map(area_group) == g]        # boolean Series: safe when t is empty
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
    axC.text(0.03, 0.97, f'ρ = {rho:.2f}, {p_text(pval)}', transform=axC.transAxes, va='top', fontsize=6.5)
    m.group_legend(axC, loc='lower right', handletextpad=0.3)

    axD = fig.add_subplot(gs[1, 1])
    cross_block_panel(axD)

    place_letters(fig, [axA, axB, axC, axD], 'ABCD', dx=-0.075)
    save(fig, out_dir / 'figure_R2_alignment.svg')


# ----------------------------------------------------------------------------- R1, panel B
def subspace_decoding_panel(ax, tables, pooled):
    """Context decoded from the communication subspace (held-out blocks), by pair type; filled = the pair's decoding accuracy
    is above every draw of its block-permutation null, open = not. Grey box: the pair type's block-permutation null (mean
    over pairs of the null's min and max). Coloured bars: mean over all pairs with a
    frontal / non-frontal source; bracket: frontal → vs. other → sources (Mann–Whitney over pairs). Asterisks under the
    x-axis labels: each pair type vs. its block-permutation null (Wilcoxon on accuracy − null mean)."""
    for i, typ in enumerate(PAIR_TYPES):
        sub_all = pooled[pooled['pair_type'] == typ]
        ax.fill_between([i - 0.45, i + 0.45], sub_all['sub_acc_block_null_lo'].mean(), sub_all['sub_acc_block_null_hi'].mean(),
                        color='0.88', lw=0, zorder=0)
        # one swarm per pair type (all sessions together, so points of different sessions do not overlap either)
        parts = [(k, t[t['pair_type'] == typ]) for k, t in enumerate(tables)]
        yall = np.concatenate([sub['sub_acc_cv'].values for _, sub in parts])
        xall = swarm_x(yall, i, dy=0.022, dx=0.08, half_width=0.42)
        pos = 0
        for k, sub in parts:
            n = len(sub)
            x, yv, q = xall[pos:pos + n], yall[pos:pos + n], sub['sub_acc_predictive'].values.astype(bool)
            pos += n
            ax.scatter(x[q], yv[q], s=14, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[typ], edgecolor='k', linewidth=0.3, zorder=3)
            ax.scatter(x[~q], yv[~q], s=14, marker=MARKERS[k % len(MARKERS)], color='white', edgecolor=PAIR_COLOR[typ], linewidth=0.7,
                       zorder=3)
    ax.set_xticks(range(len(PAIR_TYPES)))
    short_type_ticks(ax)
    ax.set_xlim(-0.6, len(PAIR_TYPES) - 0.2)
    ax.set_ylabel('context decoding accuracy\n(held-out blocks)')
    ax.axhline(0.5, color='k', ls='--', lw=0.7)
    ax.set_ylim(0.2, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('pair type (source → target)')
    # each pair type vs. its block-permutation null: mark next to the tick label
    labels = []
    for i, typ in enumerate(PAIR_TYPES):
        sub = pooled[pooled['pair_type'] == typ]
        p_typ = stats.wilcoxon(sub['sub_acc_cv'] - sub['sub_acc_block_null_mean']).pvalue
        labels.append(f'{SHORT[typ]}\n{"*" if p_typ < 0.05 else "n.s."}')
    ax.set_xticklabels(labels)
    # group means and the frontal → vs. other → comparison over pairs
    fr = pooled['pair_type'].str.startswith('frontal')
    p_fr = stats.mannwhitneyu(pooled['sub_acc_cv'][fr], pooled['sub_acc_cv'][~fr]).pvalue
    m_fr, m_ot = pooled['sub_acc_cv'][fr].mean(), pooled['sub_acc_cv'][~fr].mean()
    group_bars_and_bracket(ax, [0, 1], [2, 3], m_fr, m_ot, p_fr, yb=0.9)
    ax.set_title(f'Comm. subspace: context decoding\nfrontal→ {m_fr:.0%} vs. others→ {m_ot:.0%}, {p_text(p_fr)}', loc='left')
    h = [Line2D([], [], marker='o', color='0.4', ls='', ms=4, label='Above null'),
         Line2D([], [], marker='o', mfc='white', mec='0.4', ls='', ms=4, label='Not above null')]
    ax.legend(handles=h, loc='lower left', frameon=False, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)
    return p_fr


# ----------------------------------------------------------------------------- R2, panel D
def cross_block_panel(ax, d=None):
    """Cross-block control, qualified pairs: cosine between the task axis fit on the other three blocks and the
    communication subspace fit on the remaining three, vs. the shuffled-label axis null of that half-data subspace (an
    axis fit on the same three blocks with shuffled labels); lines join the same pair, black bars = means; bracket =
    paired two-sided Wilcoxon signed-rank test across pairs."""
    base = RESULTS_DIR if d is None else BUNDLE / d
    c = pd.read_csv(base / 'pooled' / 'controls_by_pair_pooled.csv')
    tables = [c[(c['session'] == s) & c['qualified']] for s in SESSIONS]
    pooled = pd.concat(tables, ignore_index=True)
    cols = ['cos_cross', 'cos_cross_null']
    rng = np.random.default_rng(0)
    for k, t in enumerate(tables):
        for _, r in t.iterrows():
            j = rng.uniform(-0.08, 0.08)
            y = [r[cc] for cc in cols]
            ax.plot(np.arange(2) + j, y, color=PAIR_COLOR[r['pair_type']], lw=0.45, alpha=0.5)
            ax.scatter(np.arange(2) + j, y, s=7, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[r['pair_type']], edgecolor='k', linewidth=0.2, zorder=3)
    ax.plot(np.arange(2) + 0.22, [pooled[cc].mean() for cc in cols], 'k_', ms=10, mew=1.5, zorder=4)
    p = stats.wilcoxon(pooled['cos_cross'], pooled['cos_cross_null']).pvalue
    yb = pooled[cols].values.max() + 0.06
    ax.plot([0, 0, 1, 1], [yb - 0.03, yb, yb, yb - 0.03], color='k', lw=0.8)
    ax.text(0.5, yb, '*' if p < 0.05 else 'n.s.', ha='center', va='bottom', fontsize=8 if p < 0.05 else 6.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(['axis from\nother blocks', 'null (shuffled\nlabels, same blocks)'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Cosine (axis, subspace)')
    n_up = int(((pooled['cos_cross_q'] <= FDR_ALPHA) & (pooled['cos_cross_z'] > 0)).sum())
    ax.set_title(f'Cross-block control\n{n_up}/{len(pooled)} pairs above the null', loc='left')
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
