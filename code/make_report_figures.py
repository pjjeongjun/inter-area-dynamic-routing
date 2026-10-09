"""Portrait-format (6.5 in wide) figures for the short report, drawn from the pooled outputs of one
results tree (by default results_3sessions/; set RESULTS_DIR=../results_11sessions for the eleven-session capsule run).

    python make_report_figures.py                                   # ../results_3sessions -> ../results_3sessions/report/figures/*.svg
    RESULTS_DIR=../results_11sessions python make_report_figures.py  # -> ../results_11sessions/report/figures
    ALIGN_QC_DIR=<dir> also writes PNG copies (the report builder reads report/figures_png)

Sessions are every ``<session>/alignment_results.pkl`` found under RESULTS_DIR (the three
local sessions, or all sessions of a capsule run). ``REPORT_DIR`` overrides the output folder.

Figure R1  communication subspace (cross-validated R², context decoding), per-area context decoding, and subspace minus
           source-area decoding (default run)
Figure S1  communication-subspace dimensionality (supplementary)
Figure R2  alignment by area (source, target) (default run)
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
from matplotlib import patheffects as pe
from matplotlib import transforms as mtrans
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy import stats

import make_alignment_figures as m
from make_alignment_figures import (FDR_ALPHA, GROUP_COLOR, MARKERS, PAIR_COLOR, PAIR_LABEL, PAIR_TYPES, REGION_COLOR,
                                    area_color, area_group, area_region, region_legend,
                                    area_order, place_letters, POINT_EDGE, point_color, qualify, r2_at_dimensionality, save, soft)

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'legend.fontsize': 6.2, 'legend.title_fontsize': 6.5})

BUNDLE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', BUNDLE / 'results_3sessions'))
SESSIONS = sorted(p.parent.name for p in RESULTS_DIR.glob('*/alignment_results.pkl'))
SHORT = {'frontal->frontal': 'Frontal\n→ Frontal', 'frontal->other': 'Frontal\n→ Others', 'other->frontal': 'Others\n→ Frontal',
         'other->other': 'Others\n→ Others'}
W = 6.5
READER = os.environ.get('SUBSPACE_READER', 'sub_acc_nc')   # reader for Figure 1B/1D: 'sub_acc_nc' = nearest centroid, 'sub_acc' = LDA
AREA_READER = 'acc_nc' if READER == 'sub_acc_nc' else 'acc'   # the same reader on all units of an area (Figure 1D)
READER_NAME = 'nearest centroid' if READER == 'sub_acc_nc' else 'LDA'


def load_results(d=None):
    base = RESULTS_DIR if d is None else BUNDLE / d
    return [pickle.load(open(base / s / 'alignment_results.pkl', 'rb')) for s in SESSIONS]


def short_type_ticks(ax):
    ax.set_xticklabels([SHORT[t] for t in PAIR_TYPES])


def strip(ax, tables, ycol, ylabel, zero_line=None, log=False, show_n=False, text_loc='top', color_by=None):
    m.strip_by_type_sessions(ax, tables, ycol, ylabel, zero_line=zero_line, log=log, show_n=show_n, text_loc=text_loc,
                             color_by=color_by)
    short_type_ticks(ax)


def session_legend(ax, loc, mouse_only=False, **kw):
    h = [Line2D([], [], marker=MARKERS[k % len(MARKERS)], color='0.5', ls='', ms=4, label=s.split('_')[0] if mouse_only else s)
         for k, s in enumerate(SESSIONS)]
    ax.legend(handles=h, loc=loc, frameon=False, title='mouse' if mouse_only else 'session', **kw)


def session_legend_layout(width_in, **kw):
    """(columns, height in inches) of session_legend with the most columns that fit width_in."""
    tmp = plt.figure(figsize=(width_in * 2, 3))
    ax = tmp.add_axes([0, 0, 1, 1])
    r = tmp.canvas.get_renderer()
    for ncol in range(len(SESSIONS), 0, -1):
        session_legend(ax, 'upper left', ncol=ncol, **kw)
        bb = ax.get_legend().get_window_extent(r)
        if bb.width / tmp.dpi <= width_in or ncol == 1:
            break
    plt.close(tmp)
    return ncol, bb.height / tmp.dpi


def pair_legend(ax, loc, **kw):
    h = [Line2D([], [], color=PAIR_COLOR[t], marker='o', ls='', ms=4, label=PAIR_LABEL[t]) for t in PAIR_TYPES]
    ax.legend(handles=h, loc=loc, frameon=False, title='source → target', **kw)


def scatter_types(ax, tables, xcol, ycol, s=14):
    for k, t in enumerate(tables):
        for typ in PAIR_TYPES:
            sub = t[t['pair_type'] == typ]
            ax.scatter(sub[xcol], sub[ycol], s=s, marker=MARKERS[k % len(MARKERS)], color=soft(PAIR_COLOR[typ]), edgecolor=PAIR_COLOR[typ], linewidth=POINT_EDGE)


# ----------------------------------------------------------------------------- R1
ACC_YLIM = (0.1, 0.9)   # shared by every decoding-accuracy panel
SWARM_S = 10            # marker size (pt²) of the per-pair points in Figure 1A-B


_U = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 180, endpoint=False)),
                      np.sin(np.linspace(0, 2 * np.pi, 180, endpoint=False))])   # directions for marker outlines


def marker_support(marker, size_pt, dpi, edge_pt=POINT_EDGE):
    """Support function (px, over the directions _U) of a marker's outer outline: its exact shape at size size_pt (sqrt(s)
    for scatter, ms for plot) plus half its edge line."""
    if marker == 'o':
        h = np.full(len(_U), size_pt / 2)
    else:
        from matplotlib.markers import MarkerStyle
        st = MarkerStyle(marker)
        h = (st.get_path().transformed(st.get_transform()).vertices * size_pt @ _U.T).max(axis=0)
    return (h + edge_pt / 2) * dpi / 72


def colour_group(area):
    """Legend entry that sets an area's colour: frontal subgroup, otherwise region."""
    return m.FRONTAL_SUBGROUP.get(area, 'Frontal: MOs') if area_group(area) == 'frontal' else area_region(area)


COLOUR_GROUP_ORDER = m.SUBGROUP_ORDER + m.REGION_ORDER + ['Unassigned']


def colour_group_x(areas, centre):
    """x per colour group present among areas, evenly spaced around centre in legend order."""
    present = set(map(colour_group, areas))
    groups = [g for g in COLOUR_GROUP_ORDER if g in present]
    step = min(0.12, 0.7 / max(len(groups), 1))
    return dict(zip(groups, centre + (np.arange(len(groups)) - (len(groups) - 1) / 2) * step))


def r2_swarm_panel(ax, tables, ycol, ylabel, color_by='target', s=SWARM_S):
    """Strip plot by pair type with one marker shape per session, as make_alignment_figures.strip_by_type_sessions, on a
    linear axis. Within a pair type, each colour group of the color_by area (frontal subgroup or region) has its own x,
    shared by all its pairs, in legend order (JeongJun, 2026-10-08); no mean bar. Returns the column centres (data x)."""
    pooled = pd.concat(tables, ignore_index=True)
    ax.set_ylim(0, pooled[ycol].max() * 1.18)            # linear axis (JeongJun, 2026-10-08: no log scale here)
    ax.set_xlim(-0.5, len(PAIR_TYPES) - 0.5)
    ax.axhline(0, color='k', ls='--', lw=0.7)
    centres = np.arange(len(PAIR_TYPES), dtype=float)
    for i, t in enumerate(PAIR_TYPES):
        sub = pooled[pooled['pair_type'] == t]
        gx = colour_group_x(sub[color_by], centres[i])
        for k, tab in enumerate(tables):
            st = tab[tab['pair_type'] == t]
            if len(st):
                ax.scatter(st[color_by].map(colour_group).map(gx), st[ycol], s=s, marker=MARKERS[k % len(MARKERS)],
                           c=[point_color(a) for a in st[color_by]], edgecolor=[area_color(a) for a in st[color_by]],
                           linewidth=POINT_EDGE, zorder=3)
    ax.set_xticks(centres)
    short_type_ticks(ax)
    ax.set_ylabel(ylabel)
    return centres


def stars(p):
    return '*' if p < 0.05 else 'n.s.'


def p_text(p):
    return 'p < 0.001' if p < 0.001 else (f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}')


OTHERS_BAR = GROUP_COLOR['other']   # mean bar over all non-frontal areas: the 'others' orange of the pair-type figures


def mean_bar(ax, xs, mval, col, pad=0.35):
    """Group-mean bar spanning the group's x range. Drawn above the markers, slightly transparent, with thin opaque black
    edges (offset so they do not overlap the coloured line and tint it) so the bar stays visible over a dense swarm."""
    edges = [pe.Stroke(offset=(0, dy), linewidth=0.5, foreground='k', alpha=1) for dy in (0.95, -0.95)]
    ax.plot([min(xs) - pad, max(xs) + pad], [mval] * 2, color=col, lw=1.4, alpha=0.75, zorder=4,
            solid_capstyle='butt', path_effects=edges + [pe.Normal()])


def group_bars_and_bracket(ax, x_fr, x_ot, mean_fr, mean_ot, p, yb, h=0.02, bar_x=None):
    """Thick coloured bars at the two group means (spanning the groups' x ranges, or bar_x = (x range of the frontal bar,
    x range of the other bar) when given) and a bracket between the groups annotated with the test result."""

    bars = ((x_fr, 0.35), (x_ot, 0.35)) if bar_x is None else ((bar_x[0], 0), (bar_x[1], 0))
    for (xs, pad), mval, col in zip(bars, (mean_fr, mean_ot), (GROUP_COLOR['frontal'], OTHERS_BAR)):
        mean_bar(ax, xs, mval, col, pad=pad)
    for xs in (x_fr, x_ot):
        ax.plot([min(xs), min(xs), max(xs), max(xs)], [yb - h, yb, yb, yb - h], color='k', lw=0.8)
    c_fr, c_ot = np.mean([min(x_fr), max(x_fr)]), np.mean([min(x_ot), max(x_ot)])
    yc = yb + 0.012
    ax.plot([c_fr, c_fr, c_ot, c_ot], [yc, yc + h, yc + h, yc], color='k', lw=0.8)
    ax.text((c_fr + c_ot) / 2, yc + h + 0.003, stars(p), ha='center', va='bottom', fontsize=8 if p < 0.05 else 6.5, zorder=5)


def acc_area_panel(ax, areas_tab, order, pre='acc'):
    """Context decoding accuracy from all 30 units of an area (stratified 10-fold held-out trials) per area instance,
    read out with the context axis (pre='acc': LDA, default) or nearest centroid (pre='acc_nc'); filled = above every
    draw of the block-permutation null (grey bar: that instance's null range, min to max over the 18 block permutations).
    Instances of the same area (different sessions) are offset sideways. Coloured bars: mean over the frontal and the
    non-frontal area instances; bracket: Mann–Whitney test across area instances."""
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0.2, 1.0)
    # instances of one area recorded in several sessions sit side by side, far enough apart (on screen) that a marker
    # never touches the neighbouring instance's null bar
    ms, lw_multi = 4.2, 2.4
    upx = ax.transData.transform((1, 0.5))[0] - ax.transData.transform((0, 0.5))[0]
    half_marker = max(marker_support(MARKERS[k % len(MARKERS)], ms, ax.figure.dpi)[0] for k in range(len(SESSIONS)))
    step = max(0.26, (half_marker + (lw_multi / 2 + 0.5) * ax.figure.dpi / 72) / upx)   # 0.5-pt clearance
    for i, a in enumerate(order):
        rows = areas_tab[areas_tab['area'] == a]
        offs = (np.arange(len(rows)) - (len(rows) - 1) / 2) * step
        for (_, r), dx in zip(rows.iterrows(), offs):
            k = SESSIONS.index(r['session'])
            pred = bool(r[f'{pre}_predictive'])
            c = point_color(a)
            ax.plot([i + dx, i + dx], [r[f'{pre}_block_null_lo'], r[f'{pre}_block_null_hi']], color='0.8',
                    lw=lw_multi if len(rows) > 1 else 4, solid_capstyle='butt', alpha=0.6, zorder=1)
            ax.errorbar(i + dx, r[f'{pre}_cv'], yerr=r[f'{pre}_cv_sd'], fmt=MARKERS[k % len(MARKERS)], ms=ms, color=c, mfc=c if pred else 'white',
                        mec=area_color(a), mew=POINT_EDGE, elinewidth=0.6, capsize=0, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    for lab, a in zip(ax.get_xticklabels(), order):
        lab.set_color(area_color(a))
    ax.axhline(0.5, color='k', ls='--', lw=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('Area (colour: region)')
    ax.set_ylabel('context decoding accuracy\n(held-out trials)')
    # group means over area instances and the frontal vs. non-frontal comparison (Mann–Whitney across instances)
    grp = areas_tab['area'].map(area_group)
    fr, ot = areas_tab.loc[grp == 'frontal', f'{pre}_cv'], areas_tab.loc[grp != 'frontal', f'{pre}_cv']
    p_fr = stats.mannwhitneyu(fr, ot).pvalue
    i_fr = [i for i, a in enumerate(order) if area_group(a) == 'frontal']
    i_ot = [i for i, a in enumerate(order) if area_group(a) != 'frontal']
    group_bars_and_bracket(ax, i_fr, i_ot, fr.mean(), ot.mean(), p_fr, yb=0.945, h=0.015)
    # each group vs. its block-permutation null (Wilcoxon on accuracy − null mean, over area instances); one common height
    y_mark = float((areas_tab[f'{pre}_cv'] + areas_tab[f'{pre}_cv_sd']).max()) + 0.012
    for xs, mask in ((i_fr, grp == 'frontal'), (i_ot, grp != 'frontal')):
        d = areas_tab.loc[mask, f'{pre}_cv'] - areas_tab.loc[mask, f'{pre}_block_null_mean']
        p_grp = stats.wilcoxon(d).pvalue
        ax.text(np.mean([min(xs), max(xs)]), y_mark, stars(p_grp), ha='center', va='bottom', fontsize=8 if p_grp < 0.05 else 6.5)
    head = 'Area population (nearest centroid)' if pre == 'acc_nc' else 'Context axis (LDA)'
    ax.set_title(f'{head}: context decoding\nfrontal {fr.mean():.0%} vs. others {ot.mean():.0%}, {p_text(p_fr)}', loc='left')
    return p_fr


def figure_r1(results, out_dir, color_by='target'):
    """Communication subspace: cross-validated R² per pair (A); context decoded from the subspace per pair with READER
    (nearest centroid by default) (B); context decoded from all 30 units of each area with the context axis (LDA) per area
    instance (C); context decoded from each source area's 30 units vs. from its communication subspace, AREA_READER /
    READER (both nearest centroid by default), one line per source area instance, next to C on the same y scale (D).
    color_by: 'target' (default) or 'source' -- which area of the pair colours the points in A-B."""
    tables = [add_source_reader(qualify(r), r) for r in results]
    pooled = pd.concat(tables, ignore_index=True)
    areas_tab = pd.concat([r['area_table'] for r in results], ignore_index=True)

    # the session legend gets its own row below the region and null legends, with as many columns as fit the width; the
    # figure grows by that row's height so the panels keep their size
    kw = dict(handletextpad=0.3, labelspacing=0.25, columnspacing=1.0, borderaxespad=0.0, fontsize=5.5)
    leg_w = 0.865 * W
    ses_ncol, ses_h = session_legend_layout(leg_w, title_fontsize=6, **kw)
    H = 8.0 + ses_h + 0.08                                # 8.0 in: panels plus region / null legends
    fy = lambda y_in: y_in / H                            # figure fraction from inches above the bottom edge
    fig = plt.figure(figsize=(W, H))                      # near-square panels
    gs = fig.add_gridspec(2, 2, width_ratios=[0.9, 1.1], hspace=0.62, wspace=0.3, left=0.12, right=0.985,
                          top=1 - fy(0.055 * 8.0), bottom=fy(0.19 * 8.0 + ses_h + 0.08))
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])   # same columns as A and B, so edges and letters align
    axL = fig.add_axes([0.12, fy(0.04), 0.865, fy(0.09 * 8.0 + ses_h + 0.04)])   # region, null and session legends
    axL.axis('off')

    cA = r2_swarm_panel(axA, tables, 'r2_cv_dim', 'cross-validated R²\n(rank-d fit, held-out trials)', color_by=color_by)
    axA.set_xlabel('source → target')
    counts = pooled['pair_type'].value_counts()
    trA = mtrans.blended_transform_factory(axA.transData, axA.transAxes)
    for i, t in enumerate(PAIR_TYPES):                  # n per pair type, in the empty band at the top of the axis
        axA.text(cA[i], 0.97, f'n = {counts.get(t, 0)}', transform=trA, ha='center', va='top', fontsize=6)
    n_sig = int(pooled['r2_significant'].sum())
    axA.set_title(f'Comm. subspace: cross-validated R²\n{n_sig}/{len(pooled)} pairs > trial-shuffle null', loc='left')

    # B: subspace read out with READER (nearest centroid by default: no fitted weights)
    subspace_decoding_panel(axB, tables, pooled, color_by=color_by, prefix=READER)
    second = axB.get_title(loc='left').split('\n')[-1]
    axB.set_title(f'Comm. subspace ({READER_NAME})\n{second}', loc='left')

    acc_area_panel(axC, areas_tab, area_order(areas_tab['area'].unique()), pre='acc')   # C: context axis (LDA)
    population_vs_subspace_panel(axD, pooled)

    # region and null legends side by side; session legend in its own row below them
    reg = region_legend(axL, areas_tab['area'].unique(), loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=3, title_fontsize=6, **kw)
    h = [Line2D([], [], marker='o', color='0.4', ls='', ms=4, label='Above null'),
         Line2D([], [], marker='o', mfc='white', mec='0.4', ls='', ms=4, label='Not above null')]
    nul = axL.legend(handles=h, loc='upper left', bbox_to_anchor=(0.86, 1.0), frameon=False, title=' ', title_fontsize=6, **kw)
    axL.add_artist(nul)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y_ses = axL.transAxes.inverted().transform((0, reg.get_window_extent(r).y0))[1] - fy(0.08) / axL.get_position().height
    session_legend(axL, 'upper left', bbox_to_anchor=(0.0, y_ses), title_fontsize=6, ncol=ses_ncol, **kw)
    axL.get_legend().get_title().set_text('Session')     # add_artist legends are not capitalized by place_letters
    # bottom labels of the second row level: D's group labels at the height of C's x-axis label (same font size)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top_c = axC.xaxis.label.get_window_extent(r).y1
    for t in axD.texts:
        if t.get_text() in ('Frontal source', 'Non-frontal source'):
            t.xyann = (0, -(axD.get_window_extent(r).y0 - top_c) * 72 / fig.dpi)

    place_letters(fig, [axA, axB, axC, axD], 'ABCD', dx=-0.075)
    save(fig, out_dir / ('figure_R1_subspace_and_decoding.svg' if color_by == 'target' else f'figure_R1_subspace_and_decoding_by_{color_by}.svg'))


def subspace_vs_axis_panel(ax, tables, pooled):
    """Context decoding from the communication subspace (y) against context decoding from the source's context axis, i.e.
    the LDA on all 30 source units (x), one point per pair; colour = source group, marker = session, filled = source axis
    predictive. Dashed line: identity. Text: Spearman correlation and mean difference across pairs."""
    for k, t in enumerate(tables):
        for g in GROUP_COLOR:
            sub = t[t['source'].map(area_group) == g]
            q = sub['source_axis_predictive'].values.astype(bool)
            for mask, mfc in ((q, GROUP_COLOR[g]), (~q, 'white')):
                ax.scatter(sub['source_acc_cv'].values[mask], sub['sub_acc_cv'].values[mask], s=14, marker=MARKERS[k % len(MARKERS)],
                           color=mfc, edgecolor=GROUP_COLOR[g] if mfc == 'white' else 'k', linewidth=0.7 if mfc == 'white' else 0.3,
                           zorder=3)
    lo, hi = 0.4, 0.9
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7, zorder=1)
    ax.axhline(0.5, color='0.6', lw=0.5, ls=':')
    ax.axvline(0.5, color='0.6', lw=0.5, ls=':')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel('context decoding accuracy\ncontext axis (LDA, 30 units)')
    ax.set_ylabel('context decoding accuracy\ncomm. subspace')
    rho, pval = stats.spearmanr(pooled['source_acc_cv'], pooled['sub_acc_cv'])
    diff = (pooled['sub_acc_cv'] - pooled['source_acc_cv']).mean()
    ax.set_title(f'Subspace vs. context-axis decoding\nρ = {rho:.2f}, {p_text(pval)}; subspace − axis = {diff:+.2f}', loc='left')
    h = [Line2D([], [], marker='o', color=GROUP_COLOR['frontal'], ls='', ms=4, label='Frontal source'),
         Line2D([], [], marker='o', color=GROUP_COLOR['other'], ls='', ms=4, label='Other source'),
         Line2D([], [], marker='o', mfc='white', mec='0.4', ls='', ms=4, label='Source axis not predictive')]
    ax.legend(handles=h, loc='lower right', frameon=False, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1, fontsize=5.8)
    return rho, diff


def add_source_reader(pt, res):
    """Pair table plus the source area's all-unit decoding accuracy and block-null flag for AREA_READER."""
    at = res['area_table'].set_index('area')
    pt['source_reader_acc'] = at.loc[pt['source'], f'{AREA_READER}_cv'].values
    pt['source_reader_predictive'] = at.loc[pt['source'], f'{AREA_READER}_predictive'].values.astype(bool)
    return pt


POP_SUB_X = {True: (0.0, 1.0), False: (2.1, 3.1)}   # (population, subspace) x positions: frontal, non-frontal source


def population_vs_subspace_panel(ax, pooled):
    """Does restricting a source area to its communication subspace lose context? One line per source area instance:
    left end = context decoding from all 30 source units (AREA_READER), right end = from the source activity projected
    onto the communication subspace (READER), averaged over that source's targets; same reader and held-out trials on
    both ends. Filled = the source's all-unit decoding is above every block-permutation draw, open = not; marker =
    session. Each session's lines share one sideways offset. Bold lines: mean over all sources of each group (JeongJun, 2026-10-08: no filter on
    the source's own decoding); brackets: frontal vs. non-frontal sources for the population and for the subspace accuracy
    (Mann–Whitney across source instances); title: mean change (subspace − population) per group, same test."""
    q = pooled.copy()
    q['frontal_src'] = q['pair_type'].str.startswith('frontal')
    g = (q.groupby(['session', 'source'], sort=False)
         .agg(sub=(f'{READER}_cv', 'mean'), pop=('source_reader_acc', 'first'), pred=('source_reader_predictive', 'first'),
              frontal=('frontal_src', 'first'))
         .reset_index())
    g['d'] = g['sub'] - g['pop']
    ax.set_ylim(0.2, 1.0)                                     # same range as panels B and C
    ax.set_xlim(-0.5, 3.6)
    for fr in (True, False):
        col = GROUP_COLOR['frontal' if fr else 'other']
        x0, x1 = POP_SUB_X[fr]
        rows = g[g['frontal'] == fr].sort_values('pred').copy()   # open (not above null) first, filled on top
        step = min(0.12, 0.7 / len(SESSIONS))                  # session offsets stay within ±0.35 of the column
        rows['dx'] = [(SESSIONS.index(sid) - (len(SESSIONS) - 1) / 2) * step for sid in rows['session']]   # one x per session
        for _, r in rows.iterrows():
            k = SESSIONS.index(r['session'])
            dx = r['dx']
            pred = bool(r['pred'])
            ax.plot([x0 + dx, x1 + dx], [r['pop'], r['sub']], color=col, lw=0.7, alpha=0.55 if pred else 0.3, zorder=2,
                    solid_capstyle='round')
            ax.scatter([x0 + dx, x1 + dx], [r['pop'], r['sub']], s=14, marker=MARKERS[k % len(MARKERS)],
                       color=soft(col) if pred else 'white', edgecolor=col, linewidth=POINT_EDGE, zorder=3)
        ax.plot([x0, x1], [rows['pop'].mean(), rows['sub'].mean()], color=col, lw=1.6, zorder=5, solid_capstyle='butt',
                path_effects=[pe.Stroke(linewidth=2.6, foreground='k'), pe.Normal()])
    d_fr, d_ot = g.loc[g['frontal'], 'd'], g.loc[~g['frontal'], 'd']
    p_mw = stats.mannwhitneyu(d_fr, d_ot).pvalue
    # frontal vs. non-frontal sources within each readout: population bracket below, subspace bracket above it
    y_top, h = float(g[['pop', 'sub']].max().max()), 0.015
    for j, (col, yb) in enumerate((('pop', y_top + 0.035), ('sub', y_top + 0.105))):
        p_t = stats.mannwhitneyu(g.loc[g['frontal'], col], g.loc[~g['frontal'], col]).pvalue
        xa, xb = POP_SUB_X[True][j], POP_SUB_X[False][j]
        ax.plot([xa, xa, xb, xb], [yb - h, yb, yb, yb - h], color='k', lw=0.8)
        ax.text((xa + xb) / 2, yb + 0.003, stars(p_t), ha='center', va='bottom', fontsize=8 if p_t < 0.05 else 6.5)
    ax.axhline(0.5, color='k', ls='--', lw=0.7, zorder=1)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks([*POP_SUB_X[True], *POP_SUB_X[False]])
    dim = pooled['dim'].median()
    ax.set_xticklabels(['Population\n(30 units)', f'Subspace\n(~{dim:.0f} dims)'] * 2)
    # group labels a fixed distance below the two-line tick labels (tick length + pad + two text lines + gap, in points)
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    rc = plt.rcParams
    below = rc['xtick.major.size'] + rc['xtick.major.pad'] + 2 * 1.2 * plt.matplotlib.font_manager.FontProperties(
        size=rc['xtick.labelsize']).get_size_in_points() + 3
    for fr, lab in ((True, 'Frontal source'), (False, 'Non-frontal source')):
        ax.annotate(lab, xy=(np.mean(POP_SUB_X[fr]), 0), xycoords=tr, xytext=(0, -below), textcoords='offset points',
                    ha='center', va='top', fontsize=7, color=GROUP_COLOR['frontal' if fr else 'other'])
    ax.set_ylabel('context decoding accuracy\n(held-out trials)')
    ax.set_title(f'Population vs. comm. subspace\n'
                 f'Frontal {100 * d_fr.mean():+.1f} vs. others {100 * d_ot.mean():+.1f} pts, {p_text(p_mw)}', loc='left')
    return p_mw


def figure_rs1(results, out_dir):
    """Supplementary Figure 1: how the communication-subspace dimensionality is chosen. Colour = region / frontal
    subgroup of the target area, as in Figures R1 and R2."""
    tables = [qualify(r) for r in results]
    pooled = pd.concat(tables, ignore_index=True)
    n = results[0]['min_units']
    fig = plt.figure(figsize=(W, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5], wspace=0.36, left=0.09, right=0.99, top=0.84, bottom=0.2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axL = fig.add_subplot(gs[0, 2])
    axL.axis('off')
    ranks = np.arange(1, n + 1)
    for res, tab in zip(results, tables):
        curves = res['r2_curve'].mean(axis=0)
        for p in range(len(res['pairs'])):
            tgt = tab.iloc[p]['target']
            axA.plot(ranks, curves[p], color=area_color(tgt), lw=0.45, alpha=0.8)
            d = tab.iloc[p]['dim']
            axA.plot(d, np.interp(d, ranks, curves[p]), marker='o', ms=2.2, color=point_color(tgt), mec=area_color(tgt), mew=POINT_EDGE, ls='')
    axA.axhline(0, color='k', lw=0.6, ls=':')
    axA.set_xlim(0.5, n + 0.5)
    axA.set_xlabel('Rank')
    axA.set_ylabel('R² (cross-validated)')
    axA.set_title('R² vs. rank', loc='left')
    strip(axB, tables, 'dim', 'Dimensionality', color_by='target')
    axB.set_xlabel('Source → target')
    axB.set_ylim(0, 12)
    axB.set_title('Dimensionality', loc='left')
    region_legend(axL, set(pooled['target']), loc='center left', bbox_to_anchor=(0.0, 0.5), handletextpad=0.3,
                  labelspacing=0.3, borderaxespad=0.0, fontsize=5.5, title_fontsize=6)
    place_letters(fig, [axA, axB], 'AB', dx=-0.075)
    save(fig, out_dir / 'figure_S1_dimensionality.svg')


# ----------------------------------------------------------------------------- R2
def excess_strip(ax, tables, col, chance_col, ylabel):
    """Per-pair excess (observed − chance cosine) by pair type, one marker shape per session, black = mean ± SEM."""
    tabs = [t.assign(_ex=t[col] - t[chance_col]) for t in tables]
    strip(ax, tabs, '_ex', ylabel, zero_line=0)
    return pd.concat(tabs, ignore_index=True)


def alignment_by_area_panel(ax, pooled, area_col, excess, order, frontal_mask, yb=0.74, color_col=None):
    color_col = color_col or area_col                    # which area of the pair gives the marker colour
    """Alignment excess (observed − chance cosine) per area, one point per qualified pair; coloured bars = mean over all
    pairs whose plotted area is frontal / not frontal, compared by the bracket (Mann–Whitney)."""
    rng = np.random.default_rng(0)
    for i, a in enumerate(order):
        sel = (pooled[area_col] == a).values
        for sid in SESSIONS:                              # one marker shape per session, as in the other panels
            m_ = sel & (pooled['session'] == sid).values
            if m_.any():
                ax.scatter(i + rng.uniform(-0.15, 0.15, int(m_.sum())), excess[m_].values, s=12,
                           marker=MARKERS[SESSIONS.index(sid) % len(MARKERS)],
                           c=[point_color(b) for b in pooled.loc[m_, color_col]],
                           edgecolor=[area_color(b) for b in pooled.loc[m_, color_col]], linewidth=POINT_EDGE, zorder=3)
    ax.axhline(0, color='k', ls='--', lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    for lab, a in zip(ax.get_xticklabels(), order):
        lab.set_color(area_color(a))
    ax.set_xlim(-0.6, len(order) - 0.4)
    frontal_mask = np.asarray(frontal_mask, dtype=bool)
    i_fr = [i for i, a in enumerate(order) if area_group(a) == 'frontal']
    i_ot = [i for i, a in enumerate(order) if area_group(a) != 'frontal']
    for g, mask, col in ((i_fr, frontal_mask, GROUP_COLOR['frontal']), (i_ot, ~frontal_mask, OTHERS_BAR)):
        if g and mask.any():        # group mean bar only when the group has pairs
            mean_bar(ax, g, excess[mask].mean(), col)
    # frontal-vs-other comparison needs both groups (one session alone may have only one)
    if not i_fr or not i_ot or frontal_mask.sum() < 2 or (~frontal_mask).sum() < 2:
        return np.nan
    p_fr = stats.mannwhitneyu(excess[frontal_mask], excess[~frontal_mask]).pvalue
    yb = max(yb, float(excess.max()) + 0.06)
    for g in (i_fr, i_ot):
        ax.plot([min(g), min(g), max(g), max(g)], [yb - 0.025, yb, yb, yb - 0.025], color='k', lw=0.8)
    c_fr, c_ot = np.mean([min(i_fr), max(i_fr)]), np.mean([min(i_ot), max(i_ot)])
    ax.plot([c_fr, c_fr, c_ot, c_ot], [yb + 0.01, yb + 0.05, yb + 0.05, yb + 0.01], color='k', lw=0.8)
    ax.text((c_fr + c_ot) / 2, yb + 0.05, '*' if p_fr < 0.05 else 'n.s.', ha='center', va='bottom',
            fontsize=8 if p_fr < 0.05 else 6.5)
    return p_fr


def figure_r2(results, out_dir, color_by='target'):
    """color_by='source' colours panel B (x = target area) by the pair's source area instead of the target itself."""
    tables_all = [qualify(r) for r in results]
    pooled_all = pd.concat(tables_all, ignore_index=True)
    tables = [t[t['qualified']] for t in tables_all]
    pooled = pooled_all[pooled_all['qualified']]
    pooled_t = pooled_all[pooled_all['qualified_tgt']]
    ylab = 'Alignment\n(cosine − shuffled-label null)'

    fig = plt.figure(figsize=(W, 5.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.45], hspace=0.62, wspace=0.18, left=0.1, right=0.985, top=0.93, bottom=0.11)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axC = fig.add_subplot(gs[1, 0:2], sharey=axA)             # full bottom row (panel D was removed)
    axL = fig.add_subplot(gs[0, 2])                      # session legend shared by A and B
    axL.axis('off')
    region_legend(axL, set(pooled['source']) | set(pooled_t['target']), loc='center right',
                  bbox_to_anchor=(1.0, 0.5), handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0, fontsize=5.5, title_fontsize=6)
    axL2 = fig.add_subplot(gs[1, 2])                     # session legend (marker shapes used in every panel)
    axL2.axis('off')
    session_legend(axL2, 'center right', bbox_to_anchor=(1.0, 0.5), handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0,
                   fontsize=5.5)

    # A: source side, by source area; bracket = frontal vs. non-frontal sources
    order = area_order(pooled['source'].unique())
    alignment_by_area_panel(axA, pooled, 'source', pooled['excess'], order, pooled['pair_type'].str.startswith('frontal'))
    axA.set_ylim(-0.2, 0.86)
    axA.set_ylabel(ylab)
    axA.set_xlabel('Source area')
    n_up = int(((pooled['cos_shuf_q'] <= FDR_ALPHA) & (pooled['cos_shuf_z'] > 0)).sum())
    axA.set_title(f'Source side: alignment by area\n{n_up}/{len(pooled)} pairs above the null', loc='left')

    # B: target side, by target area; bracket = frontal vs. non-frontal targets
    ex_t = pooled_t['excess_tgt']
    order_t = area_order(pooled_t['target'].unique())
    alignment_by_area_panel(axB, pooled_t, 'target', ex_t, order_t, pooled_t['pair_type'].str.endswith('frontal'),
                            color_col='source' if color_by == 'source' else 'target')
    axB.set_ylabel('')                                   # y axis shared with A (label drawn once)
    axB.set_xlabel('Target area')
    n_up_t = int(((pooled_t['cos_tgt_shuf_q'] <= FDR_ALPHA) & (pooled_t['cos_tgt_shuf_z'] > 0)).sum())
    axB.set_title(f'Target side: alignment by area\n{n_up_t}/{len(pooled_t)} pairs above the null', loc='left')

    # C: source-side excess against the source axis's decoding accuracy, qualified pairs only
    for k, t in enumerate(tables):
        if len(t):                                             # colour = region / frontal subgroup of the source area
            axC.scatter(t['source_acc_cv'], t['excess'], s=14, marker=MARKERS[k % len(MARKERS)],
                        c=[point_color(a) for a in t['source']], edgecolor=[area_color(a) for a in t['source']], linewidth=POINT_EDGE)
    axC.axhline(0, color='k', ls='--', lw=0.7)
    rho, pval = stats.spearmanr(pooled['source_acc_cv'], pooled['excess'])
    fit = stats.linregress(pooled['source_acc_cv'], pooled['excess'])
    xfit = np.array([pooled['source_acc_cv'].min(), pooled['source_acc_cv'].max()])
    axC.plot(xfit, fit.intercept + fit.slope * xfit, color='k', lw=1.1, zorder=4)
    axC.set_xlabel('Context decoding accuracy (source area)')
    axC.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axC.xaxis.set_major_locator(MultipleLocator(0.05))
    axC.set_ylabel(ylab)
    axC.set_title('Alignment vs. context decoding', loc='left')
    axC.text(0.03, 0.97, f'ρ = {rho:.2f}, {p_text(pval)}', transform=axC.transAxes, va='top', fontsize=6.5)

    place_letters(fig, [axA, axB, axC], 'ABC', dx=-0.075)
    save(fig, out_dir / ('figure_R2_alignment.svg' if color_by == 'target' else f'figure_R2_alignment_by_{color_by}.svg'))


# ----------------------------------------------------------------------------- R1, panel B
N_SOURCE_PERMUTATIONS = 20000


def source_label_permutation_p(df, ycol, n=N_SOURCE_PERMUTATIONS, seed=0):
    """Two-sided p of the frontal − non-frontal difference in pair means of ``ycol``. Null: the frontal / non-frontal label
    is permuted across the source areas of each session, so all pairs of one source keep one label together (pairs that
    share a source are not independent)."""
    src = df[['session', 'source']].astype(str).agg('|'.join, axis=1)
    keys, pair_src = np.unique(src.values, return_inverse=True)
    src_fr = df.groupby(src)['pair_type'].first().loc[keys].str.startswith('frontal').values
    src_sess = df.groupby(src)['session'].first().loc[keys].values
    y = df[ycol].values.astype(float)
    obs_lab = src_fr[pair_src]
    obs = y[obs_lab].mean() - y[~obs_lab].mean()
    rng = np.random.default_rng(seed)
    lab = np.tile(src_fr, (n, 1))
    for s in np.unique(src_sess):
        idx = np.flatnonzero(src_sess == s)
        lab[:, idx] = rng.permuted(lab[:, idx], axis=1)
    m = lab[:, pair_src]                                             # draws x pairs
    n_fr = m.sum(axis=1)
    ok = (n_fr > 0) & (n_fr < len(y))
    d = (m @ y)[ok] / n_fr[ok] - ((~m) @ y)[ok] / (len(y) - n_fr[ok])
    return (np.sum(np.abs(d) >= abs(obs) - 1e-12) + 1) / (ok.sum() + 1)


def subspace_decoding_panel(ax, tables, pooled, color_by='target', prefix='sub_acc'):
    """Context decoded from the communication subspace (prefix selects the reader: sub_acc = LDA, sub_acc_nc = nearest centroid) (held-out trials), by pair type; filled = the pair's decoding accuracy
    is above every draw of its block-permutation null, open = not. Grey box: the pair type's block-permutation null (mean
    over pairs of the null's min and max). Coloured bars: mean over all pairs (JeongJun, 2026-10-08: no filter on the
    source's own decoding), frontal / non-frontal source; bracket: their difference, p from permutations of the frontal label across source areas within each session
    (source_label_permutation_p). Asterisks under the x-axis labels: each pair type vs. its block-permutation null
    (Wilcoxon on accuracy − null mean, all pairs)."""
    ax.set_ylim(0.2, 1.0)
    # within a pair type, each colour group of the color_by area has its own x, shared by all its pairs (as Figure 1A)
    ax.set_xlim(-0.5, len(PAIR_TYPES) - 0.5)
    centres = np.arange(len(PAIR_TYPES), dtype=float)
    half = np.zeros(len(PAIR_TYPES))                          # half-width of each column's groups (data x)
    for i, typ in enumerate(PAIR_TYPES):
        sub_all = pooled[pooled['pair_type'] == typ]
        gx = colour_group_x(sub_all[color_by], centres[i])
        half[i] = max(0.12, (max(gx.values()) - min(gx.values())) / 2 + 0.06)
        ax.fill_between([centres[i] - 0.45, centres[i] + 0.45], sub_all[f'{prefix}_block_null_lo'].mean(),
                        sub_all[f'{prefix}_block_null_hi'].mean(), color='0.88', lw=0, zorder=0)
        for k, t in enumerate(tables):
            sub = t[t['pair_type'] == typ]
            x, yv, q = sub[color_by].map(colour_group).map(gx).values, sub[f'{prefix}_cv'].values, sub[f'{prefix}_predictive'].values.astype(bool)
            ax.scatter(x[q], yv[q], s=SWARM_S, marker=MARKERS[k % len(MARKERS)], c=[point_color(t_) for t_ in sub[color_by].values[q]], edgecolor=[area_color(t_) for t_ in sub[color_by].values[q]], linewidth=POINT_EDGE, zorder=3)
            ax.scatter(x[~q], yv[~q], s=SWARM_S, marker=MARKERS[k % len(MARKERS)], color='white', edgecolor=[area_color(t_) for t_ in sub[color_by].values[~q]], linewidth=POINT_EDGE,
                       zorder=3)
    ax.set_xticks(centres)
    short_type_ticks(ax)
    ax.set_ylabel('context decoding accuracy\n(held-out trials)')
    ax.axhline(0.5, color='k', ls='--', lw=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('source → target')
    # each pair type vs. its block-permutation null: mark next to the tick label
    y_mark = pooled[f'{prefix}_cv'].max() + 0.012          # one common height so the marks do not encode a value
    for i, typ in enumerate(PAIR_TYPES):                 # each pair type vs. its block-permutation null
        sub = pooled[pooled['pair_type'] == typ]
        p_typ = stats.wilcoxon(sub[f'{prefix}_cv'] - sub[f'{prefix}_block_null_mean']).pvalue
        ax.text(centres[i], y_mark, stars(p_typ), ha='center', va='bottom', fontsize=8 if p_typ < 0.05 else 6.5)
    ax.set_xticklabels([SHORT[typ] for typ in PAIR_TYPES])
    # group means over all pairs, frontal → vs. other → sources; p from permuting the frontal label across source areas
    # within session (pairs of one source move together)
    q = pooled
    fr = q['pair_type'].str.startswith('frontal')
    p_fr = source_label_permutation_p(q, f'{prefix}_cv')
    m_fr, m_ot = q[f'{prefix}_cv'][fr].mean(), q[f'{prefix}_cv'][~fr].mean()
    span = [(centres[i] - half[i], centres[i] + half[i]) for i in range(len(PAIR_TYPES))]
    group_bars_and_bracket(ax, centres[:2], centres[2:], m_fr, m_ot, p_fr, yb=0.945, h=0.015,
                           bar_x=([span[0][0], span[1][1]], [span[2][0], span[3][1]]))
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
            ax.scatter(np.arange(2) + j, y, s=7, marker=MARKERS[k % len(MARKERS)], color=soft(PAIR_COLOR[r['pair_type']]), edgecolor=PAIR_COLOR[r['pair_type']], linewidth=POINT_EDGE, zorder=3)
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
    default_out = RESULTS_DIR / 'report' / 'figures'
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
