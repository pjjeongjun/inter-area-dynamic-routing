"""Figures for the task-axis vs. communication-subspace analysis.

    RESULTS_DIR=../results python make_alignment_figures.py [session_id ...]

Per session (``results/<session>/figures/``):
    figure1_<session>.svg  task axis and communication subspace are meaningful
    figure2_<session>.svg  alignment between them, all ordered pairs, vs. chance
With more than one session a pooled figure is also written to
``results/pooled/figure1_pooled.svg`` and ``figure2_pooled.svg``, with a pooled pair table.
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
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from task_axis_comm_subspace import FDR_ALPHA, area_group, fdr_bh, p_two_sided

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
})

GROUP_COLOR = {'frontal': '#1f6fb2', 'other': '#d9772b'}
PAIR_TYPES = ['frontal->frontal', 'frontal->other', 'other->frontal', 'other->other']
PAIR_LABEL = {'frontal->frontal': 'Frontal → Frontal', 'frontal->other': 'Frontal → Others',
              'other->frontal': 'Others → Frontal', 'other->other': 'Others → Others'}
PAIR_COLOR = {'frontal->frontal': '#1f6fb2', 'frontal->other': '#6fb0dd',
              'other->frontal': '#e8a86a', 'other->other': '#d9772b'}
EXTRA_PNG_DIR = os.environ.get('ALIGN_QC_DIR')  # optional PNG previews outside the results tree

# Finer anatomical grouping of the recorded areas (Allen CCF ontology), used to colour and order the
# by-area panels. Statistics still compare frontal cortex against everything else.
REGION_ORDER = ['Frontal cortex', 'Other neocortex', 'Olfactory cortex', 'Hippocampal formation',
                'Striatum / Septum', 'Thalamus', 'Midbrain']
REGION_OF = {
    # neocortex outside the frontal set: sensory, motor and association areas
    'MOp': 'Other neocortex', 'SSp': 'Other neocortex', 'SSs': 'Other neocortex', 'VISp': 'Other neocortex',
    'VISC': 'Other neocortex', 'AUDp': 'Other neocortex', 'AUDpo': 'Other neocortex', 'AUDv': 'Other neocortex',
    'AUDd': 'Other neocortex', 'TEa': 'Other neocortex', 'ECT': 'Other neocortex', 'PERI': 'Other neocortex',
    'RSPd': 'Other neocortex', 'RSPv': 'Other neocortex', 'RSPagl': 'Other neocortex', 'PTLp': 'Other neocortex',
    'VISa': 'Other neocortex', 'VISam': 'Other neocortex', 'VISrl': 'Other neocortex', 'VISl': 'Other neocortex',
    # olfactory areas (CCF 'OLF'): piriform, taenia tecta, dorsal peduncular, olfactory tubercle, AON
    'OLF': 'Olfactory cortex', 'PIR': 'Olfactory cortex', 'TTd': 'Olfactory cortex', 'TTv': 'Olfactory cortex',
    'DP': 'Olfactory cortex', 'OT': 'Olfactory cortex', 'AON': 'Olfactory cortex', 'NLOT': 'Olfactory cortex',
    # hippocampal formation (hippocampal and retrohippocampal regions)
    'CA1': 'Hippocampal formation', 'CA2': 'Hippocampal formation', 'CA3': 'Hippocampal formation',
    'DG': 'Hippocampal formation', 'ProS': 'Hippocampal formation', 'SUB': 'Hippocampal formation',
    'POST': 'Hippocampal formation', 'PRE': 'Hippocampal formation', 'ENTl': 'Hippocampal formation',
    'ENTm': 'Hippocampal formation',
    # striatum and lateral septal complex
    'CP': 'Striatum / Septum', 'ACB': 'Striatum / Septum', 'LSr': 'Striatum / Septum', 'LSc': 'Striatum / Septum',
    'LSv': 'Striatum / Septum', 'FS': 'Striatum / Septum',
    # thalamus
    'MGv': 'Thalamus', 'MGd': 'Thalamus', 'MGm': 'Thalamus', 'PoT': 'Thalamus', 'PO': 'Thalamus', 'LP': 'Thalamus',
    'LGd': 'Thalamus', 'VPM': 'Thalamus', 'VPL': 'Thalamus', 'MD': 'Thalamus', 'VAL': 'Thalamus', 'VM': 'Thalamus',
    # midbrain
    'MRN': 'Midbrain', 'RN': 'Midbrain', 'SCm': 'Midbrain', 'SCs': 'Midbrain', 'APN': 'Midbrain', 'PAG': 'Midbrain',
    'SNr': 'Midbrain', 'SNc': 'Midbrain', 'VTA': 'Midbrain', 'IC': 'Midbrain', 'NOT': 'Midbrain', 'MB': 'Midbrain',
}
REGION_COLOR = {'Frontal cortex': '#1f6fb2', 'Other neocortex': '#c9262c', 'Olfactory cortex': '#8c564b',
                'Hippocampal formation': '#2ca02c', 'Striatum / Septum': '#9467bd', 'Thalamus': '#e377c2',
                'Midbrain': '#b5a800', 'Unassigned': '#7f7f7f'}


def area_region(area: str) -> str:
    if area_group(area) == 'frontal':
        return 'Frontal cortex'
    return REGION_OF.get(area, 'Unassigned')


# Frontal cortex split into subgroups, each a shade of blue (statistics still pool all frontal areas)
FRONTAL_SUBGROUP = {'ACAd': 'Frontal: mPFC', 'ACAv': 'Frontal: mPFC', 'PL': 'Frontal: mPFC', 'ILA': 'Frontal: mPFC',
                    'MOs': 'Frontal: MOs', 'FRP': 'Frontal: MOs',
                    'ORBl': 'Frontal: ORB', 'ORBm': 'Frontal: ORB', 'ORBvl': 'Frontal: ORB',
                    'AId': 'Frontal: AI', 'AIv': 'Frontal: AI', 'AIp': 'Frontal: AI'}
SUBGROUP_ORDER = ['Frontal: mPFC', 'Frontal: MOs', 'Frontal: ORB', 'Frontal: AI']
SUBGROUP_COLOR = {'Frontal: mPFC': '#0d2d6b', 'Frontal: MOs': '#2b6cb8', 'Frontal: ORB': '#2fa4e7', 'Frontal: AI': '#9bc4ea'}


def area_color(area: str) -> str:
    """Marker colour of an area: a blue shade per frontal subgroup, otherwise the region colour."""
    if area_group(area) == 'frontal':
        return SUBGROUP_COLOR[FRONTAL_SUBGROUP.get(area, 'Frontal: MOs')]
    return REGION_COLOR[area_region(area)]


POINT_LIGHTEN, POINT_ALPHA = 0.1, 0.5   # data-point fill: colour blended 10 % towards white, 50 % opaque
POINT_EDGE = 0.5   # marker edge width (pt) of every data point, filled or open


def soft(color, alpha=POINT_ALPHA):
    """Fill colour of data points: the colour slightly lighter and semi-transparent, so dense clouds of points read
    less heavily. Marker edges are the full, opaque colour (filled and open markers alike); tick labels, legends, lines
    and bars keep the full colour too."""
    rgb = np.asarray(to_rgb(color))
    return (*(rgb + (1 - rgb) * POINT_LIGHTEN), alpha)


def point_color(area):
    """soft() of an area's colour."""
    return soft(area_color(area))


def region_legend(ax, areas, loc='upper left', **kw):
    """Square colour swatches: one per frontal subgroup present, then one per non-frontal region present."""
    areas = set(areas)
    sub_present = {FRONTAL_SUBGROUP.get(a, 'Frontal: MOs') for a in areas if area_group(a) == 'frontal'}
    reg_present = {area_region(a) for a in areas if area_group(a) != 'frontal'}
    handles = [Line2D([], [], color=SUBGROUP_COLOR[g], marker='s', ls='', ms=5, label=g) for g in SUBGROUP_ORDER if g in sub_present]
    handles += [Line2D([], [], color=REGION_COLOR[r], marker='s', ls='', ms=5, label=r)
                for r in REGION_ORDER + ['Unassigned'] if r in reg_present]
    leg = ax.legend(handles=handles, loc=loc, frameon=False, title='Region', **kw)
    ax.add_artist(leg)
    return leg


# ----------------------------------------------------------------------------- helpers
LETTER_PT = 12


def capitalize_labels(fig):
    """First letter in upper case for every line of the panel titles, x / y labels (colorbar labels included), legend
    titles and entries, and in-plot notes. Tick labels are capitalized at the source (PAIR_LABEL), since matplotlib re-creates them on every draw."""
    import re

    def cap(s):  # every line; a lone statistic symbol (q < ..., p = ...) and 'n.s.' stay lower case
        return '\n'.join(line if re.match(r'[^\W\d_](\s|$)|n\.s\.$', line) else line[:1].upper() + line[1:]
                         for line in s.split('\n'))
    for ax in fig.axes:
        for text in (ax.xaxis.label, ax.yaxis.label):  # one label: only its first line (the rest continue it)
            first, _, rest = text.get_text().partition('\n')
            text.set_text(cap(first) + ('\n' + rest if rest else ''))
        texts = [ax._left_title, ax.title, ax._right_title] + list(ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            texts += [legend.get_title()] + list(legend.get_texts())
        for text in texts:
            text.set_text(cap(text.get_text()))


_CAP_PT = {}


def _cap_height_pt(size, weight):
    """Ink height (pt) of a capital 'H' above the baseline, measured by rendering it once off-screen; the renderer's
    text metrics include padding that differs between font sizes, so they do not give the visible cap height."""
    key = (size, weight)
    if key not in _CAP_PT:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        dpi = 600
        f = Figure(figsize=(1, 1), dpi=dpi)
        FigureCanvasAgg(f)
        f.text(0.1, 0.5, 'H', fontsize=size, fontweight=weight, va='baseline')
        f.canvas.draw()
        img = np.asarray(f.canvas.buffer_rgba())[..., 0]
        rows = np.where((img < 128).any(1))[0]
        _CAP_PT[key] = (rows.max() + 1 - rows.min()) * 72 / dpi
    return _CAP_PT[key]


def place_letters(fig, axes, letters, dx=None):
    """Bold capital panel letters, aligned: y-axis labels of panels in the same grid column are aligned first, then each
    letter is left-aligned with the outer edge of its panel's y-axis label, and each letter's cap top is level with the
    cap top of the first line of its panel title (shared across a row of letters). Also capitalizes the axis labels.
    dx is kept for compatibility and ignored."""
    capitalize_labels(fig)
    fig.canvas.draw()
    fig.align_ylabels([ax for ax in axes if ax.get_subplotspec() is not None])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    from matplotlib.font_manager import FontProperties
    letter_fp = FontProperties(size=LETTER_PT, weight='bold')

    def extent(s, fp):
        return r.get_text_width_height_descent(s, fp, ismath=False)[1:]

    def cap_height(fp):  # visible height of a capital 'H' above the baseline, in display units
        return _cap_height_pt(fp.get_size_in_points(), fp.get_weight()) * fig.dpi / 72

    xs, cap_tops, titles = [], [], []
    for ax in axes:
        yl = ax.yaxis.label
        xs.append((yl.get_window_extent(r) if yl.get_text() else ax.get_tightbbox(r)).x0)
        title = next((t for t in (ax._left_title, ax.title, ax._right_title) if t.get_text()), None)
        titles.append(title)
        if title is not None:
            fp = title.get_fontproperties()
            (h1, d1), (hl, dl) = extent(title.get_text().split('\n')[0] or 'H', fp), extent('lp', fp)
            cap_tops.append(title.get_window_extent(r).y1 - (max(h1, hl) - max(d1, dl)) + cap_height(fp))
        else:
            cap_tops.append(ax.get_window_extent(r).y1 + 4 * fig.dpi / 72 + cap_height(letter_fp))
    xs, cap_tops = np.array(xs), np.array(cap_tops)

    def snap(vals, tol, agg):
        out, groups = vals.copy(), []
        for i in np.argsort(vals):
            if groups and vals[i] - vals[groups[-1][-1]] <= tol:
                groups[-1].append(i)
            else:
                groups.append([i])
        for g in groups:
            out[g] = agg(vals[g])
        return out

    row_tops = snap(cap_tops, 0.3 * fig.dpi, np.max)  # one top per row: the highest first-line cap top
    for ax, title, top, own in zip(axes, titles, row_tops, cap_tops):
        if title is not None and top - own > 0.5:  # titles only move up, level with the row's highest
            ax._autotitlepos = False
            for t in (ax._left_title, ax.title, ax._right_title):
                tr = t.get_transform()
                X, Y = tr.transform(t.get_position())
                t.set_position(tuple(tr.inverted().transform((X, Y + top - own))))
    inv = fig.transFigure.inverted()
    for L, x, top in zip(letters, xs, row_tops):
        fx, fy = inv.transform((x, top - cap_height(letter_fp)))
        fig.text(fx, fy, L, fontproperties=letter_fp, ha='left', va='baseline')


def plain_log_ticks(axis):
    """Log-axis tick labels as plain decimals (0.01, 0.1) instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(NullFormatter())


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format='svg')
    if EXTRA_PNG_DIR:
        Path(EXTRA_PNG_DIR).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(EXTRA_PNG_DIR) / (path.stem + '.png'), dpi=160)
    plt.close(fig)


def matrix(table, areas, col):
    m = pd.DataFrame(np.nan, index=areas, columns=areas, dtype=float)
    for _, r in table.iterrows():
        m.loc[r['source'], r['target']] = r[col]
    return m


def heatmap(ax, m, cmap, vmin, vmax, fmt='{:.2f}', cbar_label='', marks=None, cbar=True, text_color_split=None):
    areas = list(m.index)
    vals = m.values.astype(float)
    masked = np.ma.masked_invalid(vals)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_anchor('N')
    ax.set_xticks(range(len(areas)))
    ax.set_yticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=45, ha='right')
    ax.set_yticklabels(areas)
    for lab, a in zip(ax.get_xticklabels(), areas):
        lab.set_color(GROUP_COLOR[area_group(a)])
    for lab, a in zip(ax.get_yticklabels(), areas):
        lab.set_color(GROUP_COLOR[area_group(a)])
    ax.set_xlabel('target area')
    ax.set_ylabel('source area')
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    split = (vmin + vmax) / 2 if text_color_split is None else text_color_split
    for i in range(len(areas)):
        for j in range(len(areas)):
            if i == j or not np.isfinite(vals[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='0.92', lw=0))
                continue
            txt = fmt.format(vals[i, j])
            if marks is not None and marks[i, j]:
                txt += marks[i, j]
            color = 'white' if (vals[i, j] > split) == (cmap in ('viridis', 'Blues', 'Blues')) else 'black'
            if cmap == 'RdBu_r':
                color = 'white' if abs(vals[i, j] - split) > 0.6 * (vmax - split) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=6.8 if len(areas) <= 6 else (5.6 if len(areas) <= 8 else 4.9),
                    color=color)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label)
        cb.outline.set_visible(False)
    return im


def group_legend(ax, loc='upper right', **kw):
    handles = [Line2D([], [], color=GROUP_COLOR[g], marker='s', ls='', ms=6,
                      label={'frontal': 'frontal cortex', 'other': 'others (non-frontal)'}[g]) for g in GROUP_COLOR]
    ax.legend(handles=handles, loc=loc, frameon=False, **kw)


def pair_legend(ax, loc='upper left', **kw):
    handles = [Line2D([], [], color=PAIR_COLOR[t], marker='o', ls='', ms=5, label=PAIR_LABEL[t]) for t in PAIR_TYPES]
    ax.legend(handles=handles, loc=loc, frameon=False, title='Source → Target', **kw)


def strip_by_type(ax, table, ycol, ylabel, zero_line=None, y_err=None, show_mean=True):
    rng = np.random.default_rng(0)
    for i, t in enumerate(PAIR_TYPES):
        sub = table[table['pair_type'] == t]
        if sub.empty:
            continue
        x = i + rng.uniform(-0.18, 0.18, len(sub))
        ax.scatter(x, sub[ycol], s=16, color=soft(PAIR_COLOR[t]), edgecolor=PAIR_COLOR[t], linewidth=POINT_EDGE, zorder=3)
        if y_err is not None:
            ax.errorbar(x, sub[ycol], yerr=sub[y_err], fmt='none', ecolor=soft(PAIR_COLOR[t]), elinewidth=0.6, zorder=2)
        if show_mean and len(sub) > 1:
            m, se = sub[ycol].mean(), sub[ycol].std(ddof=1) / np.sqrt(len(sub))
            ax.errorbar(i + 0.32, m, yerr=se, fmt='_', color='k', ms=9, mew=1.4, elinewidth=1.0, capsize=0, zorder=4)
    ax.set_xticks(range(len(PAIR_TYPES)))
    ax.set_xticklabels([PAIR_LABEL[t].replace(' → ', '\n→ ') for t in PAIR_TYPES])
    ax.set_xlim(-0.6, len(PAIR_TYPES) - 0.3)
    ax.set_ylabel(ylabel)
    if zero_line is not None:
        ax.axhline(zero_line, color='k', ls='--', lw=0.7)


# ----------------------------------------------------------------------------- figure 1
def task_axis_panels(axA, axB, at, areas, xlabels, colors, sessions_of=None, sessions=None, null_band=True):
    """A: task-axis in-sample R² per area; B: cross-validated (held-out blocks) decoding accuracy per area."""
    x = np.arange(len(areas))
    for ax, col, err, ylab in [(axA, 'task_r2', 'task_r2_sd', 'task-axis R²\n(variance of the projection explained by context)'),
                               (axB, 'acc_cv', 'acc_cv_sd', 'context decoding accuracy\n(held-out blocks)')]:
        for i, a in enumerate(areas):
            rows = at[at['area'] == a] if 'area' in at else at.loc[[a]]
            for _, r in rows.iterrows():
                pred = bool(r['acc_predictive'])
                mk = MARKERS[sessions.index(r['session']) % len(MARKERS)] if sessions is not None else 'o'
                if ax is axB and null_band:      # this area instance's block-permutation null range
                    ax.plot([i, i], [r['acc_block_null_lo'], r['acc_block_null_hi']], color='0.8', lw=5, solid_capstyle='butt',
                            alpha=0.6, zorder=1)
                ax.errorbar(i, r[col], yerr=r[err], fmt=mk, ms=6, color=soft(colors[i]), mfc=soft(colors[i]) if pred else 'white',
                            mec=colors[i], mew=POINT_EDGE, elinewidth=0.8, capsize=0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45 if len(areas) > 6 else 0, ha='right' if len(areas) > 6 else 'center')
        for lab, c in zip(ax.get_xticklabels(), colors):
            lab.set_color(c)
        ax.set_xlim(-0.6, len(areas) - 0.4)
        ax.set_ylabel(ylab)
        ax.set_xlabel('area')
    axA.set_ylim(0, 1)
    axB.axhline(0.5, color='k', ls='--', lw=0.7)
    axB.set_ylim(min(0.2, at['acc_cv'].min() - at['acc_cv_sd'].max() - 0.03, at['acc_block_null_lo'].min() - 0.03), 1.0)
    n_pred = int(at['acc_predictive'].sum())
    axA.set_title('Task axis: in-sample R²\nfilled = predictive in B, open = not', loc='left')
    axB.set_title(f'Task axis: predictive accuracy\n{n_pred}/{len(at)} above the block-permutation null (grey: null range)',
                  loc='left')


def comm_significance(res, results_dir):
    """(q-values aligned to res['pairs'], label) for the rank-d R² test if available, else the full-model test."""
    q = r2_dim_significance(res, results_dir)
    if q is not None:
        return q, 'trial-shuffle null at rank d'
    return res['pair_table']['r2_q'].values, 'trial-shuffle null, full model'


def figure1(res, out_dir: Path):
    sid, areas = res['session_id'], res['areas']
    pt, at = res['pair_table'], res['area_table']
    n = res['min_units']
    at = at.assign(session=sid) if 'session' not in at else at

    fig = plt.figure(figsize=(9, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.45, left=0.09, right=0.95, top=0.9, bottom=0.1)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    colors = [GROUP_COLOR[area_group(a)] for a in areas]
    task_axis_panels(axA, axB, at, areas, areas, colors)

    # C, D: communication subspace R² at the 1-SEM dimensionality, in-sample and cross-validated, shared scale
    m_tr = matrix(pt, areas, 'r2_train_dim')
    m_cv = matrix(pt, areas, 'r2_cv_dim')
    vmax = max(0.05, np.nanmax(m_tr.values))
    heatmap(axC, m_tr, 'viridis', 0, vmax, fmt='{:.2f}', cbar_label='in-sample R²')
    axC.set_title('Communication subspace: in-sample R²\nrank-d reduced-rank fit, pooled over target units', loc='left')
    q, label = comm_significance(res, out_dir.parent.parent)
    sig = matrix(pt.assign(q=q), areas, 'q').values <= FDR_ALPHA
    marks = np.where(sig, '', '†')
    heatmap(axD, m_cv, 'viridis', 0, vmax, fmt='{:.2f}', cbar_label='cross-validated R²', marks=marks)
    n_sig = int(sig[~np.eye(len(areas), dtype=bool)].sum())
    short = 'rank-d' if 'rank d' in label else 'full-model'
    axD.set_title(f'Communication subspace: predictive R²\n{n_sig}/{len(pt)} pairs > {short} shuffle null (q < {FDR_ALPHA})'
                  + ('; † = n.s.' if n_sig < len(pt) else ''), loc='left')

    fig.suptitle(f'Figure 1. Task axis and communication subspace, session {sid} '
                 f'({res["n_trials"]} trials, {res["n_blocks"]} blocks, {n} units per area)', fontsize=9, x=0.09,
                 ha='left', y=0.985)
    place_letters(fig, [axA, axB, axC, axD], 'ABCD')
    save(fig, out_dir / f'figure1_{sid}.svg')


def supp_figure1(res, out_dir: Path):
    """Supplementary Figure 1: how the communication-subspace dimensionality is chosen."""
    sid, areas = res['session_id'], res['areas']
    pt = res['pair_table']
    n = res['min_units']
    fig = plt.figure(figsize=(9, 3.8))
    gs = fig.add_gridspec(1, 2, wspace=0.45, left=0.09, right=0.95, top=0.82, bottom=0.18)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    curves = res['r2_curve'].mean(axis=0)
    ranks = np.arange(1, n + 1)
    for p, (s_, t_) in enumerate(res['pairs']):
        typ = pt.iloc[p]['pair_type']
        axA.plot(ranks, curves[p], color=PAIR_COLOR[typ], lw=0.9, alpha=0.9)
        d = pt.iloc[p]['dim']
        axA.plot(d, np.interp(d, ranks, curves[p]), marker='o', ms=3.2, color=soft(PAIR_COLOR[typ]), mec=PAIR_COLOR[typ], mew=POINT_EDGE, ls='')
    axA.axhline(0, color='k', lw=0.6, ls=':')
    axA.set_xlim(0.5, n + 0.5)
    axA.set_xlabel('number of predictive dimensions (rank)')
    axA.set_ylabel('cross-validated R² (pooled over target units)')
    axA.set_title('Cross-validated R² vs. rank, every ordered pair\ndot = 1-SEM dimensionality; colours = pair type',
                  loc='left')
    m_d = matrix(pt, areas, 'dim')
    heatmap(axB, m_d, 'Blues', 0, max(np.nanmax(m_d.values), 1), fmt='{:.1f}', cbar_label='dimensionality (1-SEM rule)')
    axB.set_title(f'Dimensionality of the communication subspace\nof {n} source units, mean over subsamples', loc='left')
    fig.suptitle(f'Supplementary Figure 1. Communication-subspace dimensionality, session {sid}', fontsize=9, x=0.09,
                 ha='left', y=0.975)
    place_letters(fig, [axA, axB], 'AB')
    save(fig, out_dir / f'supp_figure1_{sid}.svg')


# ----------------------------------------------------------------------------- figure 2
def alignment_marks(q, z):
    marks = np.full(q.shape, '', dtype=object)
    marks[(q <= FDR_ALPHA) & (z > 0)] = '*'
    marks[(q <= FDR_ALPHA) & (z < 0)] = '−'
    return marks


def qualify(res):
    """Pair table with qualification flags. A pair's source-side alignment is 'qualified'
    when the source task axis decodes context above the block-permutation null AND the
    communication-subspace R² beats the trial-shuffle null (FDR); target-side alignment
    needs the target's axis to be predictive instead. 'excess' is the observed cosine
    minus the mean of the shuffled-label axis null (the significance null); FDR q-values
    for that null are recomputed over the qualified pairs only."""
    pt = res['pair_table'].copy()
    at = res['area_table'].set_index('area')
    pred = {a: bool(at.loc[a, 'acc_predictive']) for a in res['areas']}
    pt['source_acc_cv'] = at.loc[pt['source'], 'acc_cv'].values
    pt['source_acc_block_null_hi'] = at.loc[pt['source'], 'acc_block_null_hi'].values
    pt['source_axis_predictive'] = pt['source'].map(pred)
    pt['target_axis_predictive'] = pt['target'].map(pred)
    pt['r2_significant'] = pt['r2_q'] <= FDR_ALPHA
    pt['qualified'] = pt['source_axis_predictive'] & pt['r2_significant']
    pt['qualified_tgt'] = pt['target_axis_predictive'] & pt['r2_significant']
    pt['excess'] = pt['cos'] - pt['cos_shuf_mean']
    pt['excess_tgt'] = pt['cos_tgt'] - pt['cos_tgt_shuf_mean']
    for pcol, qcol, mask in [('cos_shuf_p', 'cos_shuf_q', 'qualified'),
                             ('cos_tgt_shuf_p', 'cos_tgt_shuf_q', 'qualified_tgt')]:
        pt[qcol] = np.nan
        m = pt[mask].values
        if m.any():
            pt.loc[m, qcol] = fdr_bh(pt.loc[m, pcol].values)
    return pt


def figure2(res, out_dir: Path):
    sid, areas = res['session_id'], res['areas']
    pt_all = qualify(res)
    pt = pt_all[pt_all['qualified']]
    pt_t = pt_all[pt_all['qualified_tgt']]
    n = res['min_units']
    n_excl = int((~pt_all['qualified']).sum())

    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.55, wspace=0.55,
                          left=0.07, right=0.955, top=0.91, bottom=0.1)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0:2])
    axF = fig.add_subplot(gs[1, 2])

    at = res['area_table'].set_index('area')
    not_pred = [a for a in areas if not at.loc[a, 'acc_predictive']]

    def flag_sources(ax):
        ax.set_yticklabels([f'{a} \u2020' if a in not_pred else a for a in areas])
        for lab, a in zip(ax.get_yticklabels(), areas):
            lab.set_color(GROUP_COLOR[area_group(a)])

    # A: cosine heatmap, shuffled-label axis null marks
    m_cos = matrix(pt, areas, 'cos')
    marks = alignment_marks(matrix(pt, areas, 'cos_shuf_q').values, matrix(pt, areas, 'cos_shuf_z').values)
    heatmap(axA, m_cos, 'viridis', 0, 1, cbar_label='cosine similarity', marks=marks)
    n_up = int(np.sum(marks == '*'))
    n_dn = int(np.sum(marks == '−'))
    axA.set_title(f'Source task axis in comm. subspace\nq < {FDR_ALPHA}: */− {n_up}/{n_dn} of {len(pt)} qualified',
                  loc='left')
    flag_sources(axA)
    if not_pred:
        axA.set_ylabel('source area (\u2020 task axis not predictive)')

    # B: observed vs the shuffled-label axis null
    for t in PAIR_TYPES:
        sub = pt[pt['pair_type'] == t]
        sig = sub['cos_shuf_q'] <= FDR_ALPHA
        for flag, mfc in [(True, soft(PAIR_COLOR[t])), (False, 'white')]:
            s2 = sub[sig == flag]
            if s2.empty:
                continue
            axB.errorbar(s2['cos_shuf_mean'], s2['cos'], yerr=s2['cos_sd'],
                         xerr=np.vstack([s2['cos_shuf_mean'] - s2['cos_shuf_lo'], s2['cos_shuf_hi'] - s2['cos_shuf_mean']]),
                         fmt='o', ms=4.5, color=soft(PAIR_COLOR[t]), mfc=mfc, mec=PAIR_COLOR[t], mew=POINT_EDGE, elinewidth=0.6, capsize=0)
    axB.plot([0, 1], [0, 1], 'k--', lw=0.7)
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1)
    axB.set_aspect('equal')
    axB.set_xlabel('null: LDA axis on shuffled labels, same subspace\n(mean, 95% range)')
    axB.set_ylabel('observed cosine (mean ± SD over subsamples)')
    axB.set_title('Every pair vs. its null\nnull = shuffled-label axis, same subspace', loc='left')
    pair_legend(axB, loc='lower right')
    axB.text(0.03, 0.97, 'open = not significant', transform=axB.transAxes, va='top', fontsize=7)

    # C: z-score vs the shuffled-label axis null by pair type
    strip_by_type(axC, pt, 'cos_shuf_z', 'z-score vs. shuffled-label axis null', zero_line=0)
    axC.set_title('By pair type\nz-score of the subsample-averaged cosine', loc='left')
    axC.text(0.02, 0.1, 'black = mean ± SEM over pairs', transform=axC.transAxes, va='bottom', fontsize=7)

    # D: angle excess by source area and by target area
    pt2 = pt_all
    xs = np.arange(len(areas))
    for i, a in enumerate(areas):
        as_src = pt2[pt2['source'] == a]['excess'].values
        as_tgt = pt2[pt2['target'] == a]['excess'].values
        col = GROUP_COLOR[area_group(a)]
        q_src = pt2[pt2['source'] == a]['qualified'].values
        q_tgt = pt2[pt2['target'] == a]['qualified_tgt'].values
        for vals, qmask, dx, marker, alpha in [(as_src, q_src, -0.17, 'o', 1.0), (as_tgt, q_tgt, 0.17, 's', 0.65)]:
            axD.scatter(np.full(qmask.sum(), i + dx), vals[qmask], s=14, color=soft(col, POINT_ALPHA * alpha), marker=marker,
                        edgecolor=col, linewidth=POINT_EDGE, zorder=3)
            axD.scatter(np.full((~qmask).sum(), i + dx), vals[~qmask], s=14, color='white', marker=marker, edgecolor=col,
                        linewidth=POINT_EDGE, zorder=3)
            if qmask.any():
                axD.plot([i + dx - 0.13, i + dx + 0.13], [vals[qmask].mean()] * 2, color='k', lw=1.4, zorder=4)
    axD.axhline(0, color='k', ls='--', lw=0.7)
    axD.set_xticks(xs)
    axD.set_xticklabels(areas, rotation=45 if len(areas) > 6 else 0, ha='right' if len(areas) > 6 else 'center')
    for lab, a in zip(axD.get_xticklabels(), areas):
        lab.set_color(GROUP_COLOR[area_group(a)])
    axD.set_ylabel('observed − null cosine\n(null = shuffled-label axis)')
    axD.set_xlabel('area')
    axD.set_title('By area\nas source (●, its task axis) and as target (■)', loc='left')
    h = [Line2D([], [], marker='o', color='0.4', ls='', ms=5, label='as source (open: excluded)'),
         Line2D([], [], marker='s', color='0.4', ls='', ms=5, alpha=0.65, label='as target (open: excluded)'),
         Line2D([], [], color='k', lw=1.4, label='mean of qualified')]
    ymax = pt2['excess'].max()
    axD.set_ylim(-0.3 * ymax, ymax * 1.08)
    axD.legend(handles=h, loc='lower left', frameon=False, handletextpad=0.4, ncol=3, columnspacing=1.2,
               bbox_to_anchor=(-0.02, -0.01))

    # F: target-side alignment
    m_ct = matrix(pt_t, areas, 'cos_tgt')
    marks = alignment_marks(matrix(pt_t, areas, 'cos_tgt_shuf_q').values, matrix(pt_t, areas, 'cos_tgt_shuf_z').values)
    heatmap(axF, m_ct, 'viridis', 0, 1, cbar_label='cosine similarity', marks=marks)
    n_up = int(np.sum(marks == '*'))
    n_dn = int(np.sum(marks == '−'))
    axF.set_title(f'Target side: task axis in predicted dims\nq < {FDR_ALPHA}: */− {n_up}/{n_dn} of {len(pt_t)} qualified',
                  loc='left')
    for lab, a in zip(axF.get_yticklabels(), areas):
        lab.set_color(GROUP_COLOR[area_group(a)])
    axF.set_yticklabels([f'{a} \u2020' if a in not_pred else a for a in areas])
    for lab, a in zip(axF.get_xticklabels(), areas):
        lab.set_color(GROUP_COLOR[area_group(a)])
    axF.set_xticklabels([f'{a} \u2020' if a in not_pred else a for a in areas], rotation=45, ha='right')

    fig.suptitle(f'Figure 2. Task axis vs. communication subspace, session {sid}: {len(pt)} of {len(pt_all)} ordered '
                 f'pairs qualify (source task axis predictive, R² significant; grey = excluded)', fontsize=9, x=0.07,
                 ha='left', y=0.985)
    place_letters(fig, [axA, axB, axC, axD, axF], 'ABCDE')
    save(fig, out_dir / f'figure2_{sid}.svg')
    pt_all.to_csv(out_dir.parent / 'alignment_by_pair_qualified.csv', index=False)


# ----------------------------------------------------------------------------- pooled figures
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', '*', 'h', 'p']


def session_legend(ax, sessions, loc='lower right'):
    h = [Line2D([], [], marker=MARKERS[k % len(MARKERS)], color='0.5', ls='', ms=5, label=s) for k, s in enumerate(sessions)]
    ax.legend(handles=h, loc=loc, frameon=False, title='session')


def strip_by_type_sessions(ax, tables, ycol, ylabel, zero_line=None, log=False, text_loc='top', show_n=True, color_by=None,
                           color_fn=point_color):
    """Like strip_by_type, but one marker shape per session."""
    rng = np.random.default_rng(0)
    pooled = pd.concat(tables, ignore_index=True)
    for i, t in enumerate(PAIR_TYPES):
        for k, tab in enumerate(tables):
            sub = tab[tab['pair_type'] == t]
            if sub.empty:
                continue
            x = i + rng.uniform(-0.2, 0.2, len(sub))
            ecol = [area_color(a) for a in sub[color_by]] if color_by else PAIR_COLOR[t]   # edges: full colour
            col = [color_fn(a) for a in sub[color_by]] if color_by else soft(PAIR_COLOR[t])   # e.g. color_by='target'
            ax.scatter(x, sub[ycol], s=16, marker=MARKERS[k % len(MARKERS)], c=col, edgecolor=ecol,
                       linewidth=POINT_EDGE, zorder=3)
        sub = pooled[pooled['pair_type'] == t]
        if len(sub) > 1:
            m, se = sub[ycol].mean(), sub[ycol].std(ddof=1) / np.sqrt(len(sub))
            ax.errorbar(i + 0.34, m, yerr=se, fmt='_', color='k', ms=9, mew=1.4, elinewidth=1.0, capsize=0, zorder=4)
    ax.set_xticks(range(len(PAIR_TYPES)))
    ax.set_xticklabels([PAIR_LABEL[t].replace(' → ', '\n→ ') for t in PAIR_TYPES])
    ax.set_xlim(-0.6, len(PAIR_TYPES) - 0.3)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')
        plain_log_ticks(ax.yaxis)
    if zero_line is not None:
        ax.axhline(zero_line, color='k', ls='--', lw=0.7)
    counts = pooled['pair_type'].value_counts()
    if show_n:
        y = 0.98 if text_loc == 'top' else (0.02 if text_loc == 'bottom' else text_loc)
        ax.text(0.02, y, '\n'.join(f'{PAIR_LABEL[t]}: n = {counts.get(t, 0)}' for t in PAIR_TYPES),
                transform=ax.transAxes, va='top' if text_loc == 'top' else 'bottom', fontsize=6.5)


def area_order(areas):
    """Frontal areas first, then the other regions in REGION_ORDER, alphabetical within a region."""
    rank = {r: i for i, r in enumerate(REGION_ORDER + ['Unassigned'])}
    sub_rank = {g: i for i, g in enumerate(SUBGROUP_ORDER)}
    return sorted(areas, key=lambda a: (rank[area_region(a)], sub_rank.get(FRONTAL_SUBGROUP.get(a, 'Frontal: MOs'), 0)
                                        if area_group(a) == 'frontal' else 0, a))


def r2_at_dimensionality(res):
    """Per pair: cross-validated R² of the reduced-rank model at the pair's own 1-SEM dimensionality, taken per
    subsample (rank = that subsample's dimensionality) and averaged over subsamples."""
    curves, dims = res['r2_curve'], res['dims']                 # (K, pairs, ranks), (K, pairs); ranks are 1-based
    K, P = dims.shape
    return np.array([curves[np.arange(K), p, dims[:, p] - 1].mean() for p in range(P)])


def r2_dim_significance(res, results_dir: Path):
    """FDR q-values of the rank-d R² against its own trial-shuffle null (r2_dim_null.py), aligned to res['pairs'];
    None if that file has not been computed for this session."""
    f = results_dir / res['session_id'] / 'r2_dim_significance.csv'
    if not f.exists():
        return None
    sig = pd.read_csv(f).set_index(['source', 'target'])
    return sig.loc[res['pairs'], 'r2_dim_q'].values


def figure1_pooled(results, out_dir: Path):
    sessions = [r['session_id'] for r in results]
    tables = [qualify(r) for r in results]
    q_dim = [r2_dim_significance(r, out_dir.parent) for r in results]
    rank_d_test = all(q is not None for q in q_dim)
    if rank_d_test:
        tables = [t.assign(r2_sig_dim=(q <= FDR_ALPHA)) for t, q in zip(tables, q_dim)]
    else:
        tables = [t.assign(r2_sig_dim=t['r2_significant']) for t in tables]
    pooled = pd.concat(tables, ignore_index=True)
    areas_tab = pd.concat([r['area_table'] for r in results], ignore_index=True)
    n = results[0]['min_units']

    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.3, left=0.07, right=0.98, top=0.9, bottom=0.1)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    order = area_order(areas_tab['area'].unique())
    colors = [GROUP_COLOR[area_group(a)] for a in order]
    task_axis_panels(axA, axB, areas_tab, order, order, colors, sessions=sessions)
    axA.set_xlabel('area (frontal left, others right); one marker per session')
    axB.set_xlabel('area (frontal left, others right); one marker per session')
    session_legend(axA, sessions, loc='upper right')

    # C, D: communication subspace R² at the 1-SEM dimensionality, by pair type, shared log scale
    lo = min(pooled['r2_train_dim'].min(), pooled['r2_cv_dim'].min())
    use_log = lo > 0
    strip_by_type_sessions(axC, tables, 'r2_train_dim', 'in-sample R² (rank-d fit)', log=use_log, show_n=True)
    strip_by_type_sessions(axD, tables, 'r2_cv_dim', 'cross-validated R² (rank-d fit)', log=use_log, show_n=False)
    hi = max(pooled['r2_train_dim'].max(), pooled['r2_cv_dim'].max()) * 1.3
    for ax in (axC, axD):
        ax.set_ylim(lo * 0.7 if use_log else min(0, lo) - 0.01, hi)
        if use_log:
            plain_log_ticks(ax.yaxis)
    axC.set_title('Communication subspace: in-sample R²\nreduced-rank fit at the 1-SEM dimensionality', loc='left')
    n_sig = int(pooled['r2_sig_dim'].sum())
    label = 'rank d' if rank_d_test else 'full model'
    axD.set_title(f'Communication subspace: predictive R²\n{n_sig}/{len(pooled)} pairs > trial-shuffle null '
                  f'({label}, q < {FDR_ALPHA})', loc='left')

    fig.suptitle(f'Figure 1 (pooled). Task axis and communication subspace over {len(results)} sessions: '
                 f'{len(areas_tab)} area instances ({areas_tab["area"].nunique()} distinct areas), {len(pooled)} ordered pairs, '
                 f'{n} units per area', fontsize=9, x=0.07, ha='left', y=0.98)
    place_letters(fig, [axA, axB, axC, axD], 'ABCD')
    save(fig, out_dir / 'figure1_pooled.svg')


def supp_figure1_pooled(results, out_dir: Path):
    sessions = [r['session_id'] for r in results]
    tables = [qualify(r) for r in results]
    pooled = pd.concat(tables, ignore_index=True)
    n = results[0]['min_units']
    fig = plt.figure(figsize=(9, 3.8))
    gs = fig.add_gridspec(1, 2, wspace=0.4, left=0.09, right=0.97, top=0.82, bottom=0.2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    ranks = np.arange(1, n + 1)
    for res, tab in zip(results, tables):
        curves = res['r2_curve'].mean(axis=0)
        for p in range(len(res['pairs'])):
            typ = tab.iloc[p]['pair_type']
            axA.plot(ranks, curves[p], color=PAIR_COLOR[typ], lw=0.5, alpha=0.8)
            d = tab.iloc[p]['dim']
            axA.plot(d, np.interp(d, ranks, curves[p]), marker='o', ms=2.4, color=soft(PAIR_COLOR[typ]), mec=PAIR_COLOR[typ], mew=POINT_EDGE, ls='')
    axA.axhline(0, color='k', lw=0.6, ls=':')
    axA.set_xlim(0.5, n + 0.5)
    axA.set_xlabel('number of predictive dimensions (rank)')
    axA.set_ylabel('cross-validated R² (pooled over target units)')
    axA.set_title(f'Cross-validated R² vs. rank, {len(pooled)} ordered pairs\ndot = 1-SEM dimensionality; colours = pair '
                  f'type (as in B)', loc='left')
    strip_by_type_sessions(axB, tables, 'dim', f'dimensionality (of {n} source units)')
    axB.set_ylim(0, max(12, pooled['dim'].max() + 3))
    axB.set_title(f'Dimensionality by pair type\nmedian {pooled["dim"].median():.1f}, range {pooled["dim"].min():.1f}'
                  f'–{pooled["dim"].max():.1f}', loc='left')
    fig.suptitle(f'Supplementary Figure 1 (pooled). Communication-subspace dimensionality over {len(results)} sessions',
                 fontsize=9, x=0.09, ha='left', y=0.975)
    place_letters(fig, [axA, axB], 'AB')
    save(fig, out_dir / 'supp_figure1_pooled.svg')


def figure2_pooled(results, out_dir: Path):
    sessions = [r['session_id'] for r in results]
    tables_all = [qualify(res) for res in results]
    pooled_all = pd.concat(tables_all, ignore_index=True)
    pooled_all.to_csv(out_dir / 'alignment_by_pair_pooled.csv', index=False)
    tables = [t[t['qualified']] for t in tables_all]
    pooled = pooled_all[pooled_all['qualified']]
    tables_t = [t[t['qualified_tgt']] for t in tables_all]
    pooled_t = pooled_all[pooled_all['qualified_tgt']]
    from scipy import stats
    ylab = 'cosine(task axis, comm. subspace) − null\n(null = shuffled-label axis)'

    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.5, wspace=0.3, left=0.07, right=0.98, top=0.89, bottom=0.09)
    axC = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    n_up = int(((pooled['cos_shuf_q'] <= FDR_ALPHA) & (pooled['cos_shuf_z'] > 0)).sum())

    # B: the same excess vs source-axis accuracy, qualified pairs only
    for k, t in enumerate(tables):
        for g in GROUP_COLOR:
            sub = t[t['source'].map(area_group) == g]        # boolean Series: safe when t is empty
            axB.scatter(sub['source_acc_cv'], sub['excess'], s=18, marker=MARKERS[k % len(MARKERS)], color=soft(GROUP_COLOR[g]),
                        edgecolor=GROUP_COLOR[g], linewidth=POINT_EDGE)
    axB.axhline(0, color='k', ls='--', lw=0.7)
    rho, pval = stats.spearmanr(pooled['source_acc_cv'], pooled['excess'])
    fit = stats.linregress(pooled['source_acc_cv'], pooled['excess'])
    xfit = np.array([pooled['source_acc_cv'].min(), pooled['source_acc_cv'].max()])
    axB.plot(xfit, fit.intercept + fit.slope * xfit, color='k', lw=1.2, zorder=4)
    axB.set_xlabel('source task axis: decoding accuracy (held-out blocks)')
    axB.set_ylabel(ylab)
    axB.set_ylim(-0.3, max(0.7, pooled['excess'].max() + 0.05))
    axB.set_title(f'Alignment vs. context decoding of the source area\nSpearman ρ = {rho:.2f}, p = {pval:.2f}; line = least squares',
                  loc='left')
    group_legend(axB, loc='lower right', ncol=2)

    # C: the same excess by area, as source and as target
    order = area_order(set(pooled['source']) | set(pooled_t['target']))
    rng = np.random.default_rng(0)
    for i, a in enumerate(order):
        col = GROUP_COLOR[area_group(a)]
        src = pooled_all[(pooled_all['source'] == a) & pooled_all['qualified']]['excess'].values
        tgt = pooled_all[(pooled_all['target'] == a) & pooled_all['qualified_tgt']]['excess_tgt'].values
        for vals, dx, marker, alpha in [(src, -0.18, 'o', 1.0), (tgt, 0.18, 's', 0.65)]:
            if len(vals) == 0:
                continue
            axC.scatter(i + dx + rng.uniform(-0.08, 0.08, len(vals)), vals, s=13, marker=marker, color=soft(col, POINT_ALPHA * alpha),
                        edgecolor=col, linewidth=POINT_EDGE, zorder=3)
            axC.plot([i + dx - 0.14, i + dx + 0.14], [vals.mean()] * 2, color='k', lw=1.4, zorder=4)
    axC.axhline(0, color='k', ls='--', lw=0.7)
    axC.set_xticks(range(len(order)))
    axC.set_xticklabels(order, rotation=45, ha='right')
    for lab, a in zip(axC.get_xticklabels(), order):
        lab.set_color(GROUP_COLOR[area_group(a)])
    axC.set_xlim(-0.6, len(order) - 0.4)
    axC.set_ylabel(ylab)
    axC.set_xlabel('area (frontal left, others right); points = partner areas × sessions, black = mean')
    axC.set_title(f'Alignment by area, as source (●) and as target (■)\n{n_up}/{len(pooled)} qualified pairs above chance',
                  loc='left')
    ymax = max(pooled['excess'].max(), pooled_t['excess_tgt'].max() if 'excess_tgt' in pooled_t else 0)
    axC.set_ylim(-0.12, max(0.7, ymax + 0.05))
    h = [Line2D([], [], marker='o', color='0.4', ls='', ms=5, label='as source'),
         Line2D([], [], marker='s', color='0.4', ls='', ms=5, alpha=0.65, label='as target'),
         Line2D([], [], color='k', lw=1.4, label='mean')]
    axC.legend(handles=h, loc='lower left', frameon=False, ncol=3, columnspacing=1.2, handletextpad=0.4)

    fig.suptitle(f'Figure 2 (pooled). Alignment over {len(results)} sessions: {len(pooled)} of {len(pooled_all)} ordered '
                 f'pairs qualify (source task axis predictive, R² significant); null = shuffled-label axis',
                 fontsize=9, x=0.07, ha='left', y=0.98)
    place_letters(fig, [axC, axB], 'AB')
    save(fig, out_dir / 'figure2_pooled.svg')

    # Supplementary Figure 2: observed vs the shuffled-label axis null, source and target side
    fig = plt.figure(figsize=(9, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.4, left=0.09, right=0.97, top=0.82, bottom=0.15)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for k, t in enumerate(tables):
        for typ in PAIR_TYPES:
            sub = t[t['pair_type'] == typ]
            axA.scatter(sub['cos_shuf_mean'], sub['cos'], s=18, marker=MARKERS[k % len(MARKERS)], color=soft(PAIR_COLOR[typ]),
                        edgecolor=PAIR_COLOR[typ], linewidth=POINT_EDGE)
    for k, t in enumerate(tables_t):
        for typ in PAIR_TYPES:
            sub = t[t['pair_type'] == typ]
            axB.scatter(sub['cos_tgt_shuf_mean'], sub['cos_tgt'], s=18, marker=MARKERS[k % len(MARKERS)], color=soft(PAIR_COLOR[typ]),
                        edgecolor=PAIR_COLOR[typ], linewidth=POINT_EDGE)
    for ax in (axA, axB):
        ax.plot([0, 1], [0, 1], 'k--', lw=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
    axA.set_xlabel('null mean (shuffled-label axis in the same subspace)')
    axA.set_ylabel('observed cosine (source task axis)')
    n_dn = int(((pooled['cos_shuf_q'] <= FDR_ALPHA) & (pooled['cos_shuf_z'] < 0)).sum())
    axA.set_title(f'Source task axis in the comm. subspace\n{n_up} above / {n_dn} below the null of {len(pooled)} qualified pairs',
                  loc='left')
    session_legend(axA, sessions, loc='lower right')
    axB.set_xlabel('null mean (shuffled-label axis in the predicted directions)')
    axB.set_ylabel('observed cosine (target task axis)')
    n_up_t = int(((pooled_t['cos_tgt_shuf_q'] <= FDR_ALPHA) & (pooled_t['cos_tgt_shuf_z'] > 0)).sum())
    n_dn_t = int(((pooled_t['cos_tgt_shuf_q'] <= FDR_ALPHA) & (pooled_t['cos_tgt_shuf_z'] < 0)).sum())
    axB.set_title(f'Target task axis in the predicted directions\n{n_up_t} above / {n_dn_t} below the null of {len(pooled_t)} qualified',
                  loc='left')
    pair_legend(axB, loc='lower right')
    fig.suptitle(f'Supplementary Figure 2 (pooled). Observed cosine vs. the shuffled-label axis null over {len(results)} sessions',
                 fontsize=9, x=0.09, ha='left', y=0.975)
    place_letters(fig, [axA, axB], 'AB')
    save(fig, out_dir / 'supp_figure2_pooled.svg')
    return pooled_all


def pooled_figures(results, out_dir: Path):
    figure1_pooled(results, out_dir)
    supp_figure1_pooled(results, out_dir)
    return figure2_pooled(results, out_dir)


def main(argv):
    code_dir = Path(__file__).resolve().parent
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results_3sessions'))
    sessions = argv or sorted(p.parent.name for p in results_dir.glob('*/alignment_results.pkl'))
    results = []
    for sid in sessions:
        with open(results_dir / sid / 'alignment_results.pkl', 'rb') as f:
            res = pickle.load(f)
        results.append(res)
        out = results_dir / sid / 'figures'
        figure1(res, out)
        supp_figure1(res, out)
        figure2(res, out)
        print(f'{sid}: figures written to {out}')
    if len(results) > 1:
        out = results_dir / 'pooled'
        out.mkdir(parents=True, exist_ok=True)
        pooled_figures(results, out)
        print(f'pooled figure written to {out}')


if __name__ == '__main__':
    main(sys.argv[1:])
