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
from matplotlib.lines import Line2D

from task_axis_comm_subspace import FDR_ALPHA, area_group, fdr_bh, p_two_sided

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
})

GROUP_COLOR = {'frontal': '#1f6fb2', 'other': '#d9772b'}
PAIR_TYPES = ['frontal->frontal', 'frontal->other', 'other->frontal', 'other->other']
PAIR_LABEL = {'frontal->frontal': 'Frontal → Frontal', 'frontal->other': 'Frontal → Other',
              'other->frontal': 'Other → Frontal', 'other->other': 'Other → Other'}
PAIR_COLOR = {'frontal->frontal': '#1f6fb2', 'frontal->other': '#6fb0dd',
              'other->frontal': '#e8a86a', 'other->other': '#d9772b'}
EXTRA_PNG_DIR = os.environ.get('ALIGN_QC_DIR')  # optional PNG previews outside the results tree


# ----------------------------------------------------------------------------- helpers
LETTER_PT = 12


def capitalize_labels(fig):
    """First letter in upper case for every line of the panel titles, x / y labels (colorbar labels included), legend
    titles and entries, and in-plot notes. Tick labels are capitalized at the source (PAIR_LABEL), since matplotlib re-creates them on every draw."""
    import re

    def cap(s):  # every line; a lone statistic symbol (q < ..., p = ...) stays lower case
        return '\n'.join(line if re.match(r'[^\W\d_](\s|$)', line) else line[:1].upper() + line[1:]
                         for line in s.split('\n'))
    for ax in fig.axes:
        texts = [ax.xaxis.label, ax.yaxis.label, ax._left_title, ax.title, ax._right_title] + list(ax.texts)
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
    """Bold capital panel letters, aligned: each letter is left-aligned with its panel's y-axis label, letters are snapped
    to shared columns, and each letter's cap top is level with the cap top of the first line of its panel title (shared
    across a row of letters). Also capitalizes the axis labels. dx is kept for compatibility and ignored."""
    capitalize_labels(fig)
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

    xs = snap(xs, 0.05 * fig.bbox.width, np.min)
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
                      label={'frontal': 'frontal cortex', 'other': 'other (non-frontal)'}[g]) for g in GROUP_COLOR]
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
        ax.scatter(x, sub[ycol], s=16, color=PAIR_COLOR[t], edgecolor='k', linewidth=0.3, zorder=3)
        if y_err is not None:
            ax.errorbar(x, sub[ycol], yerr=sub[y_err], fmt='none', ecolor=PAIR_COLOR[t], elinewidth=0.6, zorder=2)
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
def figure1(res, out_dir: Path):
    sid, areas = res['session_id'], res['areas']
    pt, at = res['pair_table'], res['area_table']
    n = res['min_units']

    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.55, wspace=0.55,
                          left=0.07, right=0.955, top=0.91, bottom=0.1)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    # A: example projection onto the task axis
    ex = max(areas, key=lambda a: res['unit_counts'][a])  # area with the most available units
    proj = res['proj_example'][ex]
    ctx, blocks, tidx = res['context'], res['blocks'], np.arange(len(proj))
    # orient the axis so that visual context is positive
    sign = 1 if proj[ctx == 'vis'].mean() > proj[ctx == 'aud'].mean() else -1
    proj = sign * proj
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        if ctx[idx[0]] == 'vis':
            axA.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color='#1f6fb2', alpha=0.07, lw=0)
    for c, col, lab in [('vis', '#1f6fb2', 'visual-rewarded block'), ('aud', '#555555', 'auditory-rewarded block')]:
        axA.scatter(tidx[ctx == c], proj[ctx == c], s=5, color=col, label=lab, lw=0)
    axA.axhline(0, color='k', lw=0.6, ls=':')
    axA.set_xlabel('stimulus trial (session order)')
    axA.set_ylabel(f'projection onto task axis ({ex}, a.u.)')
    axA.set_title(f'Task axis: LDA on quiescent activity\n{ex}, {n} units, subsample 0', loc='left')
    axA.legend(loc='lower right', frameon=False, markerscale=2.5, handletextpad=0.2, ncol=1)

    # B: decoding accuracy per area
    x = np.arange(len(areas))
    at_i = at.set_index('area').loc[areas]
    null_hi = at_i['acc_null_q975'].values
    axB.fill_between([-0.6, len(areas) - 0.4], 0.5 - (null_hi.max() - 0.5), null_hi.max(), color='0.88', lw=0,
                     label='label-shuffle null (95%)')
    axB.axhline(0.5, color='k', ls='--', lw=0.7)
    for i, a in enumerate(areas):
        col = GROUP_COLOR[area_group(a)]
        axB.errorbar(i, at_i.loc[a, 'acc_cv'], yerr=at_i.loc[a, 'acc_cv_sd'], fmt='o', color=col, ms=6, capsize=0,
                     elinewidth=1, zorder=3)
        axB.plot(i, at_i.loc[a, 'acc_train'], marker='o', mfc='white', mec=col, ms=5, ls='', zorder=3)
    axB.set_xticks(x)
    axB.set_xticklabels(areas)
    for lab, a in zip(axB.get_xticklabels(), areas):
        lab.set_color(GROUP_COLOR[area_group(a)])
    axB.set_xlim(-0.6, len(areas) - 0.4)
    axB.set_ylim(min(0.25, at_i['acc_cv'].min() - at_i['acc_cv_sd'].max() - 0.03), 1.0)
    axB.set_ylabel('context decoding accuracy')
    axB.set_xlabel('area')
    axB.set_title('Task axis is predictive\nleave-one-block-out decoding', loc='left')
    h = [Line2D([], [], marker='o', color='k', ls='', ms=5, label='cross-validated (mean ± SD, subsamples)'),
         Line2D([], [], marker='o', mfc='white', mec='k', ls='', ms=5, label='training'),
         plt.Rectangle((0, 0), 1, 1, color='0.88', label='label-shuffle null (95% range)')]
    axB.legend(handles=h, loc='lower left', frameon=False, handletextpad=0.4, bbox_to_anchor=(-0.02, -0.01))

    # C: CV R^2 vs rank
    curves = res['r2_curve'].mean(axis=0)
    ranks = np.arange(1, n + 1)
    for p, (s, t) in enumerate(res['pairs']):
        typ = pt.iloc[p]['pair_type']
        axC.plot(ranks, curves[p], color=PAIR_COLOR[typ], lw=0.9, alpha=0.9)
        d = pt.iloc[p]['dim']
        axC.plot(d, np.interp(d, ranks, curves[p]), marker='o', ms=3.2, color=PAIR_COLOR[typ], mec='k', mew=0.3, ls='')
    axC.axhline(0, color='k', lw=0.6, ls=':')
    axC.set_xlabel('number of predictive dimensions (rank)')
    axC.set_ylabel('cross-validated R² (pooled over target units)')
    axC.set_title('Communication subspace\ncross-validated R² vs. rank', loc='left')
    axC.set_xlim(0.5, n + 0.5)
    axC.text(0.98, 0.98, 'dot = 1-SEM dimensionality\ncolours = pair type (legend in F)', transform=axC.transAxes,
             ha='right', va='top', fontsize=7)

    # D: R^2 heatmap with significance
    m_r2 = matrix(pt, areas, 'r2')
    sig = matrix(pt, areas, 'r2_q').values <= FDR_ALPHA
    marks = np.where(sig, '', '†')
    heatmap(axD, m_r2, 'viridis', 0, max(0.05, np.nanmax(m_r2.values)), fmt='{:.2f}',
            cbar_label='cross-validated R² (full ridge)', marks=marks)
    n_sig = int(sig[~np.eye(len(areas), dtype=bool)].sum())
    axD.set_title(f'Predictive R² of source → target\n{n_sig}/{len(pt)} pairs > trial-shuffle null (q < {FDR_ALPHA})'
                  + ('; † = n.s.' if n_sig < len(pt) else ''), loc='left')

    # E: dimensionality heatmap
    m_d = matrix(pt, areas, 'dim')
    heatmap(axE, m_d, 'Blues', 0, max(np.nanmax(m_d.values), 1), fmt='{:.1f}', cbar_label='dimensionality (1-SEM rule)')
    axE.set_title(f'Dimensionality (of {n} source units)\nmean over subsamples', loc='left')

    # F: dimensionality vs R^2 by pair type
    for t in PAIR_TYPES:
        sub = pt[pt['pair_type'] == t]
        axF.errorbar(sub['r2'], sub['dim'], xerr=sub['r2_sd'], yerr=sub['dim_sd'], fmt='o', ms=4.5, color=PAIR_COLOR[t],
                     mec='k', mew=0.3, elinewidth=0.6, capsize=0, label=PAIR_LABEL[t])
    axF.set_xlabel('cross-validated R² (full ridge)')
    axF.set_ylabel('dimensionality')
    axF.set_ylim(0, max(8, m_d.values[np.isfinite(m_d.values)].max() + 4))
    axF.set_title('Dimensionality vs. R²\nmean ± SD over subsamples', loc='left')
    pair_legend(axF, loc='upper left')

    fig.suptitle(f'Figure 1. Task axis and communication subspace, session {sid} '
                 f'({res["n_trials"]} stimulus trials, {res["n_blocks"]} blocks, quiescent window, {n} units per area, '
                 f'{res["n_subsamples"]} subsamples)', fontsize=9, x=0.07, ha='left', y=0.985)
    place_letters(fig, [axA, axB, axC, axD, axE, axF], 'ABCDEF')
    save(fig, out_dir / f'figure1_{sid}.svg')


# ----------------------------------------------------------------------------- figure 2
def alignment_marks(q, z):
    marks = np.full(q.shape, '', dtype=object)
    marks[(q <= FDR_ALPHA) & (z > 0)] = '*'
    marks[(q <= FDR_ALPHA) & (z < 0)] = '−'
    return marks


def qualify(res):
    """Pair table with qualification flags. A pair's source-side alignment is 'qualified'
    when the source task axis decodes context above the label-shuffle null AND the
    communication-subspace R² beats the trial-shuffle null (FDR); target-side alignment
    needs the target's axis to be predictive instead. FDR q-values for the alignment
    tests are recomputed over the qualified pairs only."""
    pt = res['pair_table'].copy()
    at = res['area_table'].set_index('area')
    pred = {a: bool(at.loc[a, 'acc_cv'] > at.loc[a, 'acc_null_q975']) for a in res['areas']}
    pt['source_acc_cv'] = at.loc[pt['source'], 'acc_cv'].values
    pt['source_acc_null_q975'] = at.loc[pt['source'], 'acc_null_q975'].values
    pt['source_axis_predictive'] = pt['source'].map(pred)
    pt['target_axis_predictive'] = pt['target'].map(pred)
    pt['r2_significant'] = pt['r2_q'] <= FDR_ALPHA
    pt['qualified'] = pt['source_axis_predictive'] & pt['r2_significant']
    pt['qualified_tgt'] = pt['target_axis_predictive'] & pt['r2_significant']
    pt['excess'] = pt['cos'] - pt['cos_chance']
    for pcol, qcol, mask in [('cos_p_random', 'cos_q_random', 'qualified'),
                             ('cos_tgt_p_random', 'cos_tgt_q_random', 'qualified_tgt')]:
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
    not_pred = [a for a in areas if at.loc[a, 'acc_cv'] <= at.loc[a, 'acc_null_q975']]

    def flag_sources(ax):
        ax.set_yticklabels([f'{a} \u2020' if a in not_pred else a for a in areas])
        for lab, a in zip(ax.get_yticklabels(), areas):
            lab.set_color(GROUP_COLOR[area_group(a)])

    # A: cosine heatmap, random-axis null marks
    m_cos = matrix(pt, areas, 'cos')
    marks = alignment_marks(matrix(pt, areas, 'cos_q_random').values, matrix(pt, areas, 'cos_z_random').values)
    heatmap(axA, m_cos, 'viridis', 0, 1, cbar_label='cosine similarity', marks=marks)
    n_up = int(np.sum(marks == '*'))
    n_dn = int(np.sum(marks == '−'))
    axA.set_title(f'Source task axis in comm. subspace\nq < {FDR_ALPHA}: */− {n_up}/{n_dn} of {len(pt)} qualified',
                  loc='left')
    flag_sources(axA)
    if not_pred:
        axA.set_ylabel('source area (\u2020 task axis not predictive)')

    # B: observed vs chance
    for t in PAIR_TYPES:
        sub = pt[pt['pair_type'] == t]
        sig = sub['cos_q_random'] <= FDR_ALPHA
        for flag, mfc in [(True, PAIR_COLOR[t]), (False, 'white')]:
            s2 = sub[sig == flag]
            if s2.empty:
                continue
            axB.errorbar(s2['cos_chance'], s2['cos'], yerr=s2['cos_sd'],
                         xerr=np.vstack([s2['cos_chance'] - s2['cos_chance_lo'], s2['cos_chance_hi'] - s2['cos_chance']]),
                         fmt='o', ms=4.5, color=PAIR_COLOR[t], mfc=mfc, mec=PAIR_COLOR[t], elinewidth=0.6, capsize=0)
    axB.plot([0, 1], [0, 1], 'k--', lw=0.7)
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1)
    axB.set_aspect('equal')
    axB.set_xlabel('chance: random axis in same subspace\n(mean, 95% range)')
    axB.set_ylabel('observed cosine (mean ± SD over subsamples)')
    axB.set_title('Every pair vs. its chance level\nchance = random axis, same dimensionality', loc='left')
    pair_legend(axB, loc='lower right')
    axB.text(0.03, 0.97, 'open = not significant', transform=axB.transAxes, va='top', fontsize=7)

    # C: z-score vs random-axis null by pair type
    strip_by_type(axC, pt, 'cos_z_random', 'z-score vs. random-axis null', zero_line=0)
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
            axD.scatter(np.full(qmask.sum(), i + dx), vals[qmask], s=14, color=col, marker=marker, edgecolor='k',
                        linewidth=0.3, zorder=3, alpha=alpha)
            axD.scatter(np.full((~qmask).sum(), i + dx), vals[~qmask], s=14, color='white', marker=marker, edgecolor=col,
                        linewidth=0.6, zorder=3)
            if qmask.any():
                axD.plot([i + dx - 0.13, i + dx + 0.13], [vals[qmask].mean()] * 2, color='k', lw=1.4, zorder=4)
    axD.axhline(0, color='k', ls='--', lw=0.7)
    axD.set_xticks(xs)
    axD.set_xticklabels(areas, rotation=45 if len(areas) > 6 else 0, ha='right' if len(areas) > 6 else 'center')
    for lab, a in zip(axD.get_xticklabels(), areas):
        lab.set_color(GROUP_COLOR[area_group(a)])
    axD.set_ylabel('observed − chance cosine')
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
    marks = alignment_marks(matrix(pt_t, areas, 'cos_tgt_q_random').values, matrix(pt_t, areas, 'cos_tgt_z_random').values)
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


def strip_by_type_sessions(ax, tables, ycol, ylabel, zero_line=None, log=False, text_loc='top', show_n=True):
    """Like strip_by_type, but one marker shape per session."""
    rng = np.random.default_rng(0)
    pooled = pd.concat(tables, ignore_index=True)
    for i, t in enumerate(PAIR_TYPES):
        for k, tab in enumerate(tables):
            sub = tab[tab['pair_type'] == t]
            if sub.empty:
                continue
            x = i + rng.uniform(-0.2, 0.2, len(sub))
            ax.scatter(x, sub[ycol], s=16, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[t], edgecolor='k',
                       linewidth=0.3, zorder=3)
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
    if zero_line is not None:
        ax.axhline(zero_line, color='k', ls='--', lw=0.7)
    counts = pooled['pair_type'].value_counts()
    if show_n:
        y = 0.98 if text_loc == 'top' else (0.02 if text_loc == 'bottom' else text_loc)
        ax.text(0.02, y, '\n'.join(f'{PAIR_LABEL[t]}: n = {counts.get(t, 0)}' for t in PAIR_TYPES),
                transform=ax.transAxes, va='top' if text_loc == 'top' else 'bottom', fontsize=6.5)


def area_order(areas):
    """Frontal areas first, then the others, each alphabetical."""
    return sorted(areas, key=lambda a: (area_group(a) != 'frontal', a))


def figure1_pooled(results, out_dir: Path):
    sessions = [r['session_id'] for r in results]
    tables = [qualify(r) for r in results]
    pooled = pd.concat(tables, ignore_index=True)
    areas_tab = pd.concat([r['area_table'] for r in results], ignore_index=True)
    n = results[0]['min_units']

    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.5, wspace=0.45, left=0.07, right=0.98, top=0.89, bottom=0.09)
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[1, 2])

    # A: decoding accuracy per area, every session
    order = area_order(areas_tab['area'].unique())
    null_hi = areas_tab['acc_null_q975'].mean()
    axA.fill_between([-0.6, len(order) - 0.4], 1 - null_hi, null_hi, color='0.88', lw=0)
    axA.axhline(0.5, color='k', ls='--', lw=0.7)
    for i, a in enumerate(order):
        sub = areas_tab[areas_tab['area'] == a]
        for _, r in sub.iterrows():
            k = sessions.index(r['session'])
            pred = r['acc_cv'] > r['acc_null_q975']
            col = GROUP_COLOR[area_group(a)]
            axA.errorbar(i, r['acc_cv'], yerr=r['acc_cv_sd'], fmt=MARKERS[k % len(MARKERS)], ms=6, color=col,
                         mfc=col if pred else 'white', mec=col, mew=1.0, elinewidth=0.8, capsize=0, zorder=3)
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels(order, rotation=45, ha='right')
    for lab, a in zip(axA.get_xticklabels(), order):
        lab.set_color(GROUP_COLOR[area_group(a)])
    axA.set_xlim(-0.6, len(order) - 0.4)
    axA.set_ylim(min(0.2, areas_tab['acc_cv'].min() - 0.08), 1.0)
    axA.set_ylabel('context decoding accuracy\n(leave-one-block-out,\nmean ± SD over subsamples)')
    axA.set_xlabel('area (frontal left, others right); one marker per session')
    n_pred = int((areas_tab['acc_cv'] > areas_tab['acc_null_q975']).sum())
    axA.set_title(f'Task axis: predictive in {n_pred} of {len(areas_tab)} area instances '
                  f'(filled = above label-shuffle null, open = not)', loc='left')
    session_legend(axA, sessions, loc='lower right')
    axA.text(0.01, 0.03, 'grey band = label-shuffle null, 95% range', transform=axA.transAxes, fontsize=7)

    # B: CV R^2 vs rank, every pair of every session
    ranks = np.arange(1, n + 1)
    for res, tab in zip(results, tables):
        curves = res['r2_curve'].mean(axis=0)
        for p, (s_, t_) in enumerate(res['pairs']):
            typ = tab.iloc[p]['pair_type']
            axB.plot(ranks, curves[p], color=PAIR_COLOR[typ], lw=0.6, alpha=0.8)
            d = tab.iloc[p]['dim']
            axB.plot(d, np.interp(d, ranks, curves[p]), marker='o', ms=2.6, color=PAIR_COLOR[typ], mec='k', mew=0.3, ls='')
    axB.axhline(0, color='k', lw=0.6, ls=':')
    axB.set_xlim(0.5, n + 0.5)
    axB.set_xlabel('number of predictive dimensions (rank)')
    axB.set_ylabel('cross-validated R² (pooled over target units)')
    axB.set_title(f'Communication subspace: R² vs. rank\n{len(pooled)} ordered pairs, dot = 1-SEM dim.', loc='left')
    axB.text(0.98, 0.98, 'colours = pair type (as in C)', transform=axB.transAxes, ha='right', va='top', fontsize=7)

    # C: full R^2 by pair type
    strip_by_type_sessions(axC, tables, 'r2', 'cross-validated R² (full ridge)', log=True, show_n=False)
    n_sig = int(pooled['r2_significant'].sum())
    axC.set_title(f'Predictive R² by pair type\n{n_sig}/{len(pooled)} pairs > trial-shuffle null (q < {FDR_ALPHA})',
                  loc='left')

    # D: dimensionality by pair type
    strip_by_type_sessions(axD, tables, 'dim', f'dimensionality (of {n} source units)')
    axD.set_ylim(0, max(12, pooled['dim'].max() + 3))
    axD.set_title(f'Dimensionality by pair type\nmedian {pooled["dim"].median():.1f}, range {pooled["dim"].min():.1f}'
                  f'–{pooled["dim"].max():.1f}', loc='left')

    # E: dimensionality vs R^2
    for k, tab in enumerate(tables):
        for t in PAIR_TYPES:
            sub = tab[tab['pair_type'] == t]
            axE.scatter(sub['r2'], sub['dim'], s=18, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[t],
                        edgecolor='k', linewidth=0.3)
    from scipy import stats
    rho, pval = stats.spearmanr(pooled['r2'], pooled['dim'])
    axE.set_xscale('log')
    axE.set_xlabel('cross-validated R² (full ridge)')
    axE.set_ylabel('dimensionality')
    axE.set_ylim(0, max(10, pooled['dim'].max() + 2))
    axE.set_title(f'Dimensionality vs. R²\nSpearman ρ = {rho:.2f}, p = {pval:.2g}', loc='left')
    session_legend(axE, sessions, loc='upper left')

    fig.suptitle(f'Figure 1 (pooled). Task axis and communication subspace over {len(results)} sessions: '
                 f'{len(areas_tab)} area instances ({areas_tab["area"].nunique()} distinct areas), {len(pooled)} ordered pairs, '
                 f'{n} units per area', fontsize=9, x=0.07, ha='left', y=0.98)
    place_letters(fig, [axA, axB, axC, axD, axE], 'ABCDE', dx=-0.05)
    save(fig, out_dir / 'figure1_pooled.svg')


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

    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.5, wspace=0.45, left=0.07, right=0.98, top=0.89, bottom=0.09)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0:2])
    axE = fig.add_subplot(gs[1, 2])

    # A: observed vs chance, source side, qualified
    for k, t in enumerate(tables):
        for typ in PAIR_TYPES:
            sub = t[t['pair_type'] == typ]
            axA.scatter(sub['cos_chance'], sub['cos'], s=18, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[typ],
                        edgecolor='k', linewidth=0.3)
    axA.plot([0, 1], [0, 1], 'k--', lw=0.7)
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)
    axA.set_aspect('equal')
    axA.set_xlabel('chance (random axis in same subspace)')
    axA.set_ylabel('observed cosine')
    n_up = int(((pooled['cos_q_random'] <= FDR_ALPHA) & (pooled['cos_z_random'] > 0)).sum())
    n_dn = int(((pooled['cos_q_random'] <= FDR_ALPHA) & (pooled['cos_z_random'] < 0)).sum())
    axA.set_title(f'Source task axis in comm. subspace\n{n_up} above / {n_dn} below chance of {len(pooled)} qualified pairs',
                  loc='left')
    session_legend(axA, sessions, loc='lower right')

    # B: by pair type
    strip_by_type_sessions(axB, tables, 'cos_z_random', 'z-score vs. random-axis null', zero_line=0, text_loc=0.08)
    axB.set_title('By pair type\nz-score of the subsample-averaged cosine', loc='left')

    # C: target side observed vs chance
    for k, t in enumerate(tables_t):
        for typ in PAIR_TYPES:
            sub = t[t['pair_type'] == typ]
            axC.scatter(sub['cos_tgt_chance'], sub['cos_tgt'], s=18, marker=MARKERS[k % len(MARKERS)], color=PAIR_COLOR[typ],
                        edgecolor='k', linewidth=0.3)
    axC.plot([0, 1], [0, 1], 'k--', lw=0.7)
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    axC.set_aspect('equal')
    axC.set_xlabel('chance (random axis in predicted dims)')
    axC.set_ylabel('observed cosine (target task axis)')
    n_up = int(((pooled_t['cos_tgt_q_random'] <= FDR_ALPHA) & (pooled_t['cos_tgt_z_random'] > 0)).sum())
    n_dn = int(((pooled_t['cos_tgt_q_random'] <= FDR_ALPHA) & (pooled_t['cos_tgt_z_random'] < 0)).sum())
    axC.set_title(f'Target side: task axis in predicted dims\n{n_up} above / {n_dn} below chance of {len(pooled_t)} qualified',
                  loc='left')
    pair_legend(axC, loc='lower right')

    # D: by area, as source and as target
    pooled_all = pooled_all.assign(excess_tgt=pooled_all['cos_tgt'] - pooled_all['cos_tgt_chance'])
    order = area_order(set(pooled['source']) | set(pooled_t['target']))
    rng = np.random.default_rng(0)
    for i, a in enumerate(order):
        col = GROUP_COLOR[area_group(a)]
        src = pooled_all[(pooled_all['source'] == a) & pooled_all['qualified']]['excess'].values
        tgt = pooled_all[(pooled_all['target'] == a) & pooled_all['qualified_tgt']]['excess_tgt'].values
        for vals, dx, marker, alpha in [(src, -0.18, 'o', 1.0), (tgt, 0.18, 's', 0.65)]:
            if len(vals) == 0:
                continue
            axD.scatter(i + dx + rng.uniform(-0.08, 0.08, len(vals)), vals, s=13, marker=marker, color=col,
                        edgecolor='k', linewidth=0.3, zorder=3, alpha=alpha)
            axD.plot([i + dx - 0.14, i + dx + 0.14], [vals.mean()] * 2, color='k', lw=1.4, zorder=4)
    axD.axhline(0, color='k', ls='--', lw=0.7)
    axD.set_xticks(range(len(order)))
    axD.set_xticklabels(order, rotation=45, ha='right')
    for lab, a in zip(axD.get_xticklabels(), order):
        lab.set_color(GROUP_COLOR[area_group(a)])
    axD.set_xlim(-0.6, len(order) - 0.4)
    axD.set_ylabel('observed − chance cosine')
    axD.set_xlabel('area (frontal left, others right); points = partners × sessions, black = mean')
    axD.set_title('By area, qualified pairs only (areas with a predictive task axis)\nas source (●, its task axis '
                  'in the comm. subspace) and as target (■, its task axis in the predicted dims)', loc='left')
    ymax = max(pooled_all['excess'].max(), pooled_all['excess_tgt'].max())
    axD.set_ylim(-0.25 * ymax, ymax * 1.1)
    h = [Line2D([], [], marker='o', color='0.4', ls='', ms=5, label='as source'),
         Line2D([], [], marker='s', color='0.4', ls='', ms=5, alpha=0.65, label='as target'),
         Line2D([], [], color='k', lw=1.4, label='mean')]
    axD.legend(handles=h, loc='lower left', frameon=False, ncol=3, columnspacing=1.2, handletextpad=0.4,
               bbox_to_anchor=(-0.01, -0.01))

    # E: alignment vs task-axis predictive accuracy
    for k, t in enumerate(tables_all):
        for g in GROUP_COLOR:
            sub = t[[area_group(a) == g for a in t['source']]]
            q = sub['qualified'].values
            axE.scatter(sub['source_acc_cv'][q], sub['excess'][q], s=18, marker=MARKERS[k % len(MARKERS)],
                        color=GROUP_COLOR[g], edgecolor='k', linewidth=0.3)
            axE.scatter(sub['source_acc_cv'][~q], sub['excess'][~q], s=18, marker=MARKERS[k % len(MARKERS)],
                        color='white', edgecolor=GROUP_COLOR[g], linewidth=0.6)
    axE.axvline(pooled_all['source_acc_null_q975'].mean(), color='k', ls='--', lw=0.7)
    axE.axhline(0, color='k', ls=':', lw=0.7)
    rho, pval = stats.spearmanr(pooled['source_acc_cv'], pooled['excess'])
    axE.set_xlabel('source task axis: decoding accuracy')
    axE.set_ylabel('observed − chance cosine')
    axE.set_title(f'Alignment vs. task-axis accuracy\nρ = {rho:.2f}, p = {pval:.2f}; open = excluded', loc='left')
    axE.set_ylim(-0.25 * ymax, ymax * 1.1)
    group_legend(axE, loc='lower right')

    fig.suptitle(f'Figure 2 (pooled). Alignment over {len(results)} sessions: {len(pooled)} of {len(pooled_all)} ordered '
                 f'pairs qualify (source task axis predictive and R² significant); random-axis null',
                 fontsize=9, x=0.07, ha='left', y=0.98)
    place_letters(fig, [axA, axB, axC, axD, axE], 'ABCDE', dx=-0.05)
    save(fig, out_dir / 'figure2_pooled.svg')
    return pooled_all


def pooled_figures(results, out_dir: Path):
    figure1_pooled(results, out_dir)
    return figure2_pooled(results, out_dir)


def main(argv):
    code_dir = Path(__file__).resolve().parent
    results_dir = Path(os.environ.get('RESULTS_DIR', code_dir.parent / 'results'))
    sessions = argv or sorted(p.parent.name for p in results_dir.glob('*/alignment_results.pkl'))
    results = []
    for sid in sessions:
        with open(results_dir / sid / 'alignment_results.pkl', 'rb') as f:
            res = pickle.load(f)
        results.append(res)
        out = results_dir / sid / 'figures'
        figure1(res, out)
        figure2(res, out)
        print(f'{sid}: figures written to {out}')
    if len(results) > 1:
        out = results_dir / 'pooled'
        out.mkdir(parents=True, exist_ok=True)
        pooled_figures(results, out)
        print(f'pooled figure written to {out}')


if __name__ == '__main__':
    main(sys.argv[1:])
