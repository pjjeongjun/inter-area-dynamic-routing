# Task axis vs. communication subspace across all area pairs

Question: is the alignment between an area's task (context) axis and its communication
subspace with another area specific to the prefrontal network, or universal?

Code: `code/task_axis_comm_subspace.py` (analysis), `code/make_alignment_figures.py`
(figures), `code/notebooks/task_axis_vs_communication_subspace_all_pairs.ipynb` (wrapper
for `run_capsule.py`). Outputs per session in `results/<session>/`:
`alignment_results.pkl`, `task_axis_by_area.csv`, `alignment_by_pair.csv`,
`figures/figure1_<session>.svg`, `figures/figure2_<session>.svg`; pooled over sessions in
`results/pooled/` (`figure1_pooled.svg`, `figure2_pooled.svg`, `alignment_by_pair_pooled.csv`).

## Methods

### Data and preprocessing
- Dynamic Routing Neuropixels sessions (NWB zarr). Stimulus trials only (catch trials
  dropped). Context label = `rewarded_modality` (visual vs. auditory block).
- Activity = spike count in the quiescent window before stimulus onset
  (`quiescent_start_time` to `quiescent_stop_time`, median 1.5 s) divided by its duration,
  one value per trial and unit, z-scored per unit over trials.
- **Nuisance regression (added 2026-10-05).** Each unit's z-scored rate is regressed
  (OLS with intercept) on: the previous trial's stimulus identity (one-hot over catch /
  sound1 / sound2 / vis1 / vis2, catch as reference), whether the animal responded on
  the previous trial (`is_response`), whether it was rewarded (`is_rewarded`), and (fourth
  pass) the z-scored mean running speed and mean pupil area in the quiescent window
  (pupil from `eye_tracking.pupil_area`, bad frames excluded, missing trials filled with
  the session mean). The previous trial is the trial with `trial_index − 1` in the full
  trial table, so catch and instruction trials count as previous trials. The residual is
  re-z-scored. The first trial of the session has no predecessor and is dropped.
  `NUISANCE_REGRESSORS` controls the set: `none` reproduces the first-pass analysis
  (`results/`), `prev_stim,prev_response,prev_reward` the third pass
  (`results_regress_prev/`), and the default (all five) the fourth pass
  (`results_regress_full/`).
- **No condition mean is subtracted** (the context-mean option of the earlier notebook
  was removed). The residual therefore still contains the across-context mean shift as
  well as trial-to-trial variability. Note that previous response is itself correlated
  with context (response rate is higher in auditory-rewarded blocks), so regressing it
  out removes some context-correlated variance too.

### Qualified cells and subsampling
- QC-pass single units (`is_qc_pass` and `decoder_label == 'sua'`). `is_qc_pass` =
  default QC (presence ratio > 0.7, ISI-violations ratio < 0.5, amplitude cutoff < 0.1)
  AND `is_not_drift` (`activity_drift` < 0.1), so units with slow firing-rate drift over
  the session are excluded up front (JeongJun, 2026-10-05).
- An area is included if it has at least 30 such units. Every included area is
  subsampled to exactly 30 units, 10 times with different seeds (seed 0 reproduces the
  subsample of the earlier notebooks). Every statistic below is computed per subsample
  and averaged, so an area with 139 units and an area with 32 units are compared on
  equal footing (same dimensionality of the unit space, same estimation noise).
- Area groups for summaries: frontal cortex (MOs, PL, ILA, ACAd, ACAv, ORB*, FRP, AI*)
  vs. everything else ("other", subcortical in the sessions analysed so far).

### Task axis (per area)
- Linear discriminant analysis (LDA, SVD solver) of the 30-unit activity on the context
  label, fit on all trials. The axis is the unit-normalised LDA weight vector in the
  area's z-scored unit space (the direction `Σ_w⁻¹ Δμ`).
- Predictive accuracy (trial-wise since 2026-10-08, JeongJun's decision): stratified 10-fold
  cross-validation over trials drawn from every block, LDA with equal class priors, accuracy
  pooled over held-out trials; the trial-shuffle and block-permutation nulls use the same folds
  (block-permuted labels define their own stratified folds), and every context decoding in the
  post-processing (subspace readers, nearest centroid, cross-block control qualification)
  uses these folds (`CONTEXT_CV` in `task_axis_comm_subspace.py`; `CONTEXT_CV=block` restores
  the design below). Because held-out trials share blocks with training trials, any slow
  signal that differs between blocks also decodes the labels, so accuracies are higher and the
  block-permutation null (not the trial-shuffle null) is the reference for "above chance".
  The leave-one-block-out R² (`patch_lobo_r2.py`) and the cross-block split of the control
  stay block-based: holding out blocks is what those two quantities measure.
- Earlier design (balanced, 2026-10-06 to 2026-10-08): one block of each context is held
  out (9 folds for 3 + 3 blocks), an LDA with equal class priors is trained on the remaining
  2 + 2 blocks and tested on the two held-out blocks; accuracy is pooled over held-out trials.
  A block has a single context, so this is generalisation to unseen blocks, and training and
  test sets are balanced, so chance is 50%. (Before: leave-one-block-out with
  training-proportion priors, whose 3-vs-2 imbalance biased accuracy and its null below 50%.)
  Training accuracy is also reported.
- Nulls (revised 2026-10-06). (a) Trial-shuffle null: the same cross-validation with
  context labels shuffled across trials (200 draws). Its centre is 50% by construction
  and it ignores the block structure, so it is reported for reference only. (b)
  Block-permutation null: the context labels are permuted across whole blocks, keeping
  three blocks per context; every assignment except the observed one and its mirror
  image (18 draws for 6 blocks). Labels stay constant within blocks, so this null shows
  what slow block-constant signals achieve without a context signal (the folds are
  re-derived from the permuted labels so the design stays balanced). An axis is
  *predictive* when its accuracy exceeds every block-permutation draw (p = 1/19); this
  is the qualification criterion everywhere.
- Every null draw (one set of permuted labels) is generated once per session and
  applied to all 10 unit subsamples, and the null is averaged across subsamples draw by
  draw. The subsamples share the same trials, so this keeps the across-subsample
  dependence of the observed statistic in the null. (Before 2026-10-06 each subsample
  drew its own permutations, which shrank every null's spread by about √10 and made the
  grey 95% bands in the figures three times too narrow.)

### Communication subspace (per ordered pair, source → target)
- Ridge regression from the 30 source units to the 30 target units; the ridge penalty
  is chosen by efficient leave-one-out `RidgeCV` on a 25-point log grid (0.1 to 10⁴).
- Reduced-rank regression: SVD of the fitted prediction `X B`; the rank-r model keeps
  the first r predictive dimensions (Semedo et al., 2019).
- Predictive performance: 10-fold cross-validated R², pooled over target units
  (1 − SSE/SST on held-out trials), as a function of rank 1..30, and for the full ridge
  model.
- Dimensionality: the smallest rank whose CV R² is within one SEM (over folds) of the
  best rank (the standard 1-SEM rule).
- Subspace: source-side orthonormal basis `Q_src` (30 × d) of the first d predictive
  source directions, fit on all trials; target-side basis `Q_tgt` (30 × d) of the first
  d predicted target directions.
- Significance of the full-model R²: trial-shuffle null (target trials permuted relative
  to source trials, 1000 permutations generated once per session and applied to every
  subsample, same ridge penalty), one-sided, Benjamini–Hochberg FDR over the ordered
  pairs of the session. The same draw-sharing applies to the rank-d R² null
  (`r2_dim_null.py`) and the held-out-block R² null (`patch_lobo_r2.py`).

### Alignment
- Cosine similarity = `‖Q_srcᵀ w_s‖`, the norm of the source area's unit task axis `w_s`
  projected onto the pair's source-side communication subspace. 1 = the axis lies in
  the subspace, 0 = orthogonal. Angle = arccos(cosine). The target-side analogue
  `‖Q_tgtᵀ w_t‖` asks whether the *target's* task axis lies in the directions the
  source predicts.
- Significance null (revised 2026-10-06): the **shuffled-label axis null**. An LDA axis
  is fit exactly like the task axis but on trial-shuffled context labels (1000 draws,
  the same permutation for every unit subsample) and projected onto the same
  communication subspace. An LDA axis is Σ⁻¹Δμ, so whatever the labels it favours
  low-variance directions of the area; this null keeps that estimator geometry and asks
  whether the *context* axis in particular lies in the subspace. The null is averaged
  across subsamples draw by draw; p-values are two-sided (so pairs *below* the null can
  be detected), FDR-corrected over the pairs of a session; z = (observed − null mean) /
  null SD. "Excess" in the figures is observed − null mean. (This null was computed in
  the first pass, dropped on 2026-10-05, and reinstated on 2026-10-06 together with the
  draw-sharing fix.)
- Reference 1, random-axis chance: a uniformly random unit vector in the same
  30-dimensional unit space projected onto the same subspace (1000 draws per subsample,
  pooled). This depends only on the dimensionality d (expected squared cosine = d/30).
  It is not tied to the data, so it cannot share the subsamples' trial noise and is
  reported as a chance level (`cos_chance`, with its single-draw 95% range), not used
  for significance.
- Reference 2, block-permuted axis: the LDA axis fit on block-permuted labels (18
  draws, see the task-axis nulls) projected onto the same subspace (`cos_block_*`). It
  shows how well *any* block-constant signal aligns with the subspace; the cross-block
  control below addresses the same question with held-out blocks.

### Figures
- **Figure 1** (per session; redesigned 2026-10-05 so both objects are judged by the same
  two quantities): A, task-axis in-sample R² per area (fraction of the variance of the
  activity projected on the LDA axis that is explained by context; filled = predictive
  in B); B, task-axis predictive accuracy (held-out blocks) vs. the block-permutation
  null (grey: per-instance null range); C, communication-subspace in-sample R² (reduced-rank fit at the 1-SEM
  dimensionality, pooled over target units) as a source × target heatmap; D, the
  cross-validated R² of the same rank-d fit, with trial-shuffle significance (rank-d
  null from `r2_dim_null.py` when available, else the full-model null). C and D share
  one colour scale.
- **Supplementary Figure 1** (per session): A, cross-validated R² vs. rank for every
  pair with the 1-SEM dimensionality marked; B, dimensionality heatmap.
- **Figure 2** (per session): A, cosine heatmap with pairs above (*) or below (−) the
  shuffled-label axis null; B, observed vs. null per pair; C, z-score by pair type; D,
  observed − null by area, as source and as target; E, target-side alignment heatmap.
- **Figure 1 (pooled)**: A, task-axis in-sample R² for every area instance of every
  session; B, task-axis predictive accuracy; C, communication-subspace in-sample R² by
  pair type; D, its cross-validated R² by pair type with the trial-shuffle count. C and D
  share a log scale.
- **Supplementary Figure 1 (pooled)**: R² vs. rank for all pairs; dimensionality by pair
  type. (The dimensionality-vs-R² relationship was dropped at JeongJun's request.)
- **Figure 2 (pooled)**: A, observed − null (shuffled-label axis) by area as source and as
  target; B, alignment vs. task-axis decoding accuracy. Supplementary Figure 2 (pooled):
  observed vs. the null mean, source and target side.

## Results (3 sessions, 2026-10-05)

Sessions analysed locally: 743199_2024-12-05 (6 areas: MOs, PL, ACAd | CP, MRN, RN;
30 ordered pairs), 713655_2024-08-09 (9 areas: MOs | SSs, ECT, LSr, MOp, CP, TEa, AUDp,
AUDv; 72 pairs), 708016_2024-04-29 (7 areas: MOs, AId | SSp, MOp, CA3, MGv, ProS;
42 pairs). 144 ordered pairs, 18 distinct areas in total. The two extra sessions are in
the capsule's 12; the notebook wrapper runs the same pipeline on the rest.

### Is the task axis meaningful? (Fig. 1B)
- Training accuracy is 0.7–0.95 everywhere, but leave-one-block-out accuracy separates
  the areas. Above the label-shuffle null (97.5th percentile ≈ 0.52) in 13 of 22 area
  instances: all six areas of 743199 (0.71–0.83), MOs/AId/ProS in 708016 (0.55, 0.70,
  0.73), and SSs/ECT/MOp/CP/MOs/TEa in 713655 (0.52–0.70).
- Not predictive (at or below chance in held-out blocks): SSp 0.41, MOp 0.47, CA3 0.25,
  MGv 0.51 (708016); LSr 0.38, AUDp 0.41, AUDv 0.48 (713655). These axes fit the
  training blocks (high training accuracy) but do not generalise: they capture
  block-specific structure rather than context per se.

### Is the communication subspace meaningful? (Fig. 1C–F)
- Every one of the 144 ordered pairs has a full-model CV R² above the trial-shuffle null
  (FDR q < 0.05). R² is modest and session dependent: 0.05–0.45 in 743199 (largest for
  midbrain pairs MRN↔RN and frontal→MRN/RN), 0.01–0.11 in 713655, 0.01–0.13 in 708016.
- The subspaces are low-dimensional: 2–8.5 of 30 source dimensions (median ≈ 4), with
  the CV R² curve saturating by rank ~5. Dimensionality rises weakly with R².

### Qualified pairs (added 2026-10-05, second pass)
Following JeongJun's criterion, the alignment figures (Fig. 2, pooled Fig. 2) now include only
pairs where **both** ingredients are meaningful: the source area's task axis decodes
context above the label-shuffle null in held-out blocks, and the communication
subspace's R² beats the trial-shuffle null (FDR q < 0.05). The target-side panel
requires the *target's* task axis to be predictive instead. FDR over the alignment tests
is recomputed over qualified pairs only. Excluded cells are grey in the heatmaps and
open markers in the per-area panels. 96 of 144 ordered pairs qualify (743199: 30/30;
713655: 48/72, excluding LSr, AUDp, AUDv as sources; 708016: 18/42, excluding SSp, MOp,
CA3, MGv as sources). Every pair's R² was significant, so the filter is driven by the
task axis. Column `qualified` in `alignment_by_pair_qualified.csv` /
`alignment_by_pair_pooled.csv`.

### Alignment, qualified pairs (Fig. 2, pooled Fig. 2)
- Cosine 0.51–0.92 against chance 0.23–0.51; all 96 qualified pairs above the
  random-axis chance (z = 4.5–15.5), none below. Target side: all 96 target-qualified pairs
  above chance.
- By pair type (observed − chance): frontal→frontal 0.44 (n = 8), frontal→other 0.44
  (27), other→frontal 0.38 (16), other→other 0.40 (45). Frontal sources vs non-frontal
  sources 0.44 vs 0.40 (Mann–Whitney p = 0.07); frontal→frontal vs all other pairs
  p = 0.63.
- By source area: PL 0.51, SSs 0.51, ECT 0.47, ACAd 0.46, MOp 0.45, MOs 0.42, AId 0.41,
  CP 0.40, TEa 0.37, ProS 0.32, RN 0.31, MRN 0.23.
- Among qualified pairs the alignment excess is, if anything, weakly *negatively*
  related to the source axis's decoding accuracy (Spearman ρ = −0.21, p = 0.04), and
  weakly positively to R² (ρ = 0.18, p = 0.09).

### Alignment, all pairs (first pass, kept for reference)
- Cosine similarity between the source task axis and the communication subspace is
  0.48–0.93 across all 144 pairs, against a dimensionality-matched chance of 0.23–0.51.
  **Every pair is above chance** under the random-axis null (z = 4.5–16.5); no pair is
  below chance. The target-side
  alignment is likewise above chance for every pair.
- By pair type, pooled (observed − chance cosine): frontal→frontal 0.44 (n = 8),
  frontal→other 0.44 (27), other→frontal 0.40 (27), other→other 0.39 (82). Frontal
  sources are slightly higher than non-frontal sources (0.44 vs 0.39, Mann–Whitney
  p = 0.04), but frontal→frontal is not different from all other pairs (p = 0.47).
  Alignment is somewhat higher for pairs with larger R² (Spearman ρ = 0.28,
  p < 0.001) and unrelated to dimensionality (ρ = 0.07).
- By source area, pooled: PL 0.51, SSs 0.51, ECT 0.47, ACAd 0.46, MOp 0.43, AUDv 0.43,
  MOs 0.42, MGv/AId/AUDp 0.41, CP 0.40, LSr/SSp/TEa 0.37, CA3 0.33, ProS 0.32,
  RN 0.31, MRN 0.23. Somatosensory and ectorhinal cortex sit next to PL at the top;
  the lowest values are midbrain (MRN, RN) and hippocampal (CA3, ProS).

So, with or without the qualification filter, the answer to the big-picture question
is "universal": the alignment is not a prefrontal speciality. Among areas whose task
axis genuinely predicts context, it is present for somatosensory, ectorhinal, motor and
temporal cortex, striatum, prosubiculum and midbrain, in both directions, at the same
level as for PL/ACAd/MOs.

### Caveat that still applies: alignment does not scale with context information (pooled Fig. 2E)
Restricting to qualified pairs removes the areas whose "task axis" is pure block
structure, which is the right thing to do. It does not remove the mechanism below: the
excluded pairs (open markers in pooled Fig. 2E) were just as aligned as the qualified ones,
so above-chance alignment by itself does not certify that the aligned direction carries
context. The controls listed at the end would address this directly.

- Alignment excess is uncorrelated with how well the source task axis actually decodes
  context in held-out blocks (Spearman ρ = 0.01, p = 0.91 across 144 pairs; pairs with a
  predictive source axis 0.41 vs non-predictive 0.39, p = 0.30). CA3 in 708016 decodes
  context at 25% (below chance) yet its axis is "aligned" with every communication
  subspace at z ≈ 6.
- Interpretation. Without context-mean subtraction, both estimates are dominated by slow,
  block-wise shared fluctuations. Because QC already removes units with slow firing-rate
  drift, these are shared population state, not unit-level recording drift. An LDA axis
  fit to block-structured labels on all trials points along whatever direction varies
  most between blocks (true context signal, or arousal, satiety, engagement, time in
  session), and the ridge/RRR subspace fit on
  the same trials captures the shared part of exactly those slow fluctuations. The two
  therefore align whether or not the axis carries generalisable context information.
  This is the same phenomenon seen earlier in session 743199, where subtracting the
  context mean dropped the cosine to the random-axis baseline.
- Consequence. The current analysis establishes that task-axis / communication-subspace
  alignment is ubiquitous rather than frontal-specific, but it cannot yet say whether
  that reflects routing of *context* information or shared slow state. Two controls
  would separate these without re-introducing the context-mean subtraction:
  (1) define the task axis on held-out blocks (fit LDA on blocks 1–4, measure alignment
  with a subspace fit on blocks 5–6, and rotate), so that only the generalisable part of
  the axis counts; (2) a time-matched null: shuffle context labels at the level of
  blocks, or compare with an "axis" fit to a slow nuisance variable (trial index, running
  speed, pupil) to see whether it aligns just as well.

### Results with previous-trial regressors removed (`results_regress_prev/`)
- Regressing out previous stimulus, response and reward removes 2–8% of each area's
  z-scored quiescent variance (mean per area; highest in ProS 8% and MRN 6%).
- Task-axis decoding accuracy drops by 0.03 on average (0.58 → 0.55 across the 22 area
  instances, paired Wilcoxon p < 1e-20), consistent with previous response being
  context-correlated. Two more areas fall below the label-shuffle null (ECT and CP in
  713655), so 13 instead of 15 area instances and 80 instead of 96 ordered pairs
  qualify.
- Communication-subspace R² and dimensionality are essentially unchanged (R² 0.073 →
  0.068, dimensionality 4.0 → 4.0); all pairs still beat the trial-shuffle null.
- Alignment is unchanged: observed − chance cosine 0.405 → 0.402 (paired p = 0.39),
  z 11.1 → 11.1. All 80 qualified pairs are above the random-axis chance and none
  below. By pair type: frontal→frontal 0.44 (n = 8), frontal→other 0.44 (27),
  other→frontal 0.36 (14), other→other 0.40 (31); frontal→frontal vs rest p = 0.36.
- Conclusion: trial-history signals are not what carries the alignment.

### Controls (2026-10-05, on the previous-trial-regressed data; `alignment_controls.py`)
(The nuisance-axis control below was removed from the scripts after this pass at
JeongJun's request, once running speed and pupil were added to the regression; its
numbers are kept here for the record.)
Qualification is recomputed inside the controls script (task axis above the
label-shuffle null, leave-one-block-out); 88 of 144 pairs qualify there (80 in the main
run; ECT and CP in 713655 are borderline).

**Cross-block control.** Task axis fit on 3 blocks, communication subspace fit on the
other 3 blocks, all 18 valid splits, averaged.
- Cosine: all trials 0.75 → same 3 blocks 0.64 → axis from the other 3 blocks 0.48,
  against a chance of 0.30 for the half-data subspaces. The cross-block cosine stays
  above chance in 82 of 88 qualified pairs (none below), but the excess over chance is
  about half of the same-blocks excess (median ratio 0.52).
- So roughly half of the alignment is a context direction that generalises across
  blocks and half is block-specific shared structure that the all-trials analysis
  cannot distinguish from context.
- Pair types: frontal→frontal z 7.9, frontal→other 7.7, other→frontal 5.8, other→other
  6.1; frontal→frontal vs rest p = 0.45. Frontal sources slightly higher than
  non-frontal sources (excess 0.20 vs 0.16, p = 0.01). The generalising component is
  therefore also not frontal-specific.
- The cross-block axes decode context in the held-out blocks at 0.64 on average, almost
  as well as the leave-one-block-out axes (0.65), so they are genuine context axes.

**Nuisance-axis control.** Per area, a ridge axis predicting trial index, running speed
or pupil area (quiescent-window means) from the 30 units, projected onto the same
full-data communication subspace as the task axis.
- Alignment excess (cosine − chance) over the 88 qualified pairs: task axis 0.41;
  trial-index axis 0.30 (84/88 above chance); running-speed axis 0.51 (88/88 above
  chance, significantly *more* aligned than the task axis, Wilcoxon p = 5e-14);
  pupil axis 0.20 (75/88 above chance). Across pairs the task-axis and running-axis
  excesses are strongly correlated (Spearman ρ = 0.73).
- The nuisance axes are meaningful in their own right: the running axis predicts
  held-out-block running speed with R² 0.27–0.81 (median 0.43); the trial-index axis
  R² 0.11 median; the pupil axis does not generalise (R² ≈ 0).
- The task axis is only weakly related to the running axis within an area (|cos| 0.08
  to 0.43, median 0.22, against 0.15 for two random directions), so the task axis and
  the running axis are largely different directions that are *both* inside the
  communication subspace.

**Interpretation.** The communication subspace between any two areas is a
low-dimensional space that contains every slow, population-wide signal: context,
running, time in session, and (weakly) pupil. The task axis is in it everywhere, but so
is a running-speed axis, and more tightly. About half of the task-axis alignment
survives when the axis is estimated on different blocks than the subspace, so context
coding genuinely overlaps with inter-areal shared variance; the other half reflects
block-specific shared fluctuations. None of this is frontal-specific. A natural next
step is to regress running speed (and pupil) out of the quiescent activity as well and
ask how much task-axis alignment remains.

### Results with the full nuisance set (`results_regress_full/`: previous stimulus /
response / reward + running speed + pupil area; the current default)
- The regressors now remove 9% of quiescent variance on average per area (6–13% in
  cortex, 26–27% in MRN and RN, where running speed is strongly encoded).
- Task-axis decoding accuracy falls further (0.59 → 0.53 on average across the 22 area
  instances); 14 instances remain predictive (ECT and CP in 713655 drop out, MGv in
  708016 comes in), giving 86 qualified pairs.
- Communication-subspace R² falls by a third (0.073 → 0.048 over all 144 pairs, paired
  p < 1e-24), i.e. running and pupil account for a sizeable share of the shared
  variance, but every pair still beats the trial-shuffle null; dimensionality is
  unchanged (4.2 → 4.5).
- The task-axis alignment does **not** fall. Over all 144 pairs the observed − chance
  cosine goes from 0.405 to 0.415 (paired p = 0.02, i.e. a slight increase) and the
  z-score from 11.1 to 11.5. Over the 86 qualified pairs: cosine 0.78 vs chance 0.36,
  excess 0.42, all 86 above chance, none below. By pair type: frontal→frontal 0.43
  (n = 8), frontal→other 0.45 (27), other→frontal 0.40 (16), other→other 0.40 (35);
  frontal→frontal vs rest p = 0.72.
- Cross-block control on these data (80 qualified pairs): all-trials 0.78 → same 3
  blocks 0.64 → axis from the other 3 blocks 0.49, chance 0.31; cross-block cosine above
  chance in 78/80 pairs; median fraction of the same-blocks excess that generalises 0.56
  (0.52 before adding running and pupil). Pair types: frontal→frontal z 7.0,
  frontal→other 7.5, other→frontal 6.0, other→other 7.5.

Summary across the three regression settings (qualified pairs):

| regressed out | nuisance R² | predictive areas | qualified pairs | comm. R² | cosine | chance | excess | pairs above chance |
|---|---|---|---|---|---|---|---|---|
| nothing | 0 | 15/22 | 96 | 0.091 | 0.76 | 0.35 | 0.41 | 96/96 |
| prev. stim/resp/reward | 0.04 | 13/22 | 80 | 0.095 | 0.75 | 0.34 | 0.41 | 80/80 |
| + running speed + pupil | 0.09 | 14/22 | 86 | 0.066 | 0.78 | 0.36 | 0.42 | 86/86 |

Interpretation. Removing locomotion and pupil (the signals whose axes were most
aligned with the subspace in the earlier nuisance-axis control) shrinks the
communication subspace's predictive power but leaves the task axis exactly as embedded
in what remains, and the cross-block half of the alignment is intact. So the
context direction sits in the inter-areal shared subspace independently of trial
history, running and arousal, for every area pair with a predictive task axis, frontal
or not.

### Fit quality in matched terms (added 2026-10-05; `patch_fit_quality.py` back-filled the
existing runs, the module now computes these in `analyse_session`)
- Task axis, in-sample R² (variance of the projection explained by context, mean over
  subsamples): 0.33–0.57 in areas with a predictive axis and 0.15–0.32 in areas without
  (full-regression run; e.g. ACAd 0.49, MRN 0.57, CA3 0.15). Predictive accuracy is the
  held-out-block quantity already used for qualification.
- Communication subspace at its 1-SEM dimensionality, full-regression run: in-sample R²
  0.01–0.38 (mean 0.08), cross-validated R² 0.002–0.33 (mean 0.06); the drop from
  in-sample to cross-validated is modest, as expected for a rank-4 model, and every pair
  beats the trial-shuffle null.

### Files
- Per session: `results/<session>/alignment_by_pair.csv` (one row per ordered pair:
  R², dimensionality, cosine, chance, z and FDR q for both nulls, target-side values,
  angles), `task_axis_by_area.csv`, `figures/figure1_*.svg`, `figures/figure2_*.svg`.
- Pooled: `results/pooled/alignment_by_pair_pooled.csv`, `figure1_pooled.svg`, `figure2_pooled.svg`.
- Same layout under `results_regress_prev/` for the previous-trial-regressed run; controls in
  `results_regress_prev/<session>/controls_*.csv` and `results_regress_prev/pooled/figure_controls_pooled.svg`
  (`code/alignment_controls.py`, `code/make_controls_figure.py`).
