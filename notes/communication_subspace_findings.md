# Communication-subspace analysis: findings and decisions

This note summarizes the exploratory analyses of inter-area population prediction in the Allen Dynamic Routing sessions. It records the current analysis choices, numerical results, interpretations, and important caveats.

## Scientific objective

The main goal is to identify a low-dimensional communication subspace between simultaneously recorded populations, with particular interest in motor and frontal regions such as MOs, PL, and CP.

For a communication-subspace interpretation, prediction should concern trial-specific shared variability rather than only condition-averaged task responses. The primary nuisance model therefore uses the joint categorical variable `stimulus_context`, representing all physical-stimulus-by-task-context combinations. Running speed and trial index are also included.

Task-inclusive prediction remains useful as a complementary analysis, provided the task contribution and incremental source-area contribution are reported separately.

## Current primary configuration

- Source and target: MOs and CP, with MOs to CP as the primary direction.
- Population size: 30 neurons per area.
- Trial population: recognized visual and auditory stimuli pooled across both rewarded contexts.
- Activity representation: one mean firing rate per neuron over the complete quiescent period of each trial.
- Mean quiescent duration: approximately 1.47 s.
- Nuisance variables: joint `stimulus_context`, mean running speed over the same interval, and trial index.
- Unit inclusion: default-QC single units with nonzero trial-to-trial variance.
- Prediction: ridge regression with alpha selected from the training data.
- Evaluation: five stimulus-balanced trial folds and two neuron samples during exploratory runs.
- Temporal subdivision: not used in the primary analysis.

Thirty neurons was chosen because performance largely plateaued around 25--30 neurons, it matches a collaborator's analysis, and it retains more cross-session coverage than using nearly every available CP unit.

## Population-size sensitivity in session 743199

The neuron subsets were nested within each random sample so that larger populations contained the corresponding smaller populations.

| Neurons per area | MOs to CP ridge R2 |
|---:|---:|
| 5 | -0.009 |
| 10 | 0.016 |
| 15 | 0.034 |
| 20 | 0.028 |
| 25 | 0.049 |
| 30 | 0.048 |
| 35 | 0.046 |
| 40 | 0.048 |
| 45 | 0.046 |
| 50 | 0.051 |

Very small populations performed poorly. Performance reached an approximate plateau around 25 neurons. The nominal maximum at 50 neurons was only slightly higher than the 25--45 neuron range and leaves almost no CP population-resampling flexibility.

An independent 30-neuron draw produced MOs-to-CP R2 of approximately 0.055, illustrating the remaining neuron-sampling variability.

## Robustness of quiescent MOs--CP prediction across sessions

Three sessions contain adequate MOs and CP populations for a fixed 30-by-30 comparison.

| Session | MOs to CP R2 | CP to MOs R2 |
|---|---:|---:|
| 713655_2024-08-09 | 0.0097 | 0.0054 |
| 743199_2024-12-05 | 0.0548 | 0.0555 |
| 759434_2025-02-04 | 0.0337 | 0.0333 |
| Cross-session mean | 0.0327 | 0.0314 |
| Worst session | 0.0097 | 0.0054 |

The MOs-to-CP score was positive in all three sessions, although its magnitude varied substantially. Directionality was weak: CP-to-MOs performance was nearly equal in two sessions. The evidence therefore supports a shared MOs--CP residual subspace more strongly than a uniquely directional pathway.

## Condition-preserving permutation null

The primary null shuffled CP trial identities within each stimulus-by-context condition and separately inside the training and test partitions. This preserves condition composition, marginal activity statistics, folds, and nuisance structure while destroying simultaneous MOs--CP trial correspondence. Alpha selection and model fitting were repeated for every permutation.

| Session | Observed MOs to CP R2 | Null mean R2 | Null-corrected R2 | Permutation p |
|---|---:|---:|---:|---:|
| 743199_2024-12-05 | 0.0548 | -0.0090 | 0.0638 | 0.0099 |
| 759434_2025-02-04 | 0.0337 | -0.0081 | 0.0418 | 0.0099 |

Neither observed score was reached by any of the 100 permutations. A p-value of 1/101, or approximately 0.0099, is the smallest possible value with 100 permutations. These results provide evidence that the positive scores reflect trial-specific simultaneous correspondence rather than stimulus/context composition alone.

## Joint versus additive task regression

In session 743199, regressing a joint eight-level `stimulus_context` variable was compared with separate additive `stim_name` and `rewarded_modality` variables.

| Encoding | MOs to CP R2 | CP to MOs R2 |
|---|---:|---:|
| Joint `stimulus_context` | 0.03617 | 0.01960 |
| Separate additive variables | 0.03580 | 0.01962 |

The joint model removed slightly more neural variance but had essentially no effect on cross-area prediction. The stricter joint encoding remains the selected choice because it also removes stimulus-by-context interaction means.

## Time-window comparisons

With 30 neurons, pooled contexts, and joint stimulus-context regression:

### MOs to CP across sessions

| Session | Full quiescent period | 0--0.5 s post-stimulus | 0.5--1.0 s post-stimulus |
|---|---:|---:|---:|
| 713655 | 0.0097 | -0.0089 | -0.0092 |
| 743199 | 0.0548 | 0.0237 | 0.0311 |
| 759434 | 0.0337 | 0.0079 | 0.0099 |
| Mean | 0.0327 | 0.0076 | 0.0106 |

The complete quiescent period was the strongest and only window with positive MOs-to-CP prediction in every session. Both post-stimulus windows failed to generalize to session 713655.

The quiescent-period nuisance model may ultimately omit the upcoming stimulus identity because that stimulus has not yet occurred. A scientifically motivated quiescent model would retain context, running, trial index, and relevant trial-history variables. Upcoming-stimulus regression can remain a conservative control and a check for sequence structure or leakage.

## Temporal bin analysis

Dividing each trial into 100 ms bins did not improve held-out prediction. All bins from a trial were kept in the same CV fold to prevent leakage.

### Quiescent period, session 743199

| Representation | MOs to CP R2 | CP to MOs R2 |
|---|---:|---:|
| One mean over the full quiescent period | 0.0362 | 0.0196 |
| Fourteen 100 ms bins | 0.0061 | 0.0040 |

### First 500 ms after stimulus onset, session 743199

| Representation | MOs to CP R2 | CP to MOs R2 |
|---|---:|---:|
| One 500 ms mean | 0.0101 | 0.0073 |
| Five 100 ms bins | 0.0044 | 0.0037 |

For binned analyses, condition-specific means were removed separately at each time bin, analogous to PSTH subtraction. The results suggest that the useful MOs--CP relationship is dominated by slower trial-scale fluctuations. The primary analysis therefore uses one value per neuron per trial.

## Within-context and across-context generalization

Quiescent-period MOs-to-CP mappings were trained and tested within or across task contexts after pooled joint stimulus-context residualization. Physical stimulus identities were balanced between contexts.

| Session | Visual to visual | Visual to auditory | Auditory to auditory | Auditory to visual |
|---|---:|---:|---:|---:|
| 713655 | -0.012 | -0.012 | -0.002 | -0.021 |
| 743199 | 0.003 | -0.005 | 0.077 | -0.050 |
| 759434 | 0.004 | -0.005 | 0.041 | -0.019 |
| Mean | -0.005 | -0.008 | 0.039 | -0.030 |

The positive within-context effect was concentrated in the auditory context in sessions 743199 and 759434. Auditory-context mappings failed when applied to visual-context trials. Visual-context mappings contained little predictable signal and did not transfer.

The current evidence suggests context-specific MOs--CP predictive structure, but negative transfer R2 alone does not demonstrate a rotated low-dimensional subspace. Changes in covariance strength or scale can also impair transfer.

## Shared versus context-specific mapping trial

In session 743199, one mapping shared across both contexts was compared directly with separate visual and auditory mappings using identical samples and held-out trials.

| Model | Held-out R2 |
|---|---:|
| One shared mapping | 0.0454 |
| Separate context-specific mappings | 0.0548 |
| Context-specific minus shared | 0.0094 |

The context-specific model performed better in 9 of 10 neuron-sample/fold comparisons despite each context-specific mapping receiving fewer training trials. This supports context dependence in this example, largely because the auditory mapping was useful and inappropriate for visual trials.

This result should next be repeated across sessions and tested against a context-interaction or block-aware permutation null. Direct RRR subspace comparisons should estimate principal angles between independently fitted context-specific axes and compare cross-context alignment with within-context split-half reliability.

## Task-inclusive incremental prediction

An alternative analysis retained task signal while separating its contribution from the source-area contribution. On each training fold:

1. A task model predicted CP activity from `stimulus_context`, running speed, and trial index.
2. Ridge regression predicted the remaining CP training activity from MOs.
3. Task and MOs predictions were added on held-out trials.

All scaling, nuisance fitting, residual construction, alpha selection, and prediction were fold-local.

### Session 743199, 30 neurons per area

| Interval | Task-only R2 | Task plus MOs R2 | Incremental MOs delta R2 |
|---|---:|---:|---:|
| Full quiescent period | 0.0623 | 0.0937 | 0.0314 +/- 0.0059 |
| 0--0.5 s post-stimulus | 0.0759 | 0.0847 | 0.0089 +/- 0.0022 |

The quiescent incremental contribution was positive in all 10 neuron-sample/fold combinations; the post-stimulus contribution was positive in 9 of 10. This formulation allows task-aligned and trial-specific cross-area structure to be reported together without calling task-only prediction communication.

## MOs--PL search

A ridge-only screen examined MOs and PL in both directions across three sessions, three windows, pooled or individual contexts, and with or without joint stimulus-context regression.

Without stimulus-context regression, pooled post-stimulus prediction was robustly positive:

| Direction | Mean R2 across sessions | Worst-session R2 |
|---|---:|---:|
| PL to MOs, 0--0.5 s | 0.0745 | 0.0427 |
| MOs to PL, 0--0.5 s | 0.0815 | 0.0364 |

After joint stimulus-context regression, the best MOs--PL cross-session mean was approximately 0.006 and at least one session was below zero. The strong raw MOs--PL result therefore primarily reflects shared task-evoked condition structure and is not used as the primary residual communication result.

## Interpretation and remaining limitations

The most defensible current result is that MOs and CP share modest but above-null trial-specific activity during the full quiescent period. The effect is positive across three sessions and exceeds a condition-preserving permutation null in two tested sessions.

Important limitations remain:

- Only two neuron samples were used in most exploratory analyses.
- Neuron samples overlap substantially when an area has only slightly more than 30 eligible units.
- The three sessions, rather than folds, are the relevant units for population-level inference.
- Ridge prediction does not by itself establish a low-dimensional communication subspace.
- Directionality is weak because MOs-to-CP and CP-to-MOs scores are similar.
- Context-specific results may be confounded with blockwise drift, arousal, movement, or trial history.
- Dataset-wide residualization is transductive. Strict end-to-end claims require fold-local nuisance fitting.
- Searching many configurations makes nominal maxima exploratory; final hypotheses and statistics should be fixed before confirmatory testing.

## Recommended next steps

1. Fix the primary MOs-to-CP configuration at 30 neurons, one full-quiescent-period mean, and joint stimulus-context regression as a conservative control.
2. Increase neuron resampling and summarize uncertainty first within session and then across sessions.
3. Repeat the condition-preserving permutation null in session 713655 and use a session-level combined statistic.
4. Compare shared and context-specific mappings across all three sessions using paired folds.
5. Add block-aware and trial-history controls, especially previous stimulus, choice, reward, and context-switch position.
6. Fit regularized RRR only after the ridge configuration is fixed. Compare the low-rank curve with the full-rank ridge ceiling without selecting rank on test data.
7. Estimate context-specific RRR axes and compare their principal angles against within-context split-half reliability.
8. For stimulus-period analyses, report task-only R2, task-plus-source R2, and incremental source delta R2 alongside the residual-only analysis.

## Generated analysis files

Key scripts:

- `code/archive/scripts/check_quiescent_mos_cp.py`
- `code/archive/scripts/check_quiescent_mos_cp_30.py`
- `code/archive/scripts/check_quiescent_mos_cp_population_sweep.py`
- `code/archive/scripts/check_quiescent_100ms_mos_cp.py`
- `code/archive/scripts/check_poststim_500ms_binning_mos_cp.py`
- `code/archive/scripts/check_quiescent_cross_context_mos_cp_30.py`
- `code/archive/scripts/check_shared_vs_context_specific_mos_cp_30.py`
- `code/archive/scripts/check_quiescent_mos_cp_30_permutation.py`
- `code/archive/scripts/check_incremental_task_plus_mos_to_cp.py`
- `code/archive/scripts/search_cross_area.py`
- `code/archive/scripts/search_poststim_mos_cp_30.py`

Key result tables:

- `code/results/legacy/results_quiescent_mos_cp_30neurons_by_session.csv`
- `code/results/legacy/results_quiescent_mos_cp_population_sweep_743199.csv`
- `code/results/legacy/results_quiescent_mos_cp_encoding_comparison_743199.csv`
- `code/results/legacy/results_quiescent_100ms_mos_cp_743199.csv`
- `code/results/legacy/results_poststim_500ms_binning_comparison_743199.csv`
- `code/results/legacy/results_shared_vs_context_specific_mos_cp_743199.csv`
- `code/results/legacy/results_incremental_task_plus_mos_to_cp_743199.csv`
- `code/results/legacy/results_cross_area_summary.csv`



## Expanded quiescent-period candidate screen

This later exploratory screen supersedes the earlier MOs--CP-only candidate prioritization above. It compared repeatable frontal--motor, motor--motor, frontal--frontal, and motor--striatal pairs.

The fixed screening configuration was:

- One mean firing rate per neuron over the complete quiescent period.
- Joint `stimulus_context`, interval-matched running speed, and trial-index regression.
- Pooled visual- and auditory-context trials.
- Thirty neurons per area.
- Two random neuron samples per session.
- Five stimulus-by-context-stratified folds per sample.
- Full-rank ridge with alpha selected from the training trials.

Thus, every session and direction is summarized from 10 fold-level scores, but only two neuron samples. Folds and neuron samples are not independent animals; the session/mouse remains the unit of cross-animal replication.

### Candidate ranking

| Priority | Pair | Sessions | Mean R2 by direction | Worst-mouse R2 | Interpretation |
|---:|---|---:|---:|---:|---|
| 1 | MOs--PL | 3 | 0.0428--0.0443 | 0.0189--0.0202 | Best balance of frontal--motor relevance, coverage, and stability |
| 2 | MOs--MOp | 4 | 0.0561--0.0621 | 0.0120--0.0170 | Broadest coverage and largest mean, but highly heterogeneous |
| 3 | ACAd--MOs | 2 | 0.0341--0.0348 | 0.0323--0.0328 | Highly consistent frontal--motor result in two mice |
| 4 | PL--ACAd | 2 | 0.0379--0.0401 | 0.0339--0.0370 | Highly consistent frontal--frontal comparison |
| 5 | MOs--CP | 3 | 0.0265--0.0293 | 0.0054--0.0097 | Replicated motor--striatal prediction with weaker worst-mouse performance |

### MOs--PL: primary frontal--motor candidate

| Session | MOs to PL R2 | PL to MOs R2 |
|---|---:|---:|
| 664851_2023-11-16 | 0.0533 | 0.0387 |
| 714748_2024-06-24 | 0.0550 | 0.0752 |
| 743199_2024-12-05 | 0.0202 | 0.0189 |
| Cross-session mean | 0.0428 | 0.0443 |

MOs--PL is positive in all three mice and is not dominated by a single extreme session. Its frontal--motor interpretation and three-session coverage make it the recommended primary quiescent candidate.

### MOs--MOp: broadest replication

| Session | MOs to MOp R2 | MOp to MOs R2 |
|---|---:|---:|
| 667252_2023-09-28 | 0.1526 | 0.1689 |
| 708016_2024-04-29 | 0.0120 | 0.0170 |
| 713655_2024-08-09 | 0.0282 | 0.0225 |
| 741137_2024-10-10 | 0.0316 | 0.0400 |
| Cross-session mean | 0.0561 | 0.0621 |

MOs--MOp is positive in all four mice, but session 667252 is much stronger than the other three. The result is robust in sign but heterogeneous in magnitude and may be particularly sensitive to shared motor preparation or unmeasured movement.

### ACAd--MOs: most consistent frontal--motor candidate

| Session | MOs to ACAd R2 | ACAd to MOs R2 |
|---|---:|---:|
| 664851_2023-11-16 | 0.0323 | 0.0328 |
| 743199_2024-12-05 | 0.0374 | 0.0354 |
| Cross-session mean | 0.0348 | 0.0341 |

This pair is remarkably consistent, but it is represented by only two mice.

### PL--ACAd: consistent frontal--frontal comparison

| Session | PL to ACAd R2 | ACAd to PL R2 |
|---|---:|---:|
| 664851_2023-11-16 | 0.0339 | 0.0370 |
| 743199_2024-12-05 | 0.0419 | 0.0431 |
| Cross-session mean | 0.0379 | 0.0401 |

PL--ACAd is less directly tied to motor output than pairs involving MOp or CP and therefore provides a useful comparison for shared frontal state. Like ACAd--MOs, it is available in only two mice at 30 neurons per area.

### Updated candidate hierarchy

1. MOs--PL as the primary frontal--motor analysis.
2. MOs--MOp as a motor--motor replication analysis with strong movement controls.
3. ACAd--MOs as a focused, consistent two-mouse frontal--motor result.
4. PL--ACAd as a less movement-linked frontal--frontal comparison.
5. MOs--CP as the motor--striatal comparison.

These ridge scores establish predictive ceilings and candidate robustness, not low-dimensional communication subspaces. Each selected pair still requires RRR rank paths, condition-preserving permutation nulls, and substantially more neuron resampling.

Associated files:

- `code/archive/scripts/search_residual_area_pairs_epochs.py`
- `code/results/legacy/results_residual_area_pairs_epochs_summary.csv`
- `code/results/legacy/results_residual_area_pairs_epochs_by_session.csv`


## Expanded stimulus-epoch candidate screen

The same cross-session screen also evaluated two stimulus-aligned windows: 0--0.5 s and 0.5--1.0 s after stimulus onset. The configuration matched the expanded quiescent screen: one mean firing rate per neuron per window, pooled contexts, joint `stimulus_context` and trial-index regression, 30 neurons per area, two neuron samples per session, five stratified folds per sample, and full-rank ridge regression. Interval-matched running speed was included in this screen, together with trial index and joint stimulus context. Other movement variables were not included, so explicit video-derived movement controls remain necessary.

### Cross-session summary

| Pair and direction | Sessions | 0--0.5 s mean R2 | 0--0.5 s worst mouse | 0.5--1.0 s mean R2 | 0.5--1.0 s worst mouse |
|---|---:|---:|---:|---:|---:|
| ACAd to PL | 2 | 0.0197 | 0.0057 | 0.0151 | 0.0108 |
| PL to ACAd | 2 | 0.0186 | 0.0052 | 0.0183 | 0.0125 |
| MOs to ACAd | 2 | 0.0097 | 0.0092 | 0.0116 | 0.0050 |
| ACAd to MOs | 2 | 0.0086 | 0.0079 | 0.0104 | 0.0060 |
| MOs to PL | 3 | 0.0056 | -0.0010 | 0.0049 | -0.0017 |
| PL to MOs | 3 | 0.0056 | 0.0003 | 0.0052 | 0.0015 |
| MOs to MOp | 4 | 0.0133 | -0.0071 | 0.0144 | -0.0088 |
| MOp to MOs | 4 | 0.0144 | -0.0074 | 0.0165 | -0.0079 |
| MOs to CP | 3 | 0.0047 | -0.0089 | 0.0120 | -0.0092 |
| CP to MOs | 3 | 0.0045 | -0.0090 | 0.0141 | -0.0079 |

PL--ACAd is the strongest repeatable residual stimulus-epoch candidate in this limited screen. Both directions were positive in both mice and remained positive in both time windows. MOs--ACAd was also consistently positive in the two available mice, but its R2 was only approximately 0.005--0.015. These two pairs merit follow-up, but two mice are not enough to call either result broadly robust.

MOs--PL was close to zero after joint stimulus-context regression, despite strong raw prediction when task signal was retained. MOs--MOp had the largest cross-session mean among the motor pairs, but this was driven by session 667252 and two sessions were near zero or negative. MOs--CP was positive in sessions 743199 and 759434 but negative in session 713655. Thus, no tested motor or motor--striatal pair currently shows convincing residual stimulus-epoch prediction across every available mouse.

### Per-session values for the leading and focal pairs

| Pair/direction | Session | 0--0.5 s R2 | 0.5--1.0 s R2 |
|---|---|---:|---:|
| ACAd to PL | 664851 | 0.0336 | 0.0194 |
| ACAd to PL | 743199 | 0.0057 | 0.0108 |
| PL to ACAd | 664851 | 0.0320 | 0.0240 |
| PL to ACAd | 743199 | 0.0052 | 0.0125 |
| MOs to ACAd | 664851 | 0.0101 | 0.0181 |
| MOs to ACAd | 743199 | 0.0092 | 0.0050 |
| ACAd to MOs | 664851 | 0.0093 | 0.0147 |
| ACAd to MOs | 743199 | 0.0079 | 0.0060 |
| MOs to CP | 713655 | -0.0089 | -0.0092 |
| MOs to CP | 743199 | 0.0151 | 0.0353 |
| MOs to CP | 759434 | 0.0079 | 0.0099 |
| CP to MOs | 713655 | -0.0090 | -0.0079 |
| CP to MOs | 743199 | 0.0177 | 0.0421 |
| CP to MOs | 759434 | 0.0049 | 0.0080 |

### Comparison with the quiescent period

Across every candidate pair, the full-quiescent mean was stronger than the corresponding residual stimulus-period result. For example, quiescent MOs--PL averaged approximately 0.043--0.044 across three mice, whereas both post-stimulus windows averaged only approximately 0.005--0.006. Quiescent MOs--CP averaged approximately 0.027--0.029, whereas post-stimulus averages ranged from approximately 0.005 to 0.014 and included a negative mouse. MOs--MOp showed the same pattern, with quiescent means of approximately 0.056--0.062 versus post-stimulus means of approximately 0.013--0.017.

The current interpretation is therefore not that shared activity disappears during stimulation, but that most of the strong stimulus-period prediction is task aligned and is removed by joint `stimulus_context` regression. The remaining trial-specific component is small and mouse dependent. A stimulus-epoch follow-up should prioritize PL--ACAd and MOs--ACAd, add running and other movement covariates, use fold-local nuisance regression, and test condition-preserving permutation nulls before fitting RRR communication-subspace rank paths.


## Semedo-style MOs--MOp dimensionality pilot

A focused pilot tested whether the strong quiescent MOs--MOp prediction in session 667252 had the geometry reported by Semedo et al.: a low-dimensional predictive subspace that is more efficient than the source area's dominant variance dimensions, and potentially lower-dimensional than within-area prediction.

The comparison used the full quiescent interval, pooled contexts, and fold-local regression of joint `stimulus_context`, interval running speed, and trial index. Session 667252 contained 86 eligible MOs and 53 eligible MOp units. To permit non-overlapping within-area controls, all mappings used matched 25-neuron populations. Ten neuron resamples, five outer test folds, and four inner rank-selection folds were used. Rank was selected as the smallest rank within one standard error of the best inner-CV performance.

The matched mappings were MOs-A to MOp-A, MOp-A to MOs-A, MOs-A to MOs-B, and MOp-A to MOp-B. PCA and all preprocessing were fitted using training data only.

| Mapping | Median predictive rank | Mean selected-rank test R2 | PCA dimensions needed to match, median |
|---|---:|---:|---:|
| MOs to MOp | 5.5 | 0.1105 | 18 |
| MOp to MOs | 6 | 0.1283 | 22 |
| MOs to MOs | 6 | 0.1742 | 18 |
| MOp to MOp | 7 | 0.1315 | 20 |

For MOs to MOp, predictive RRR reached mean held-out R2 values of 0.032, 0.065, 0.089, and 0.104 at ranks 1--4, compared with 0.012, 0.039, 0.057, and 0.070 for the same number of dominant PCA dimensions. At each fold's selected predictive rank, RRR exceeded the same-dimensional PCA model by mean R2 = 0.0302 and did so in 49 of 50 sample-fold evaluations. Aggregating folds within neuron sample, the bootstrap 95% interval for this advantage was 0.0238--0.0366.

For MOp to MOs, the corresponding selected-rank advantage was mean R2 = 0.0340, positive in all 50 evaluations, with a neuron-sample bootstrap interval of 0.0265--0.0414. Thus, the pilot gives strong evidence for a Semedo Fig. 4-like result: cross-area predictive dimensions capture substantially more prediction than an equal number of dominant source-area dimensions.

Evidence for the stronger Fig. 7-like claim is currently weak. MOs-to-MOp rank averaged only 0.42 dimensions below within-MOs rank; the neuron-sample bootstrap interval for the paired difference was -1.40 to 0.60. MOp-to-MOs averaged 0.68 dimensions below within-MOp, with interval -1.58 to 0.14. Both estimates point toward a smaller between-area rank, but include no difference. Therefore this one-session pilot supports predictive axes being distinct from dominant axes, but does not yet establish that the communication subspace is lower-dimensional than within-area shared variability.

Associated files:

- `code/analyses/compare_communication_dimensions.py`
- `code/analyses/plot_communication_dimensions.py`
- `code/results/exploratory/2026-09-03_mos_mop_dimensions_667252/dimension_curves.csv`
- `code/results/exploratory/2026-09-03_mos_mop_dimensions_667252/selected_dimensions.csv`
- `code/results/exploratory/2026-09-03_mos_mop_dimensions_667252/paired_sample_summary.csv`
- `code/results/exploratory/2026-09-03_mos_mop_dimensions_667252/dimension_comparison.png`


## MOs--MOp cross-context generalization pilot (historical joint-label residualization)

Session 667252 was tested during the full quiescent interval using 30 neurons per area, 10 neuron samples, five folds, and physical-stimulus-balanced visual and auditory contexts. Joint stimulus-context, interval running speed, and trial index were regressed using the pooled session before the context split, matching the earlier exploratory cross-context convention.

| Direction | Training context | Within-context R2 | Cross-context R2 | Fraction retained |
|---|---|---:|---:|---:|
| MOs to MOp | visual | 0.1308 | 0.0421 | 32.2% |
| MOs to MOp | auditory | 0.1381 | 0.0491 | 35.6% |
| MOp to MOs | visual | 0.1806 | 0.0503 | 27.9% |
| MOp to MOs | auditory | 0.1494 | 0.0551 | 36.9% |

Neuron-sample bootstrap intervals for cross-context R2 excluded zero in every comparison: 0.0298--0.0546 and 0.0382--0.0605 for MOs to MOp, and 0.0355--0.0645 and 0.0399--0.0699 for MOp to MOs. Cross-context fold scores were positive in 84--96% of evaluations.

However, transfer losses were large and their neuron-sample intervals excluded zero. Cross-context mappings retained only approximately 28--37% of their corresponding within-context performance. Thus, the MOs--MOp relationship contains a shared cross-context component, but most of its predictive strength is context dependent. This conclusion concerns full-rank ridge mappings; it does not yet show whether the low-dimensional RRR axes themselves are geometrically aligned across contexts.

Associated files:

- `code/analyses/compare_context_generalization.py`
- `code/results/exploratory/2026-09-03_mos_mop_cross_context_667252/fold_scores.csv`
- `code/results/exploratory/2026-09-03_mos_mop_cross_context_667252/transfer_summary.csv`
- `code/results/exploratory/2026-09-03_mos_mop_cross_context_667252/sample_bootstrap_summary.csv`

### Quiescent-primary rerun

The analysis was rerun with the current primary convention: 25 neurons per
area, 10 neuron samples, five balanced folds, and context-only
(`rewarded_modality`) nuisance regression. The joint `stimulus_context` label
was used only for balancing.

| Direction | Training context | Within-context R2 | Cross-context R2 | Fraction retained |
|---|---|---:|---:|---:|
| MOs to MOp | visual | 0.1109 | 0.0274 | 24.7% |
| MOs to MOp | auditory | 0.1305 | 0.0259 | 19.9% |
| MOp to MOs | visual | 0.1452 | 0.0391 | 26.9% |
| MOp to MOs | auditory | 0.1334 | 0.0411 | 30.8% |

All four cross-context means had neuron-sample bootstrap intervals above zero.
MOp-to-MOs transfer was positive in all 10 neuron samples in both directions:
0.0213--0.0585 for visual-to-auditory and 0.0287--0.0534 for
auditory-to-visual. MOs-to-MOp was positive in all samples for
visual-to-auditory and in 9 of 10 for auditory-to-visual; the corresponding
intervals were 0.0154--0.0402 and 0.0117--0.0388.

The updated conclusion is that both mappings contain a repeatable shared
cross-context component, with stronger absolute transfer for MOp to MOs.
Nevertheless, only about 20--31% of within-context performance is retained, so
most predictive strength remains context dependent.

Associated results:

- `code/results/exploratory/2026-09-04_quiescent_primary_mos_mop_cross_context_667252/`


## Semedo-style MOs--PL dimensionality pilot

A second case study applied the matched dimensionality analysis to session 714748. The session contained 120 eligible MOs and 53 eligible PL units. As in the MOs--MOp pilot, the analysis used the full quiescent interval, 25 neurons per non-overlapping population, 10 neuron resamples, five outer folds, four inner folds, pooled contexts, and fold-local regression of joint stimulus context, running speed, and trial index.

| Mapping | Mean predictive rank | Median rank | Selected-rank test R2 | Same-rank PCA R2 |
|---|---:|---:|---:|---:|
| MOs to PL | 4.34 | 4 | 0.0349 | 0.0291 |
| PL to MOs | 3.52 | 3 | 0.0576 | 0.0253 |
| MOs to MOs | 6.00 | 6 | 0.1102 | 0.0910 |
| PL to PL | 8.66 | 8 | 0.0816 | 0.0623 |

This case shows a clearer Fig. 7-like rank separation than session 667252. MOs-to-PL rank averaged 1.66 dimensions below within-MOs rank, with a neuron-sample bootstrap interval of -2.60 to -0.74. PL-to-MOs rank averaged 5.14 dimensions below within-PL rank, with interval -6.18 to -4.12.

Predictive dimensions also outperformed equal-dimensional dominant PCA axes. The MOs-to-PL advantage was mean R2 = 0.0058, with bootstrap interval 0.0033--0.0083, and was positive in 43 of 50 sample-fold evaluations. The PL-to-MOs advantage was mean R2 = 0.0323, with interval 0.0243--0.0402, and was positive in all 50 evaluations.

This is a strong exploratory case-study result, especially for PL-to-MOs. Its main limitation is neuron-resampling independence: PL has only 53 eligible neurons, while each sample uses 50 across its two non-overlapping halves. Different PL resamples therefore overlap heavily, making the bootstrap interval optimistic. Replication in another session remains necessary for a population-level claim.

Associated files:

- `code/results/exploratory/2026-09-03_mos_pl_dimensions_714748/summary.csv`
- `code/results/exploratory/2026-09-03_mos_pl_dimensions_714748/paired_sample_summary.csv`
- `code/results/exploratory/2026-09-03_mos_pl_dimensions_714748/dimension_comparison.png`


## Ridge intercept convention and sensitivity check

The maintained residual-prediction pipelines use `fit_intercept=False` consistently for ridge penalty selection, final ridge fitting, and RRR. Neural activity is nuisance-residualized and z-scored over the analyzed trial population before prediction. This matches the no-intercept formulation in Semedo et al. and avoids selecting a penalty for a different model class than the final fit.

A paired sensitivity check used session 667252, the full quiescent interval, MOs--MOp in both directions, 30 neurons per area, 10 neuron samples, and five identical folds. The only changed setting was whether both `RidgeCV` and `Ridge` fitted an intercept.

| Direction | No-intercept R2 | Intercept R2 | No-intercept minus intercept |
|---|---:|---:|---:|
| MOs to MOp | 0.1500 | 0.1444 | 0.0056 |
| MOp to MOs | 0.1732 | 0.1681 | 0.0051 |

The no-intercept model performed better in all paired fold/sample evaluations. Neuron-sample bootstrap intervals for the improvement were 0.0053--0.0059 for MOs to MOp and 0.0047--0.0055 for MOp to MOs. Thus, the selected convention does not impair this example's predictive R2 and modestly improves it.

Associated files:

- `code/analyses/check_intercept_sensitivity.py`
- `code/results/exploratory/2026-09-03_intercept_sensitivity_667252/fold_scores.csv`
- `code/results/exploratory/2026-09-03_intercept_sensitivity_667252/summary.csv`


## 2026-09-04 preprocessing cleanup and dimensionality reruns

The maintained pipeline now has one explicit convention throughout:

1. For each area's complete analyzed trial population, z-score activity.
2. Globally regress `stimulus_context`, interval mean running speed, and trial
   index from that activity.
3. Globally z-score the residuals.
4. Cross-validate only the neural prediction model, using
   `fit_intercept=False` consistently in `RidgeCV`, `Ridge`, and RRR.

`src.residualization.residualize_population` is the single shared entry point
for steps 1--3. `compare_communication_dimensions.py`,
`compare_context_generalization.py`, and `screen_area_pairs.py` now use it.
The lower-level `fit_covariate_model` and `residualize_activity` functions
remain available because they implement that wrapper and support diagnostics;
they are not competing preprocessing conventions.

The two dimensionality case studies were regenerated with the same session,
epoch, neuron counts, folds, random seed, and 10 neuron samples as before. The
following tables supersede the earlier fold-local numerical tables above.

### Session 667252: MOs--MOp, global preprocessing

| Mapping | Mean predictive rank | Median rank | PCA dimensions to match, mean | Selected-rank test R2 | Same-rank PCA R2 |
|---|---:|---:|---:|---:|---:|
| MOs to MOp | 6.04 | 6 | 17.44 | 0.1192 | 0.0864 |
| MOp to MOs | 6.48 | 6 | 19.96 | 0.1365 | 0.1007 |
| MOs to MOs | 6.06 | 6 | 17.37 | 0.1845 | 0.1419 |
| MOp to MOp | 7.22 | 7 | 19.53 | 0.1397 | 0.1125 |

The central conclusion is unchanged: predictive RRR axes are considerably
more efficient than equal-dimensional dominant PCA axes. Between-area rank is
not clearly lower than within-area rank in this session.

### Session 714748: MOs--PL, global preprocessing

| Mapping | Mean predictive rank | Median rank | PCA dimensions to match, mean | Selected-rank test R2 | Same-rank PCA R2 |
|---|---:|---:|---:|---:|---:|
| MOs to PL | 4.14 | 4 | 8.10 | 0.0392 | 0.0336 |
| PL to MOs | 3.34 | 3 | 17.90 | 0.0627 | 0.0287 |
| MOs to MOs | 6.22 | 6 | 13.83 | 0.1152 | 0.0952 |
| PL to PL | 8.68 | 8 | 17.43 | 0.0888 | 0.0683 |

This case continues to show both predictive-axis efficiency and a lower
selected between-area dimension than the matched within-area dimension,
especially for PL to MOs. It remains an exploratory single-session case study.

Corrected result directories:

- `code/results/exploratory/2026-09-04_global_mos_mop_dimensions_667252/`
- `code/results/exploratory/2026-09-04_global_mos_pl_dimensions_714748/`

### Subsequent quiescent-primary specification

After these reruns, the primary quiescent nuisance model was refined to regress
context (`rewarded_modality`) rather than the joint `stimulus_context` label.
There is no current-trial stimulus during the quiescent interval. The joint
label remains the CV stratification variable so context-by-upcoming-stimulus
conditions are balanced without being subtracted from activity. Running speed
and trial index remain continuous nuisances. Stimulus-epoch analyses continue
to regress joint `stimulus_context`.

The two result directories above were generated with joint
`stimulus_context` residualization and are therefore historical sensitivity
results, not outputs of the new quiescent-primary specification.

The quiescent-primary reruns used 25 neurons per population, 10 neuron samples,
five outer folds, four inner folds, and the same random seed as the historical
runs. Context-only residualization changed the estimates only slightly.

| Session and mapping | Mean predictive rank | Median rank | Selected-rank test R2 | Same-rank PCA R2 |
|---|---:|---:|---:|---:|
| 667252, MOs to MOp | 5.82 | 5 | 0.1177 | 0.0842 |
| 667252, MOp to MOs | 6.40 | 6 | 0.1345 | 0.0992 |
| 667252, MOs to MOs | 5.90 | 6 | 0.1825 | 0.1393 |
| 667252, MOp to MOp | 7.50 | 7 | 0.1391 | 0.1121 |
| 714748, MOs to PL | 4.34 | 4 | 0.0390 | 0.0333 |
| 714748, PL to MOs | 3.44 | 3 | 0.0621 | 0.0277 |
| 714748, MOs to MOs | 6.18 | 6 | 0.1146 | 0.0940 |
| 714748, PL to PL | 8.72 | 8 | 0.0879 | 0.0674 |

The conclusions remain the same: both sessions show predictive axes that are
more efficient than equal-dimensional dominant axes, while the clearest
between-versus-within dimensionality separation is in session 714748,
especially PL to MOs.

Updated selection results and figures:

- `code/results/exploratory/2026-09-04_quiescent_primary_mos_mop_dimensions_667252/`
- `code/results/exploratory/2026-09-04_quiescent_primary_mos_pl_dimensions_714748/`

Final model artifacts were saved for neuron sample 0. Each artifact contains
both between-area directions and both non-overlapping within-area controls,
including the all-trial descriptive model, five fold models, exact unit IDs,
fitted preprocessing, and held-out scores. The final-model rank is the rounded
median of that sample's five selected fold ranks and is not the 10-sample rank
reported in the table above.

- `code/results/final/2026-09-04_mos_mop_667252/`
- `code/results/final/2026-09-04_mos_pl_714748/`


## MOs--PL cross-context generalization, session 714748

The quiescent-primary MOs--PL case was tested with full-rank Ridge using 25
neurons per area, 10 neuron samples, five physical-stimulus-balanced folds,
context-only nuisance regression, and `stimulus_context` balancing. This is a
population-level companion analysis using random unit samples; it is not yet a
direct test of the saved low-dimensional RRR axes.

| Direction | Training context | Within-context R2 | Cross-context R2 | Fraction retained |
|---|---|---:|---:|---:|
| MOs to PL | visual | 0.0254 | 0.0024 | 9.5% |
| MOs to PL | auditory | 0.0427 | -0.0039 | -9.2% |
| PL to MOs | visual | 0.0393 | 0.0139 | 35.4% |
| PL to MOs | auditory | 0.0644 | 0.0021 | 3.3% |

For PL to MOs, visual-to-auditory prediction was positive in 8 of 10 neuron
samples; its neuron-sample bootstrap interval was 0.0050--0.0235. In the other
direction of transfer, auditory-to-visual prediction was positive in only 6 of
10 samples and its interval included zero (-0.0049--0.0100). Transfer loss was
large: cross-context testing retained about 35% of visual-context performance
and only about 3% of auditory-context performance.

Thus there is asymmetric evidence that the PL-to-MOs mapping contains a small
cross-context component, particularly when trained in visual context, but the
overall mapping does not generalize robustly in both directions. This is
consistent with a substantially context-dependent relationship. Testing the
geometry of the selected RRR subspace across contexts remains a separate next
analysis.

Associated results:

- `code/results/exploratory/2026-09-04_quiescent_primary_mos_pl_cross_context_714748/`
