# Analysis code organization

## Maintained notebooks

- `notebooks/check_all_mice.ipynb`: session and unit inventory.
- `notebooks/check_population_prediction.ipynb`: basic population-prediction workflow.
- `notebooks/check_population_prediction_across_contexts.ipynb`: context-generalization analyses and figures.

The other historical notebooks are preserved under
`notebooks/archive/initial_pair_search/`.

## Reusable code

- `src/data.py`: resolve and load sessions, filter trials, and select units.
- `src/activity.py`: event- and interval-aligned neural and behavioral activity extraction.
- `src/residualization.py`: covariate modeling and neural residualization.
- `src/ridge.py`: full-rank ridge scoring and held-out prediction.
- `src/rrr.py`: regularized reduced-rank regression and rank-path evaluation.
- `src/utils.py`: backward-compatible re-exports for historical notebooks; new code should use the focused modules.
- `analyses/check_intercept_sensitivity.py`: paired intercept-convention sensitivity checks.
- `analyses/compare_context_generalization.py`: balanced within- and cross-context mapping tests.
- `analyses/compare_communication_dimensions.py`: matched between- and within-area RRR/PCA dimensionality comparison.
- `analyses/plot_communication_dimensions.py`: Semedo-style dimensionality panels from saved results.
- `analyses/fit_final_communication_model.py`: save an all-trial descriptive
  RRR model, fold models, preprocessing, unit identities, and held-out scores
  for a finalized example.

## Configuration-driven screen

Run the maintained area-pair screen from the repository root:

```bash
python code/analyses/screen_area_pairs.py \
  code/configs/quiescent_screen.yaml \
  code/results/exploratory/DATE_quiescent_screen
```

Use `--session SESSION_ID` one or more times for a small trial run. Each output
directory contains the exact resolved configuration, fold scores, session and
cross-session summaries, and environment/run metadata.

The maintained analyses globally z-score neural activity, fit and remove
nuisance covariates over the complete analyzed trial population, and globally
z-score residuals before prediction CV. Prediction models then use
`fit_intercept=False` consistently. Quiescent analyses regress context
(`rewarded_modality`), running speed, and trial index while stratifying folds
by the joint `stimulus_context` label. Stimulus-epoch analyses instead regress
the joint `stimulus_context` label. These choices are independent and recorded
in each configuration. The preprocessing is implemented by
`src.residualization.residualize_population`; run metadata records it
explicitly. A fold-local preprocessing analysis would be a distinct sensitivity
analysis and should be labeled as such, not mixed into the primary results.

## Archived investigations and legacy outputs

One-off investigation scripts are retained under `archive/scripts/`. Their
CSV outputs are frozen under `results/legacy/`; they should not be
overwritten. The research summary in
`../notes/communication_subspace_findings.md` points to these legacy files.

New outputs should go under:

- `results/exploratory/` while configurations are being searched.
- `results/confirmatory/` after hypotheses and statistics are fixed.

Use a new dated output directory for each run instead of overwriting tables.
