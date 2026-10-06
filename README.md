# Inter-area dynamic routing

Does the alignment between an area's **context (task) axis** and its **communication
subspace** with another area depend on frontal cortex, or is it a general property of
inter-areal communication? This repository analyses the Allen Institute Dynamic Routing
Neuropixels sessions (the `dynamicrouting_datacube` NWB files) to answer that. It is a
Code Ocean capsule: pressing **Reproducible Run** executes `code/run` on every session in
the attached datacube.

## Analysis in one paragraph

Per session, quiescent-window firing rates of QC-pass single units are z-scored and
previous-trial stimulus / response / reward, running speed and pupil area are regressed
out (`NUISANCE_REGRESSORS`; no condition-mean subtraction). Areas with at least 30 such
units are kept and every area is subsampled to exactly 30 units, 10 times. For each area
an LDA context axis is fit with balanced block-wise cross-validated decoding accuracy (one
block of each context held out, equal priors), judged against a block-permutation null (labels
permuted across whole blocks); for each ordered pair a
reduced-rank-regression communication subspace is fit with 10-fold CV R² vs. rank and
the 1-SEM dimensionality rule. Alignment is the norm of the unit context axis projected
onto the subspace, tested against a shuffled-label axis null (an LDA axis fit on shuffled
labels, projected onto the same subspace) and FDR-corrected over pairs; the
dimensionality-matched random-axis chance is reported as a reference. Every null draw is
generated once per session and applied to all unit subsamples. Post-processing adds
held-out-block R², context decoding from the subspace, the cross-block control (axis on
three blocks, subspace on the other three) and a trial-shuffle null for the rank-d R².
Full details are in the module docstring of `code/task_axis_comm_subspace.py`.

## Layout

    code/run                              capsule entry point (Reproducible Run)
    code/run_capsule.py                   runs a notebook once per session, writes /results/<session>/
    code/notebooks/
      task_axis_vs_communication_subspace_all_pairs.ipynb   the analysis wrapper (default)
      contextual_vs_communication_one_session_residual_variants.ipynb   earlier one-session exploration
    code/task_axis_comm_subspace.py       the analysis (analyse_session)
    code/patch_cv_quality.py              held-out task-axis R², context decoding from the subspace
    code/patch_lobo_r2.py                 held-out-block R² of the rank-d subspace
    code/alignment_controls.py            cross-block control
    code/r2_dim_null.py                   trial-shuffle significance of the rank-d R²
    code/patch_fit_quality.py             back-fills fit-quality fields into older result pickles
    code/make_alignment_figures.py        per-session and pooled Figures 1-2, supplementary figures
    code/make_controls_figure.py          pooled cross-block control figure
    code/make_report_figures.py           report figures R1, R2, S1
    code/nuisance_regression.py           helper used by the residual-variants notebook
    code/fetch_session.py                 download one session from s3://aind-open-data for local work
    code/tools/nbstrip.py                 git filter that strips notebook outputs (see .gitattributes)
    environment.yml                       conda env mirroring the capsule environment
    environment/, .codeocean/             capsule environment and attached dataset

## Running in the capsule

Pull `main`, pick a compute class larger than the default 2xsmall (one session takes about
40 minutes on a laptop core, post-processing included) and press Reproducible Run.
For each session `/results/<session_id>/` receives the executed notebook and its HTML,
`alignment_results.pkl`, `task_axis_by_area.csv`, `alignment_by_pair.csv`,
`alignment_by_pair_qualified.csv`, `controls_results.pkl`, `r2_dim_significance.csv` and
`figures/*.svg`. After every session the pooled figures (`/results/pooled/`), the pooled
cross-block control and the report figures (`/results/report/figures/`) are rebuilt from
all sessions finished so far, so the last session's output covers all of them.

Environment variables (capsule Environment > Variables, or exported in `code/run`):

    SESSION_IDS          comma-separated session ids (default: every session found)
    NOTEBOOKS            comma-separated notebook filenames (default: the all-pairs notebook)
    NUISANCE_REGRESSORS  default prev_stim,prev_response,prev_reward,running_speed,pupil_area;
                         `none` to skip (pupil is skipped automatically without eye tracking)
    POST_STEPS           post-processing scripts after the analysis (default
                         patch_cv_quality.py,patch_lobo_r2.py,alignment_controls.py,r2_dim_null.py;
                         `none` to skip)
    NB_TIMEOUT           per-cell timeout in seconds, -1 for no limit (default -1)

A session with fewer than two areas of 30 units is reported as FAILED and skipped by the
pooled figures; the other sessions still run.

## Running locally

    conda env create -f environment.yml && conda activate inter-area-dynamic-routing
    python code/fetch_session.py 743199_2024-12-05 data/dynamicrouting_datacube   # ~2.6 GB, resumable
    git config filter.nbstrip.clean "python code/tools/nbstrip.py"                  # once per clone
    git config filter.nbstrip.smudge cat

    DATACUBE_ROOT="$PWD/data/dynamicrouting_datacube" RESULTS_DIR="$PWD/results" \
    SESSION_IDS=743199_2024-12-05 python code/run_capsule.py

`data/` and `results/` are gitignored. The notebooks read `DATACUBE_ROOT`, `SESSION_ID`
and `RESULTS_DIR` from the environment, so the same files run locally and in the capsule.
