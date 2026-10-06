# Local analysis, then back to Code Ocean for all sessions

Everything needed to run ONE session on your own machine, push the work to
GitHub, and then run ALL sessions back in the capsule.

The ~2.6 GB of session data is not in this bundle. You pull it straight from
the public AIND S3 bucket in step 2.

    environment.yml              conda env matching the capsule
    .gitattributes               keeps notebook outputs out of git
    code/fetch_session.py        downloads one session from s3://aind-open-data
    code/nuisance_regression.py  helper module the notebooks import
    code/run_capsule.py          all-sessions runner (used by Reproducible Run)
    code/tools/nbstrip.py        git filter backing .gitattributes
    code/task_axis_comm_subspace.py  all-pairs task-axis vs communication-subspace analysis
    code/make_alignment_figures.py   per-session and pooled figures for it
    code/notebooks/*.ipynb       the analysis notebooks (all-pairs wrapper + residual variants)
    data/dynamicrouting_datacube/  empty - session data lands here

## 1. Environment

    conda env create -f environment.yml
    conda activate inter-area-dynamic-routing

## 2. One session (~2.6 GB, 4819 files)

    cd code
    python fetch_session.py 743199_2024-12-05 ../data/dynamicrouting_datacube
    cd ..

That is the session the notebooks default to. Any of the 12 works - pass a
different `<subject>_<date>` id. The script is resumable, so re-running skips
files that already match the remote size.

## 3. Point the notebooks at it

The notebooks take both of their inputs from the environment, so the same file
runs locally and in the capsule with no edits:

    root       = Path(os.environ.get("DATACUBE_ROOT", "<capsule data dir>"))
    session_id = os.environ.get("SESSION_ID", "743199_2024-12-05")

So:

    export DATACUBE_ROOT="$PWD/data/dynamicrouting_datacube"
    jupyter lab

Set `SESSION_ID` too if you want a session other than the default.

## 4. Run

Open `code/notebooks/contextual_vs_communication_one_session_3regressors.ipynb`
and Run All. Keep the `code/notebooks/` layout - the notebook does
`sys.path.insert(0, str(Path.cwd().parent))` so `import nuisance_regression`
resolves to `code/nuisance_regression.py`. Budget a minute for
`pynwb.read_nwb` and several more for the permutation tests; peak memory is a
few GB.

## 5. Push your changes

If you cloned the repo (recommended over using this zip), enable the notebook
filter once so your commits do not carry ~1.4 MB of embedded figures each:

    git config filter.nbstrip.clean "python code/tools/nbstrip.py"
    git config filter.nbstrip.smudge cat

Then commit and push to your branch as usual. `/data/` is gitignored, so the
session data stays local.

## 6. All sessions, back in the capsule

The branch `all-pairs-alignment` on GitHub carries this bundle (code modules,
output-stripped notebooks, `code/run`, `.gitattributes`; `code/notebooks/` is
no longer gitignored). In the capsule:

1. Open the git panel. If the capsule still has uncommitted edits from the
   earlier local-bundle work (same files as this bundle, older versions),
   discard them or commit them to a scratch branch first.
2. Pull / check out `all-pairs-alignment`.
3. Press Reproducible Run. `code/run` calls `code/run_capsule.py`, which finds
   every session in the attached datacube and, by default, runs only
   `task_axis_vs_communication_subspace_all_pairs.ipynb` once per session
   (about 6 min per session on a laptop; 12 sessions ~ 1-2 h). It writes to
   `/results/<session_id>/`: the executed notebook + HTML, `alignment_results.pkl`,
   `task_axis_by_area.csv`, `alignment_by_pair.csv` and `figures/*.svg`, and
   rebuilds `/results/pooled/` (pooled Figures 1-2 and `alignment_by_pair_pooled.csv`)
   after every session, so the pooled output always reflects all sessions so far.

Environment variables (set them in the capsule's Environment > Variables, or
export them in `code/run`):

    SESSION_IDS          comma-separated session ids (default: every session found)
    NOTEBOOKS            comma-separated notebook filenames (default: the all-pairs notebook)
    NUISANCE_REGRESSORS  default prev_stim,prev_response,prev_reward,running_speed,pupil_area;
                         `none` to skip (pupil is skipped automatically in sessions without eye tracking)
    POST_STEPS           post-processing scripts run after the analysis for the report figures
                         (default patch_cv_quality.py,patch_lobo_r2.py,alignment_controls.py,r2_dim_null.py;
                         `none` to skip). They roughly double the per-session time.
    NB_TIMEOUT           per-cell timeout in seconds, -1 for no limit (default -1)

A session that has fewer than two areas with 30 QC-pass single units is reported as
FAILED (RuntimeError in the analysis cell) and skipped by the pooled figures; the others
still run. The capsule's default resource class (2xsmall) is slow for 12 sessions; pick a
larger CPU class in the capsule's Environment panel before pressing Reproducible Run.

You can rehearse it locally too:

    DATACUBE_ROOT="$PWD/data/dynamicrouting_datacube" \
    RESULTS_DIR="$PWD/results" \
    SESSION_IDS=743199_2024-12-05 \
    NOTEBOOKS=task_axis_vs_communication_subspace_all_pairs.ipynb \
    python code/run_capsule.py

## Notes

- `aind-open-data` is a public bucket; fetch_session.py uses unsigned
  requests, so no AWS credentials are needed.
- A cell error does not abort the batch. The notebook is still saved and the
  run is reported as FAILED at the end, so you can open it and debug.
