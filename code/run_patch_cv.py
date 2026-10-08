#!/usr/bin/env python
"""Capsule entry point for a post-processing-only run: start from the alignment results of a previous
Reproducible Run (attached as a data asset) instead of recomputing the 14-hour analysis.

For every <session>/alignment_results.pkl found under RESULTS_SRC, the session folder is copied to
RESULTS_DIR/<session>/ and the scripts in PATCH_STEPS (default patch_cv_quality.py) are run on it; then the
pooled, controls and report figures are rebuilt from all sessions.

    RESULTS_SRC   folder holding the previous run's results (default: first /data/*/ that contains
                  <session>/alignment_results.pkl files)
    RESULTS_DIR   output (default /root/capsule/results)
    PATCH_STEPS   comma-separated scripts in code/ to run per session (default patch_cv_quality.py)
    SESSION_IDS   optional subset
"""
import os, pickle, shutil, subprocess, sys, time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))


def find_source():
    src = os.environ.get('RESULTS_SRC')
    if src:
        return Path(src)
    for d in sorted(Path('/data').glob('*')):
        if any(d.glob('*/alignment_results.pkl')):
            return d
    sys.exit('no previous results found: set RESULTS_SRC or attach a results data asset')


def main():
    src = find_source()
    out = Path(os.environ.get('RESULTS_DIR', '/root/capsule/results'))
    steps = [s.strip() for s in os.environ.get('PATCH_STEPS', 'patch_cv_quality.py').split(',') if s.strip()]
    wanted = {s.strip() for s in os.environ.get('SESSION_IDS', '').split(',') if s.strip()}
    sessions = sorted(p.parent.name for p in src.glob('*/alignment_results.pkl'))
    sessions = [s for s in sessions if not wanted or s in wanted]
    print(f'source  : {src}\nresults : {out}\nsteps   : {steps}\n{len(sessions)} session(s)\n', flush=True)
    failures = []
    for sid in sessions:
        t0 = time.time()
        dst = out / sid
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src / sid, dst)
        with open(dst / 'alignment_results.pkl', 'rb') as f:
            regressors = pickle.load(f).get('nuisance_regressors') or []
        env = dict(os.environ, RESULTS_DIR=str(out), SESSION_ID=sid, NUISANCE_REGRESSORS=','.join(regressors) or 'none')
        ok = True
        for script in steps:
            args = [] if script == 'alignment_controls.py' else [sid]
            r = subprocess.run([sys.executable, str(CODE_DIR / script), *args], env=env)
            ok = ok and r.returncode == 0
        print(f"=== {'ok' if ok else 'FAILED'} {sid} ({(time.time() - t0) / 60:.1f} min) ===\n", flush=True)
        if not ok:
            failures.append(sid)

    # pooled, controls and report figures from every session with results
    import make_alignment_figures as maf, make_controls_figure as mcf, make_report_figures as mrf
    results = [pickle.load(open(p, 'rb')) for p in sorted(out.glob('*/alignment_results.pkl'))]
    pooled_dir = out / 'pooled'; pooled_dir.mkdir(parents=True, exist_ok=True)
    if len(results) > 1:
        maf.pooled_figures(results, pooled_dir)
    controls = [pickle.load(open(p, 'rb')) for p in sorted(out.glob('*/controls_results.pkl'))]
    if controls:
        mcf.figure_controls(controls, pooled_dir)
    report_dir = out / 'report' / 'figures'; report_dir.mkdir(parents=True, exist_ok=True)
    maf.EXTRA_PNG_DIR = str(out / 'report' / 'figures_png')
    mrf.RESULTS_DIR, mrf.SESSIONS = out, [r['session_id'] for r in results]
    mrf.figure_r1(results, report_dir); mrf.figure_rs1(results, report_dir); mrf.figure_r2(results, report_dir)
    print(f'figures rebuilt from {len(results)} sessions -> {report_dir}')
    if failures:
        print('failed:', failures, file=sys.stderr)
        return 1
    print('all sessions patched')
    return 0


if __name__ == '__main__':
    sys.exit(main())
