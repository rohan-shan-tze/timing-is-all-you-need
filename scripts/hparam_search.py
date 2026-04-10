"""
Hyperparameter grid search for the Diehl & Cook (2015) STDP network.

Generates and submits one SLURM job per hyperparameter combination.
Results are collected by scripts/collect_results.py after all jobs finish.

Usage:
    # Edit the SEARCH_SPACE and FIXED_ARGS dicts below, then run:
    python scripts/hparam_search.py

    # Dry run (print job scripts without submitting):
    python scripts/hparam_search.py --dry-run

    # Limit number of combinations submitted (e.g. first 10):
    python scripts/hparam_search.py --max-jobs 10
"""
import sys
import argparse
import subprocess
import itertools
from pathlib import Path
from datetime import datetime


# =============================================================================
# SEARCH SPACE, edit to define grid
# Keys must match train.py CLI argument names (without --)
# =============================================================================
SEARCH_SPACE = {
    'eta':               [0.0001, 0.0005, 0.001],
    'theta_increment':   [0.0125, 0.05, 0.1],
    'w_inh_exc':         [10.0, 17.0],
}

# =============================================================================
# FIXED ARGS applied to every job (edit as needed)
# =============================================================================
FIXED_ARGS = {
    'n_exc':    30,
    'epochs':   3,
    'device':   'cuda',
    'evaluate': True,   # run labelling + test accuracy after training
}

# =============================================================================
# SLURM SETTINGS edit as needed, but this works for my csf environment 
# =============================================================================
SLURM_SETTINGS = {
    'partition':   'gpuL',
    'gpus':        1,
    'time':        '2-0',
    'cpus':        4,
    'output_dir':  'logs/hparam',   # SLURM stdout/stderr logs go here
    'venv':        '~/stdp_env',    # path to your virtualenv
}

# Path to the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search hyperparameter tuning")
    parser.add_argument('--dry-run', action='store_true',
                        help='Print job scripts without submitting to SLURM')
    parser.add_argument('--max-jobs', type=int, default=None,
                        help='Maximum number of jobs to submit (default: all)')
    parser.add_argument('--no-slurm', action='store_true',
                        help='Run jobs locally (sequentially) instead of via sbatch')
    return parser.parse_args()


def build_train_command(combo: dict, run_id: str) -> str:
    """Build the train.py command string for a given hyperparameter combo."""
    checkpoint_dir = f"checkpoints/hparam/{run_id}"

    parts = [f"python scripts/train.py"]
    parts.append(f"--checkpoint_dir {checkpoint_dir}")

    # Fixed args
    for key, val in FIXED_ARGS.items():
        if isinstance(val, bool):
            if val:
                parts.append(f"--{key}")
        else:
            parts.append(f"--{key} {val}")

    # Hyperparameter combo
    for key, val in combo.items():
        parts.append(f"--{key} {val}")

    return " \\\n  ".join(parts)


def build_slurm_script(command: str, run_id: str) -> str:
    """Build a SLURM batch script for a single job."""
    log_dir = PROJECT_ROOT / SLURM_SETTINGS['output_dir']
    return f"""#!/bin/bash --login

#SBATCH -o {log_dir}/{run_id}.o%j
#SBATCH -p {SLURM_SETTINGS['partition']}
#SBATCH -G {SLURM_SETTINGS['gpus']}
#SBATCH -t {SLURM_SETTINGS['time']}
#SBATCH -n {SLURM_SETTINGS['cpus']}

/bin/date
/bin/hostname

source {SLURM_SETTINGS['venv']}/bin/activate

cd {PROJECT_ROOT}

echo "Starting job: {run_id}"
echo "----"

{command}
"""


def combo_to_run_id(combo: dict, index: int) -> str:
    """Generate a short unique run ID from the combo values."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_".join(f"{k}{v}" for k, v in combo.items())
    # Keep it reasonably short
    suffix = suffix.replace(".", "p").replace("-", "m")
    return f"gs{index:03d}_{suffix}_{timestamp}"


def main():
    args = parse_args()

    # Generate all combinations
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total = len(combos)
    if args.max_jobs is not None:
        combos = combos[:args.max_jobs]

    print(f"Grid search: {total} total combinations")
    if args.max_jobs is not None:
        print(f"  Submitting first {len(combos)} only")
    print(f"Search space:")
    for k, v in SEARCH_SPACE.items():
        print(f"  {k}: {v}")
    print(f"Fixed args: {FIXED_ARGS}")
    print()

    # Create log directory
    log_dir = PROJECT_ROOT / SLURM_SETTINGS['output_dir']
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    for i, combo in enumerate(combos):
        run_id = combo_to_run_id(combo, i)
        command = build_train_command(combo, run_id)

        print(f"[{i+1}/{len(combos)}] {run_id}")
        print(f"  Params: {combo}")

        if args.dry_run:
            print(f"  Command:\n  {command}\n")
            continue

        if args.no_slurm:
            # Run locally (blocking, runs one at a time)
            print(f"  Running locally...")
            result = subprocess.run(
                command, shell=True, cwd=PROJECT_ROOT
            )
            if result.returncode != 0:
                print(f"  WARNING: job exited with code {result.returncode}")
        else:
            # Submit via sbatch
            script = build_slurm_script(command, run_id)
            script_path = log_dir / f"{run_id}.sh"
            script_path.write_text(script)

            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"  Submitted: SLURM job {job_id}")
                submitted += 1
            else:
                print(f"  ERROR submitting: {result.stderr.strip()}")

        print()

    if not args.dry_run and not args.no_slurm:
        print(f"Submitted {submitted}/{len(combos)} jobs.")
        print(f"Monitor with: squeue")
        print(f"Collect results with: python scripts/collect_results.py checkpoints/hparam/")


if __name__ == "__main__":
    main()
