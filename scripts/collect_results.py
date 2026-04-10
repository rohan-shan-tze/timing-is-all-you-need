"""
Collect and summarise hyperparameter search results.

Scans a directory of checkpoint runs, reads results.pt from each,
and prints a sorted table of accuracy vs hyperparameters.

Usage:
    python scripts/collect_results.py checkpoints/hparam/

    # Save table to CSV:
    python scripts/collect_results.py checkpoints/hparam/ --csv results/hparam_summary.csv
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hyperparameter search results")
    parser.add_argument('search_dir', type=str,
                        help='Directory containing hparam run subdirectories')
    parser.add_argument('--csv', type=str, default=None,
                        help='Save summary table to this CSV path')
    parser.add_argument('--top', type=int, default=None,
                        help='Show only top N runs by accuracy')
    return parser.parse_args()


def load_run(run_dir: Path) -> dict:
    """
    Load results and config from a single run directory.
    Returns None if the run is incomplete (no results.pt).
    Both results.pt and final_model.pt live in the checkpoint dir.
    """
    results_path = run_dir / 'results.pt'
    checkpoint_path = run_dir / 'final_model.pt'

    if not results_path.exists():
        return None

    results = torch.load(results_path, map_location='cpu')
    accuracy = results.get('accuracy', None)
    if accuracy is None:
        return None

    # Load config from checkpoint to get hyperparameter values
    config = {}
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})

    return {
        'run_id':          run_dir.name,
        'accuracy':        accuracy,
        'correct':         results.get('correct', '?'),
        'total':           results.get('total', '?'),
        # STDP
        'eta':             config.get('stdp', {}).get('eta', '?'),
        'x_tar':           config.get('stdp', {}).get('x_tar', '?'),
        'mu':              config.get('stdp', {}).get('mu', '?'),
        'tau_pre':         config.get('stdp', {}).get('tau_pre', '?'),
        # Homeostasis
        'theta_increment': config.get('homeostasis', {}).get('theta_increment', '?'),
        'tau_theta':       config.get('homeostasis', {}).get('tau_theta', '?'),
        # Synapse
        'w_inh_exc':       config.get('synapse', {}).get('w_inh_exc', '?'),
        'w_exc_inh':       config.get('synapse', {}).get('w_exc_inh', '?'),
        'w_max':           config.get('synapse', {}).get('w_max', '?'),
        # Encoding
        'presentation_time': config.get('encoding', {}).get('presentation_time', '?'),
        'max_rate':        config.get('encoding', {}).get('max_rate', '?'),
        'min_spikes':      config.get('encoding', {}).get('min_spikes', '?'),
        # Per-class accuracy (summarised as min)
        'min_class_acc':   min(results['per_class_accuracy'].values())
                           if results.get('per_class_accuracy') else '?',
    }


def format_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    args = parse_args()
    search_dir = Path(args.search_dir)

    if not search_dir.exists():
        print(f"Directory not found: {search_dir}")
        sys.exit(1)

    # Find all run subdirectories
    run_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir()])
    print(f"Found {len(run_dirs)} run directories in {search_dir}")

    rows = []
    incomplete = 0
    for run_dir in run_dirs:
        row = load_run(run_dir)
        if row is None:
            incomplete += 1
        else:
            rows.append(row)

    if incomplete > 0:
        print(f"  {incomplete} runs incomplete (no results.pt), still running or failed")

    if not rows:
        print("No completed runs found.")
        return

    # Sort by accuracy descending
    rows.sort(key=lambda r: r['accuracy'], reverse=True)

    if args.top is not None:
        rows = rows[:args.top]

    # Determine which hyperparameter columns actually vary across runs
    # (no point showing columns where every run has the same value)
    all_keys = ['eta', 'x_tar', 'mu', 'tau_pre',
                'theta_increment', 'tau_theta',
                'w_inh_exc', 'w_exc_inh', 'w_max',
                'presentation_time', 'max_rate', 'min_spikes']
    varying_keys = [k for k in all_keys
                    if len(set(str(r[k]) for r in rows)) > 1]

    # Print table
    print(f"\n{'='*80}")
    print(f"Hyperparameter Search Results ({len(rows)} completed runs)")
    print(f"{'='*80}")

    # Header
    col_headers = ['rank', 'accuracy', 'min_class_acc'] + varying_keys + ['run_id']
    col_widths  = [5, 10, 13] + [max(12, len(k)+1) for k in varying_keys] + [40]

    header = "".join(h.ljust(w) for h, w in zip(col_headers, col_widths))
    print(header)
    print("-" * sum(col_widths))

    for rank, row in enumerate(rows, 1):
        acc_str = f"{row['accuracy']:.2%}"
        min_acc_str = f"{row['min_class_acc']:.2%}" if isinstance(row['min_class_acc'], float) else '?'
        vals = [str(rank), acc_str, min_acc_str] + \
               [format_val(row[k]) for k in varying_keys] + \
               [row['run_id']]
        line = "".join(v.ljust(w) for v, w in zip(vals, col_widths))
        print(line)

    print(f"\nBest run: {rows[0]['run_id']}")
    print(f"Best accuracy: {rows[0]['accuracy']:.2%}")
    print(f"Best params:")
    for k in varying_keys:
        print(f"  {k}: {rows[0][k]}")

    # Save CSV
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        all_cols = ['rank', 'accuracy', 'min_class_acc'] + all_keys + ['correct', 'total', 'run_id']
        with open(csv_path, 'w') as f:
            f.write(",".join(all_cols) + "\n")
            for rank, row in enumerate(rows, 1):
                vals = [str(rank), f"{row['accuracy']:.4f}",
                        f"{row['min_class_acc']:.4f}" if isinstance(row['min_class_acc'], float) else '?']
                vals += [format_val(row[k]) for k in all_keys]
                vals += [str(row['correct']), str(row['total']), row['run_id']]
                f.write(",".join(vals) + "\n")
        print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()
