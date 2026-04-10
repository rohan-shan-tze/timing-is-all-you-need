"""
Evaluation script for the trained STDP network.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/final_model.pt
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.network.diehl_network import DiehlNetwork
from src.training.labeling import NeuronLabeler
from src.evaluation.classifier import Classifier, EnsembleClassifier, LogRegClassifier
from src.utils.data_loader import get_mnist_loaders
from src.utils.visualization import (
    visualize_weights,
    visualize_neuron_labels,
    plot_confusion_matrix
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained STDP Network")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to neuron labels (computed if not provided)')
    parser.add_argument('--n_exc', type=int, default=400,
                        help='Number of excitatory neurons')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory for MNIST data')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum test samples to evaluate')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='Number of trials for ensemble classification')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for output files (defaults to results/<run_id> '
                             'derived from checkpoint path)')
    parser.add_argument('--voting', type=str, default='single',
                        choices=['single', 'vfa', 'logreg'],
                        help='Decoding method: single (VFO, default), vfa (Vote-for-All), '
                             'or logreg (logistic regression on spike vectors)')
    parser.add_argument('--logreg-C', type=float, default=1.0,
                        help='Logistic regression inverse regularization strength (default: 1.0). '
                             'Smaller values = stronger L2 regularization.')
    parser.add_argument('--encoding', type=str, default='poisson',
                        choices=['poisson', 'ttfs'],
                        help='Input encoding (default: poisson). Overridden automatically '
                             'if the checkpoint contains encoding information.')
    parser.add_argument('--labeling', type=str, default='argmax',
                        choices=['argmax', 'balanced'],
                        help='Neuron-to-class assignment method: argmax (default) or '
                             'balanced (guarantees floor(n_exc/n_classes) neurons per class). '
                             'Only affects single/VFO voting.')
    parser.add_argument('--preprocessing', type=str, default='none',
                        choices=['none', 'dog'],
                        help='Input preprocessing (default: none). Overridden automatically '
                             'if the checkpoint contains preprocessing information.')
    parser.add_argument('--dog-sigma1', type=float, default=1.0,
                        help='DoG inner Gaussian sigma (default: 1.0)')
    parser.add_argument('--dog-sigma2', type=float, default=2.0,
                        help='DoG outer Gaussian sigma (default: 2.0)')
    parser.add_argument('--no-on-off', action='store_true',
                        help='Use only ON-center DoG channel (default: use both ON and OFF)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Derive output dir from checkpoint path if not specified
    # e.g. checkpoints/20260314_143022_n800_e3/final_model.pt
    #   -> results/20260314_143022_n800_e3/
    if args.output_dir is None:
        run_id = Path(args.checkpoint).parent.name
        args.output_dir = f"results/{run_id}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {args.output_dir}", flush=True)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = None

    # Create network and load checkpoint
    # Preprocessing is restored automatically from checkpoint via load_checkpoint()
    print(f"\nLoading model from {args.checkpoint}...")
    network = DiehlNetwork(
        n_input=784,
        n_exc=args.n_exc,
        dt=0.5,
        device=args.device,
        config=config,
        scale_params=False,
        encoding=args.encoding,
        preprocessing=args.preprocessing
    )
    network.load_checkpoint(args.checkpoint)
    print("Model loaded successfully")

    # Print all hyperparameters restored from checkpoint
    cfg = network.config
    print(f"\nHyperparameters (restored from checkpoint):")
    print(f"  Network:      n_exc={network.n_exc}, n_input={network.n_input}, "
          f"encoding={network.encoding}, preprocessing={network.preprocessing}")
    print(f"  STDP:         eta={cfg['stdp']['eta']}, x_tar={cfg['stdp']['x_tar']}, "
          f"mu={cfg['stdp']['mu']}, tau_pre={cfg['stdp']['tau_pre']}")
    print(f"  Homeostasis:  theta_increment={cfg['homeostasis']['theta_increment']:.6f}, "
          f"tau_theta={cfg['homeostasis']['tau_theta']:.2e}")
    print(f"  Synapse:      w_exc_inh={cfg['synapse']['w_exc_inh']}, "
          f"w_inh_exc={cfg['synapse']['w_inh_exc']}, w_max={cfg['synapse']['w_max']}")
    print(f"  Encoding:     presentation_time={cfg['encoding']['presentation_time']}, "
          f"max_rate={cfg['encoding']['max_rate']}, min_spikes={cfg['encoding']['min_spikes']}, "
          f"rate_increment={cfg['encoding']['rate_increment']}")
    print(f"  Exc neuron:   tau_membrane={cfg['exc_neuron']['tau_membrane']}, "
          f"v_thresh={cfg['exc_neuron']['v_thresh']}, v_reset={cfg['exc_neuron']['v_reset']}, "
          f"refrac_time={cfg['exc_neuron']['refrac_time']}")
    print(f"  Inh neuron:   tau_membrane={cfg['inh_neuron']['tau_membrane']}, "
          f"v_thresh={cfg['inh_neuron']['v_thresh']}, v_reset={cfg['inh_neuron']['v_reset']}, "
          f"refrac_time={cfg['inh_neuron']['refrac_time']}")

    # Print weight statistics
    weight_stats = network.get_weight_stats()
    print(f"\nWeight statistics:")
    print(f"  Mean: {weight_stats['mean']:.4f}")
    print(f"  Std: {weight_stats['std']:.4f}")
    print(f"  Near zero: {weight_stats['near_zero']:.1%}")
    print(f"  Near max: {weight_stats['near_max']:.1%}")

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=1,
        shuffle_train=False
    )

    print(f"Voting method: {args.voting}", flush=True)
    labeler = NeuronLabeler(network, n_classes=10)

    if args.voting == 'logreg':
        # Collect per-image spike vectors from training set for logreg fitting
        print("\nCollecting spike vectors from training set (logreg)...")
        spike_vectors, image_labels = labeler.compute_responses_with_vectors(
            train_loader, max_samples=10000
        )
        # Also run argmax labeling for visualization / label saving
        neuron_labels = labeler.assign_labels(method='argmax')

        logreg_clf = LogRegClassifier(network, C=args.logreg_C)
        logreg_clf.fit(spike_vectors, image_labels)

        # Optionally save the fitted model
        try:
            import joblib
            logreg_path = output_dir / "logreg_model.joblib"
            joblib.dump({'clf': logreg_clf.clf, 'scaler': logreg_clf.scaler},
                        str(logreg_path))
            print(f"Saved logreg model to {logreg_path}")
        except ImportError:
            pass  

        print(f"\nEvaluating on test set...")
        results = logreg_clf.evaluate(test_loader, max_samples=args.max_samples)

    else:
        # Load or compute neuron labels
        if args.labels and Path(args.labels).exists():
            print(f"\nLoading neuron labels from {args.labels}...")
            labeler.load(args.labels)
            neuron_labels = labeler.get_labels()
        else:
            print("\nComputing neuron labels from training set...")
            labeler.compute_responses(train_loader, max_samples=10000)
            neuron_labels = labeler.assign_labels(method=args.labeling)

            # Save computed labels
            label_path = output_dir / "neuron_labels.pt"
            labeler.save(str(label_path))
            print(f"Saved labels to {label_path}")

        # Print label distribution
        dist = labeler.get_class_distribution()
        print("\nNeuron distribution by class:")
        for c in range(10):
            print(f"  Digit {c}: {dist[c]} neurons")

        # Get response profile for VFA
        response_profile = labeler.get_response_profile() if args.voting == 'vfa' else None

        print(f"\nEvaluating on test set...")
        if args.ensemble > 1:
            classifier = EnsembleClassifier(
                network, neuron_labels,
                n_trials=args.ensemble,
                n_classes=10,
                voting_method=args.voting,
                response_profile=response_profile
            )
        else:
            classifier = Classifier(
                network, neuron_labels,
                n_classes=10,
                voting_method=args.voting,
                response_profile=response_profile
            )

        results = classifier.evaluate(test_loader, max_samples=args.max_samples)

    # Save results
    results_path = output_dir / "results.pt"
    torch.save({
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'total': results['total'],
        'confusion_matrix': results.get('confusion_matrix', None),
        'per_class_accuracy': results.get('per_class_accuracy', None),
    }, str(results_path))
    print(f"\nSaved results to {results_path}")

    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")

        # Weights
        weights_path = output_dir / "weights.png"
        visualize_weights(
            network.W_input_exc,
            title=f"Learned Weights ({args.n_exc} neurons)",
            save_path=str(weights_path)
        )

        # Weights with labels
        labeled_path = output_dir / "weights_labeled.png"
        visualize_neuron_labels(
            network.W_input_exc,
            neuron_labels,
            title="Weights with Assigned Labels",
            save_path=str(labeled_path)
        )

        # Confusion matrix
        if 'confusion_matrix' in results and results['confusion_matrix'] is not None:
            conf_path = output_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                results['confusion_matrix'],
                title="Confusion Matrix",
                save_path=str(conf_path)
            )

    # Print final summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Model: {args.checkpoint}")
    print(f"Neurons: {args.n_exc}")
    print(f"Test samples: {results['total']}")
    print(f"Voting method: {args.voting}")
    if args.ensemble > 1:
        print(f"Ensemble trials: {args.ensemble}")
    print(f"\nFinal Accuracy: {results['accuracy']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
