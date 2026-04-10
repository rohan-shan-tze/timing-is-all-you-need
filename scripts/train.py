"""
Training script for the Diehl & Cook (2015) STDP network.

Usage:
    python scripts/train.py [--n_exc 400] [--epochs 3] [--device cuda] [--max_images 1000]
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.network.diehl_network import DiehlNetwork
from src.training.trainer import Trainer
from src.training.labeling import NeuronLabeler
from src.evaluation.classifier import Classifier, LogRegClassifier
from src.utils.data_loader import get_mnist_loaders
from src.utils.visualization import visualize_weights, visualize_neuron_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train STDP Network on MNIST")
    parser.add_argument('--n_exc', type=int, default=400,
                        help='Number of excitatory neurons')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Limit training images per epoch (for quick trials)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory for checkpoints (auto-generated if not specified)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory for MNIST data')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize weights after training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate after training')
    parser.add_argument('--no-scale-params', action='store_true',
                        help='Disable automatic parameter scaling for network size '
                             '(uses paper fixed values regardless of n_exc)')
    parser.add_argument('--voting', type=str, default='single',
                        choices=['single', 'vfa', 'logreg'],
                        help='Decoding method for --evaluate: single (VFO, default), '
                             'vfa (Vote-for-All), or logreg (logistic regression on spike vectors)')
    parser.add_argument('--logreg-C', type=float, default=1.0,
                        help='Logistic regression inverse regularization strength (default: 1.0). '
                             'Smaller values = stronger L2 regularization.')
    parser.add_argument('--min_spikes', type=int, default=None,
                        help='Override min spikes threshold for adaptive retry loop '
                             '(default: 5, or scaled value if --scale-params is active)')
    parser.add_argument('--encoding', type=str, default='poisson',
                        choices=['poisson', 'ttfs'],
                        help='Input encoding: poisson (stochastic rate coding, default) or '
                             'ttfs (deterministic time-to-first-spike, ~3x faster per epoch)')
    parser.add_argument('--labeling', type=str, default='argmax',
                        choices=['argmax', 'balanced'],
                        help='Neuron-to-class assignment method: argmax (default) or '
                             'balanced (guarantees floor(n_exc/n_classes) neurons per class). '
                             'Only affects single/VFO voting.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Epoch number to start from when resuming (default: 0)')
    parser.add_argument('--preprocessing', type=str, default='none',
                        choices=['none', 'dog'],
                        help='Input preprocessing: none (raw pixels, default) or '
                             'dog (Difference-of-Gaussian edge filtering)')
    parser.add_argument('--dog-sigma1', type=float, default=1.0,
                        help='DoG inner Gaussian sigma (default: 1.0)')
    parser.add_argument('--dog-sigma2', type=float, default=2.0,
                        help='DoG outer Gaussian sigma (default: 2.0)')
    parser.add_argument('--no-on-off', action='store_true',
                        help='Use only ON-center DoG channel (default: use both ON and OFF, input_dim=1568)')

    # --- Hyperparameter overrides (all optional, default None = use config file value) ---
    # STDP
    parser.add_argument('--eta', type=float, default=None,
                        help='STDP learning rate (default: 0.0001)')
    parser.add_argument('--x_tar', type=float, default=None,
                        help='STDP target presynaptic trace value (default: 0.4)')
    parser.add_argument('--mu', type=float, default=None,
                        help='STDP weight dependence exponent (default: 0.2)')
    parser.add_argument('--tau_pre', type=float, default=None,
                        help='STDP presynaptic trace time constant in ms (default: 20.0)')
    # Homeostasis
    parser.add_argument('--theta_increment', type=float, default=None,
                        help='Adaptive threshold increment per spike in mV (default: 0.05)')
    parser.add_argument('--tau_theta', type=float, default=None,
                        help='Adaptive threshold decay time constant in ms (default: 1e7)')
    # Synapse weights
    parser.add_argument('--w_inh_exc', type=float, default=None,
                        help='Inhibitory->excitatory fixed weight in nS (default: 17.0)')
    parser.add_argument('--w_exc_inh', type=float, default=None,
                        help='Excitatory->inhibitory fixed weight in nS (default: 10.0)')
    parser.add_argument('--w_max', type=float, default=None,
                        help='Maximum plastic weight (default: 1.0)')
    parser.add_argument('--w_init_min', type=float, default=None,
                        help='Initial weight range minimum (default: 0.0)')
    parser.add_argument('--w_init_max', type=float, default=None,
                        help='Initial weight range maximum (default: 0.3)')
    # Encoding
    parser.add_argument('--presentation_time', type=float, default=None,
                        help='Image presentation time in ms (default: 350.0 for Poisson)')
    parser.add_argument('--max_rate', type=float, default=None,
                        help='Maximum Poisson firing rate in Hz (default: 63.75)')
    parser.add_argument('--rate_increment', type=float, default=None,
                        help='Rate increase per retry in Hz (default: 32.0)')
    parser.add_argument('--ttfs_presentation_time', type=float, default=None,
                        help='TTFS encoding window in ms (default: 100.0)')
    # Excitatory neuron
    parser.add_argument('--exc_tau_membrane', type=float, default=None,
                        help='Excitatory membrane time constant in ms (default: 100.0)')
    parser.add_argument('--exc_v_thresh', type=float, default=None,
                        help='Excitatory base firing threshold in mV (default: -52.0)')
    parser.add_argument('--exc_v_reset', type=float, default=None,
                        help='Excitatory reset potential in mV (default: -65.0)')
    parser.add_argument('--exc_refrac_time', type=float, default=None,
                        help='Excitatory refractory period in ms (default: 5.0)')
    # Inhibitory neuron
    parser.add_argument('--inh_tau_membrane', type=float, default=None,
                        help='Inhibitory membrane time constant in ms (default: 10.0)')
    parser.add_argument('--inh_v_thresh', type=float, default=None,
                        help='Inhibitory base firing threshold in mV (default: -40.0)')
    parser.add_argument('--inh_v_reset', type=float, default=None,
                        help='Inhibitory reset potential in mV (default: -45.0)')
    parser.add_argument('--inh_refrac_time', type=float, default=None,
                        help='Inhibitory refractory period in ms (default: 2.0)')
    # Simulation
    parser.add_argument('--dt', type=float, default=None,
                        help='Simulation timestep in ms (default: 0.5)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Generate unique run directory from timestamp + key hyperparams
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_n{args.n_exc}_e{args.epochs}"
        args.checkpoint_dir = f"checkpoints/{run_id}"
    print(f"Run directory: {args.checkpoint_dir}", flush=True)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
    else:
        config = None
        print("Using default config")

    # Create data loaders
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=1,
        shuffle_train=True
    )
    if args.max_images is not None:
        print(f"Training set: {args.max_images} images (limited from {len(train_loader.dataset)})")
    else:
        print(f"Training set: {len(train_loader.dataset)} images")
    print(f"Test set: {len(test_loader.dataset)} images")

    # Apply CLI hyperparameter overrides (only args explicitly passed, i.e. not None)
    if config is None:
        config = {}
    _overrides = [
        ('stdp',       'eta',                    args.eta),
        ('stdp',       'x_tar',                  args.x_tar),
        ('stdp',       'mu',                     args.mu),
        ('stdp',       'tau_pre',                args.tau_pre),
        ('homeostasis','theta_increment',         args.theta_increment),
        ('homeostasis','tau_theta',               args.tau_theta),
        ('synapse',    'w_inh_exc',               args.w_inh_exc),
        ('synapse',    'w_exc_inh',               args.w_exc_inh),
        ('synapse',    'w_max',                   args.w_max),
        ('synapse',    'w_init_min',              args.w_init_min),
        ('synapse',    'w_init_max',              args.w_init_max),
        ('encoding',   'presentation_time',       args.presentation_time),
        ('encoding',   'max_rate',                args.max_rate),
        ('encoding',   'rate_increment',          args.rate_increment),
        ('encoding',   'ttfs_presentation_time',  args.ttfs_presentation_time),
        ('exc_neuron', 'tau_membrane',            args.exc_tau_membrane),
        ('exc_neuron', 'v_thresh',                args.exc_v_thresh),
        ('exc_neuron', 'v_reset',                 args.exc_v_reset),
        ('exc_neuron', 'refrac_time',             args.exc_refrac_time),
        ('inh_neuron', 'tau_membrane',            args.inh_tau_membrane),
        ('inh_neuron', 'v_thresh',                args.inh_v_thresh),
        ('inh_neuron', 'v_reset',                 args.inh_v_reset),
        ('inh_neuron', 'refrac_time',             args.inh_refrac_time),
    ]
    for section, key, value in _overrides:
        if value is not None:
            config.setdefault(section, {})[key] = value
            print(f"  Override: {section}.{key} = {value}", flush=True)

    # Build preprocessing config and inject into config dict
    on_off = not args.no_on_off
    config.setdefault('preprocessing', {})
    config['preprocessing']['type'] = args.preprocessing
    config['preprocessing']['sigma1'] = args.dog_sigma1
    config['preprocessing']['sigma2'] = args.dog_sigma2
    config['preprocessing']['on_off'] = on_off

    # Create network
    scale_params = not args.no_scale_params
    print(f"\nCreating network with {args.n_exc} excitatory neurons "
          f"({'scaled' if scale_params else 'paper fixed'} parameters)...")
    dt = args.dt if args.dt is not None else 0.5
    network = DiehlNetwork(
        n_input=784,
        n_exc=args.n_exc,
        dt=dt,
        device=args.device,
        config=config,
        scale_params=scale_params,
        encoding=args.encoding,
        preprocessing=args.preprocessing
    )
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}", flush=True)
        network.load_checkpoint(args.resume)
        print(f"  Resumed from epoch {args.start_epoch}", flush=True)
    if args.min_spikes is not None:
        network.config['encoding']['min_spikes'] = args.min_spikes
        print(f"  min_spikes overridden to: {args.min_spikes}")
    if scale_params:
        print(f"  w_inh_exc:       {network.config['synapse']['w_inh_exc']:.4f}")
        print(f"  theta_increment: {network.config['homeostasis']['theta_increment']:.4f}")
    print(f"  min_spikes:      {network.config['encoding']['min_spikes']}")

    # Create trainer
    trainer = Trainer(
        network=network,
        train_loader=train_loader,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=1000,
        save_interval=10000
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    stats = trainer.train(
        n_epochs=args.epochs,
        use_adaptive=True,
        max_images=args.max_images,
        start_epoch=args.start_epoch
    )

    # Save final model
    final_path = Path(args.checkpoint_dir) / "final_model.pt"
    network.save_checkpoint(str(final_path))
    print(f"\nSaved final model to {final_path}")

    # Visualize weights
    if args.visualize:
        print("\nVisualizing learned weights...")
        vis_path = Path(args.checkpoint_dir) / "weights.png"
        visualize_weights(
            network.W_input_exc,
            title=f"Learned Weights ({args.n_exc} neurons)",
            save_path=str(vis_path)
        )

    # Evaluate
    if args.evaluate:
        print(f"\nVoting method: {args.voting}", flush=True)
        labeler = NeuronLabeler(network, n_classes=10)

        if args.voting == 'logreg':
            # Collect per-image spike vectors from the training set for logreg fitting
            print("Collecting spike vectors from training set (logreg)...")
            spike_vectors, image_labels = labeler.compute_responses_with_vectors(
                train_loader, max_samples=10000
            )
            # Also assign neuron labels (for visualization; not used by logreg classifier)
            neuron_labels = labeler.assign_labels(method='argmax')

            logreg_clf = LogRegClassifier(network, C=args.logreg_C)
            logreg_clf.fit(spike_vectors, image_labels)

            print("\nEvaluating on test set...")
            results = logreg_clf.evaluate(test_loader)

            # Optionally save the fitted model
            try:
                import joblib
                logreg_path = Path(args.checkpoint_dir) / "logreg_model.joblib"
                joblib.dump({'clf': logreg_clf.clf, 'scaler': logreg_clf.scaler},
                            str(logreg_path))
                print(f"Saved logreg model to {logreg_path}")
            except ImportError:
                pass  

        else:
            print("Assigning labels to neurons...")
            labeler.compute_responses(train_loader, max_samples=10000)
            neuron_labels = labeler.assign_labels(method=args.labeling)

            print("\nEvaluating on test set...")
            response_profile = labeler.get_response_profile() if args.voting == 'vfa' else None
            classifier = Classifier(
                network, neuron_labels,
                n_classes=10,
                voting_method=args.voting,
                response_profile=response_profile
            )
            results = classifier.evaluate(test_loader)

        # Save labels
        label_path = Path(args.checkpoint_dir) / "neuron_labels.pt"
        labeler.save(str(label_path))

        # Visualize with labels
        if args.visualize:
            labeled_vis_path = Path(args.checkpoint_dir) / "weights_labeled.png"
            visualize_neuron_labels(
                network.W_input_exc,
                neuron_labels,
                title="Weights with Labels",
                save_path=str(labeled_vis_path)
            )

        # Save results to checkpoint dir for collection by collect_results.py
        results_path = Path(args.checkpoint_dir) / "results.pt"
        torch.save({
            'accuracy':           results['accuracy'],
            'correct':            results['correct'],
            'total':              results['total'],
            'confusion_matrix':   results.get('confusion_matrix'),
            'per_class_accuracy': results.get('per_class_accuracy'),
        }, str(results_path))

        print(f"\n{'='*50}")
        print(f"Voting method: {args.voting}")
        print(f"Final Test Accuracy: {results['accuracy']:.2%}")
        print(f"{'='*50}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
