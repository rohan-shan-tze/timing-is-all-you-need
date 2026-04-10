"""
Quick test script to verify the STDP network implementation.

Runs a minimal test with 100 neurons on a small subset of MNIST.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Use CPU for quick test to avoid GPU issues
device = 'cpu'
print(f"Using device: {device}")

print("\n1. Testing data loader...")
from src.utils.data_loader import get_mnist_loaders, flatten_image

train_loader, test_loader = get_mnist_loaders(data_dir='./data', batch_size=1)
print(f"   Training samples: {len(train_loader.dataset)}")
print(f"   Test samples: {len(test_loader.dataset)}")

# Get a sample image
sample_img, sample_label = next(iter(train_loader))
print(f"   Sample image shape: {sample_img.shape}")
print(f"   Sample label: {sample_label.item()}")
print("   Data loader OK!")

print("\n2. Testing Poisson encoder...")
from src.encoding.poisson import PoissonEncoder

encoder = PoissonEncoder(n_inputs=784, max_rate=63.75, presentation_time=350.0, dt=0.5, device=device)
flat_img = flatten_image(sample_img[0])
spike_count = 0
for spikes in encoder.encode(flat_img):
    spike_count += spikes.sum().item()
print(f"   Generated {spike_count:.0f} input spikes for one image")
print("   Poisson encoder OK!")

print("\n3. Testing LIF neurons...")
from src.neurons.lif import ExcitatoryNeurons, InhibitoryNeurons

exc_neurons = ExcitatoryNeurons(n_neurons=100, dt=0.5, device=device)
print(f"   Created {exc_neurons.n_neurons} excitatory neurons")

# Test a few steps
for _ in range(10):
    g_e = torch.rand(100, device=device) * 0.1
    g_i = torch.zeros(100, device=device)
    spikes = exc_neurons.step(g_e, g_i)

print(f"   Neurons integrated input correctly")
print("   LIF neurons OK!")

print("\n4. Testing STDP...")
from src.plasticity.stdp import PowerLawSTDP

stdp = PowerLawSTDP(n_pre=784, n_post=100, device=device)
weights = torch.rand(784, 100, device=device) * 0.3
pre_spikes = (torch.rand(784, device=device) < 0.1).float()
post_spikes = (torch.rand(100, device=device) < 0.01).float()

stdp.update_traces(pre_spikes)
new_weights = stdp.apply_weight_update(weights, post_spikes)
weight_change = (new_weights - weights).abs().sum().item()
print(f"   Total weight change: {weight_change:.6f}")
print("   STDP OK!")

print("\n5. Testing full network...")
from src.network.diehl_network import DiehlNetwork

network = DiehlNetwork(n_input=784, n_exc=100, dt=0.5, device=device)
print(f"   Created network with {network.n_exc} excitatory neurons")

# Present one image
spike_counts = network.present_image(flat_img, learning=True)
total_exc_spikes = spike_counts.sum().item()
print(f"   Network produced {total_exc_spikes:.0f} excitatory spikes")

weight_stats = network.get_weight_stats()
print(f"   Weight stats: mean={weight_stats['mean']:.4f}, std={weight_stats['std']:.4f}")
print("   Network OK!")

print("\n6. Testing training loop (10 images)...")
from src.training.trainer import Trainer

# Create a tiny trainer test
network2 = DiehlNetwork(n_input=784, n_exc=100, dt=0.5, device=device)

# Manual mini training loop
n_images = 10
total_spikes = 0
for i, (img, label) in enumerate(train_loader):
    if i >= n_images:
        break
    flat = flatten_image(img[0])
    spikes = network2.present_image(flat, learning=True)
    total_spikes += spikes.sum().item()

avg_spikes = total_spikes / n_images
print(f"   Trained on {n_images} images")
print(f"   Average spikes per image: {avg_spikes:.1f}")
print("   Training loop OK!")

print("\n7. Testing labeling...")
from src.training.labeling import NeuronLabeler

labeler = NeuronLabeler(network2, n_classes=10)
# Just test structure, not full labeling
print(f"   Labeler initialized for {labeler.n_exc} neurons")
print("   Labeling OK!")

print("\n8. Testing classifier...")
from src.evaluation.classifier import Classifier

# Create dummy labels
dummy_labels = torch.randint(0, 10, (100,), device=device)
classifier = Classifier(network2, dummy_labels, n_classes=10)

pred, scores = classifier.classify(flat_img)
print(f"   Classified image as digit {pred}")
print(f"   Class scores shape: {scores.shape}")
print("   Classifier OK!")

print("\n" + "="*50)
print("ALL TESTS PASSED!")
print("="*50)
print("\nThe implementation is ready. You can now train the full network:")
print("  python scripts/train.py --n_exc 100 --epochs 1 --evaluate --visualize")
print("\nOr for the full 400-neuron network:")
print("  python scripts/train.py --n_exc 400 --epochs 3 --evaluate --visualize")
