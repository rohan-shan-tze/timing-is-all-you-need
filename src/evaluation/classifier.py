"""
Classification using the trained STDP network.

Based on Diehl & Cook (2015):
- Each neuron has an assigned class label
- For classification, average the responses of neurons in each class
- Predict the class with the highest average response
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple
import numpy as np

from ..network.diehl_network import DiehlNetwork


class Classifier:
    """
    Classifies images using neuron responses and assigned labels.

    The predicted class is determined by averaging the spike responses
    of neurons assigned to each class and selecting the class with
    the highest average.
    """

    def __init__(
        self,
        network: DiehlNetwork,
        neuron_labels: torch.Tensor,
        n_classes: int = 10,
        voting_method: str = "single",
        response_profile: Optional[torch.Tensor] = None
    ):
        """
        Initialize the classifier.

        Args:
            network: Trained DiehlNetwork
            neuron_labels: Class labels for each neuron [n_exc]
            n_classes: Number of classes (10 for MNIST)
            voting_method: "single" (VFO) or "vfa" (Vote-for-All)
            response_profile: Normalized response matrix [n_exc, n_classes],
                              required when voting_method="vfa"
        """
        if voting_method not in ("single", "vfa"):
            raise ValueError(f"voting_method must be 'single' or 'vfa', got '{voting_method}'")
        if voting_method == "vfa" and response_profile is None:
            raise ValueError("response_profile is required when voting_method='vfa'")

        self.network = network
        self.neuron_labels = neuron_labels.to(network.device)
        self.n_classes = n_classes
        self.n_exc = network.n_exc
        self.voting_method = voting_method

        if voting_method == "vfa":
            self.response_profile = response_profile.to(network.device)

        # Precompute masks for each class (used by single/VFO method)
        self.class_masks = []
        self.class_counts = []
        for c in range(n_classes):
            mask = (neuron_labels == c)
            self.class_masks.append(mask)
            self.class_counts.append(mask.sum().item())

    def classify(self, image: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Classify a single image.

        Args:
            image: MNIST image tensor

        Returns:
            Tuple of (predicted class, class scores)
        """
        spike_counts = self.network.present_image(image, learning=False)

        if self.voting_method == "vfa":
            # VFA: scores[c] = sum_i( spike_counts[i] * profile[i, c] )
            class_scores = spike_counts @ self.response_profile  # [n_classes]
        else:
            # VFO (single): average spikes across neurons assigned to each class
            class_scores = torch.zeros(self.n_classes, device=self.network.device)
            for c in range(self.n_classes):
                if self.class_counts[c] > 0:
                    class_scores[c] = spike_counts[self.class_masks[c]].sum() / self.class_counts[c]

        prediction = class_scores.argmax().item()
        return prediction, class_scores

    def evaluate(
        self,
        data_loader: DataLoader,
        max_samples: int = None
    ) -> dict:
        """
        Evaluate classification accuracy on a dataset.

        Args:
            data_loader: DataLoader with test data
            max_samples: Maximum samples to evaluate (None = all)

        Returns:
            Dictionary with evaluation metrics
        """
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        all_scores = []

        pbar = tqdm(data_loader, desc="Evaluating", unit="img")

        for images, labels in pbar:
            for img, label in zip(images, labels):
                pred, scores = self.classify(img)

                predictions.append(pred)
                true_labels.append(label.item())
                all_scores.append(scores.cpu())

                if pred == label.item():
                    correct += 1
                total += 1

                # Update progress
                acc = correct / total
                pbar.set_postfix({'acc': f"{acc:.2%}"})

                if total % 1000 == 0:
                    print(f"  [{total} images] Accuracy so far: {acc:.2%} ({correct}/{total})",
                          flush=True)

                if max_samples and total >= max_samples:
                    break

            if max_samples and total >= max_samples:
                break

        # Compute metrics
        accuracy = correct / total
        predictions = torch.tensor(predictions)
        true_labels = torch.tensor(true_labels)
        all_scores = torch.stack(all_scores)

        # Confusion matrix
        confusion = self._compute_confusion_matrix(predictions, true_labels)

        # Per-class accuracy
        per_class_acc = self._compute_per_class_accuracy(predictions, true_labels)

        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'confusion_matrix': confusion,
            'per_class_accuracy': per_class_acc,
            'predictions': predictions,
            'true_labels': true_labels,
            'scores': all_scores,
        }

        self._log_results(results)

        return results

    def _compute_confusion_matrix(
        self,
        predictions: torch.Tensor,
        true_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute confusion matrix."""
        confusion = torch.zeros(self.n_classes, self.n_classes, dtype=torch.long)
        for pred, true in zip(predictions, true_labels):
            confusion[true, pred] += 1
        return confusion

    def _compute_per_class_accuracy(
        self,
        predictions: torch.Tensor,
        true_labels: torch.Tensor
    ) -> dict:
        """Compute accuracy for each class."""
        per_class = {}
        for c in range(self.n_classes):
            mask = (true_labels == c)
            if mask.sum() > 0:
                correct = (predictions[mask] == c).sum().item()
                total = mask.sum().item()
                per_class[c] = correct / total
            else:
                per_class[c] = 0.0
        return per_class

    def _log_results(self, results: dict):
        """Log evaluation results."""
        print("\n" + "=" * 50)
        print("Classification Results")
        print("=" * 50)
        print(f"Overall Accuracy: {results['accuracy']:.2%}")
        print(f"Correct: {results['correct']} / {results['total']}")
        print("\nPer-class Accuracy:")
        for c, acc in results['per_class_accuracy'].items():
            print(f"  Digit {c}: {acc:.2%}")
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])


class LogRegClassifier:
    """
    Classifies test images using a logistic regression trained on spike-count
    vectors collected during the labeling pass.

    Bypasses neuron-to-class voting entirely: instead of asking which class
    each neuron belongs to, it learns the optimal linear boundary in the
    n_exc-dimensional spike-count space.

    Args:
        network: Trained DiehlNetwork (used during evaluate() to run test images)
        C: Inverse regularization strength. Smaller = stronger L2 regularization.
        max_iter: Maximum solver iterations.
    """

    def __init__(self, network: DiehlNetwork, C: float = 1.0, max_iter: int = 1000):
        self.network = network
        self.C = C
        self.max_iter = max_iter
        self.clf = None
        self.scaler = None
        self.n_classes = 10

    def fit(self, spike_vectors: np.ndarray, labels: np.ndarray):
        """
        Train the logistic regression on labeled spike-count vectors.

        Z-score normalizes each neuron's spike counts before fitting so that
        neurons with high baseline firing rates don't dominate the regression.

        Args:
            spike_vectors: numpy array [n_samples, n_exc], raw spike counts
            labels: numpy array [n_samples], ground-truth digit labels (0-9)
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(spike_vectors)

        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            multi_class='multinomial',
            solver='lbfgs'
        )
        self.clf.fit(X, labels)

        train_acc = self.clf.score(X, labels)
        n_features = spike_vectors.shape[1]
        print(f"Logistic regression fitted on {len(labels)} spike vectors, "
              f"{n_features} features")
        print(f"Training accuracy: {train_acc:.2%}")

    def _classify_counts(self, spike_counts: torch.Tensor) -> int:
        """Classify from a precomputed spike-count vector."""
        X = spike_counts.cpu().numpy().reshape(1, -1)
        X = self.scaler.transform(X)
        return int(self.clf.predict(X)[0])

    def evaluate(
        self,
        data_loader: DataLoader,
        max_samples: int = None
    ) -> dict:
        """
        Evaluate on the full test set.

        Returns the same dict structure as Classifier.evaluate() so results
        are directly comparable and compatible with collect_results.py.

        Args:
            data_loader: DataLoader with test data
            max_samples: Maximum samples to evaluate (None = all)

        Returns:
            Dictionary with accuracy, correct, total, confusion_matrix,
            per_class_accuracy.
        """
        correct = 0
        total = 0
        predictions = []
        true_labels = []

        pbar = tqdm(data_loader, desc="Evaluating (logreg)", unit="img")

        for images, labels in pbar:
            for img, label in zip(images, labels):
                spike_counts = self.network.present_image(img, learning=False)
                pred = self._classify_counts(spike_counts)

                predictions.append(pred)
                true_labels.append(label.item())

                if pred == label.item():
                    correct += 1
                total += 1

                pbar.set_postfix({'acc': f"{correct/total:.2%}"})

                if total % 1000 == 0:
                    print(f"  [{total} images] Accuracy so far: {correct/total:.2%} "
                          f"({correct}/{total})", flush=True)

                if max_samples and total >= max_samples:
                    break

            if max_samples and total >= max_samples:
                break

        accuracy = correct / total
        predictions = torch.tensor(predictions)
        true_labels_t = torch.tensor(true_labels)

        # Confusion matrix
        confusion = torch.zeros(self.n_classes, self.n_classes, dtype=torch.long)
        for pred, true in zip(predictions, true_labels_t):
            confusion[true, pred] += 1

        # Per-class accuracy
        per_class_acc = {}
        for c in range(self.n_classes):
            mask = (true_labels_t == c)
            if mask.sum() > 0:
                per_class_acc[c] = (predictions[mask] == c).sum().item() / mask.sum().item()
            else:
                per_class_acc[c] = 0.0

        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'confusion_matrix': confusion,
            'per_class_accuracy': per_class_acc,
        }

        print("\n" + "=" * 50)
        print("Classification Results")
        print("=" * 50)
        print(f"Overall Accuracy: {accuracy:.2%}")
        print(f"Correct: {correct} / {total}")
        print("\nPer-class Accuracy:")
        for c, acc in per_class_acc.items():
            print(f"  Digit {c}: {acc:.2%}")
        print("\nConfusion Matrix:")
        print(confusion)

        return results


class EnsembleClassifier:
    """
    Classification using multiple passes over the same image.

    Since Poisson encoding is stochastic, running multiple trials
    and averaging can improve accuracy.
    """

    def __init__(
        self,
        network: DiehlNetwork,
        neuron_labels: torch.Tensor,
        n_trials: int = 10,
        n_classes: int = 10,
        voting_method: str = "single",
        response_profile: Optional[torch.Tensor] = None
    ):
        """
        Initialize the ensemble classifier.

        Args:
            network: Trained DiehlNetwork
            neuron_labels: Class labels for each neuron
            n_trials: Number of trials to average
            n_classes: Number of classes
            voting_method: "single" (VFO) or "vfa" (Vote-for-All)
            response_profile: Required when voting_method="vfa"
        """
        self.classifier = Classifier(
            network, neuron_labels, n_classes,
            voting_method=voting_method,
            response_profile=response_profile
        )
        self.n_trials = n_trials

    def classify(self, image: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Classify image using multiple trials.

        Args:
            image: MNIST image tensor

        Returns:
            Tuple of (predicted class, averaged class scores)
        """
        all_scores = []

        for _ in range(self.n_trials):
            _, scores = self.classifier.classify(image)
            all_scores.append(scores)

        # Average scores across trials
        avg_scores = torch.stack(all_scores).mean(dim=0)
        prediction = avg_scores.argmax().item()

        return prediction, avg_scores

    def evaluate(
        self,
        data_loader: DataLoader,
        max_samples: int = None
    ) -> dict:
        """
        Evaluate with ensemble classification.

        Args:
            data_loader: DataLoader with test data
            max_samples: Maximum samples to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        correct = 0
        total = 0

        pbar = tqdm(data_loader, desc=f"Evaluating ({self.n_trials} trials)", unit="img")

        for images, labels in pbar:
            for img, label in zip(images, labels):
                pred, _ = self.classify(img)

                if pred == label.item():
                    correct += 1
                total += 1

                pbar.set_postfix({'acc': f"{correct/total:.2%}"})

                if max_samples and total >= max_samples:
                    break

            if max_samples and total >= max_samples:
                break

        accuracy = correct / total
        print(f"\nEnsemble Accuracy ({self.n_trials} trials): {accuracy:.2%}")

        return {'accuracy': accuracy, 'correct': correct, 'total': total}
