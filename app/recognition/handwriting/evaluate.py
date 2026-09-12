"""
Handwriting Recognition Evaluation Script

This module evaluates a trained handwriting CNN model on test data,
computing metrics, confusion matrices, and per-font/source performance.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from app.recognition.handwriting.dataset import HandwritingDataset
from app.recognition.handwriting.preprocessing import HandwritingPreprocessor
from app.recognition.handwriting.model import HandwritingCNN


class Evaluator:
    """
    Evaluation orchestration for handwriting recognition model.
    
    Loads a trained checkpoint and evaluates on test data with detailed metrics.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        csv_path: str,
        output_dir: str = "app/data/models/handwriting/evaluation",
        batch_size: int = 128,
        num_workers: int = 0  # Changed to 0 for Windows compatibility
    ):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            csv_path: Path to test CSV manifest
            output_dir: Directory to save evaluation results
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.dataset = None
        self.loader = None
        self.label_to_idx = None
        self.idx_to_label = None
        self.checkpoint_info = None
        
    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load the trained model checkpoint.
        
        Returns:
            Checkpoint dictionary with model state and metadata
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please train the model first using train.py"
            )
        
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Extract metadata
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
            'num_classes': checkpoint.get('num_classes', 0),
            'target_size': checkpoint.get('target_size', 64)
        }
        
        print(f"  Epoch: {self.checkpoint_info['epoch']}")
        print(f"  Best validation accuracy: {self.checkpoint_info['best_val_accuracy']:.2f}%")
        print(f"  Number of classes: {self.checkpoint_info['num_classes']}")
        
        # Get label mappings from checkpoint
        self.label_to_idx = checkpoint.get('label_to_idx', {})
        self.idx_to_label = checkpoint.get('idx_to_label', {})
        
        if not self.label_to_idx:
            raise ValueError("Checkpoint missing label_to_idx mapping")
        
        # Create model
        num_classes = len(self.label_to_idx)
        target_size = self.checkpoint_info['target_size']
        
        self.model = HandwritingCNN(
            num_classes=num_classes,
            input_channels=1,
            input_size=target_size
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully with {num_classes} classes")
        
        return checkpoint
    
    def load_data(self):
        """Load the test dataset."""
        if not self.csv_path.exists():
            # Try alternative path if the default doesn't exist
            alt_path = Path("C:/Users/Sameer/AI-Smart-Blackboard/app/data/handwriting/manifest.csv")
            if alt_path.exists():
                print(f"CSV not found at default path. Using: {alt_path}")
                self.csv_path = alt_path
            else:
                raise FileNotFoundError(f"Test CSV not found: {self.csv_path}")
        
        print(f"Loading test data from {self.csv_path}")
        
        # Create transform matching training
        preprocessor = HandwritingPreprocessor(
            target_size=self.checkpoint_info['target_size']
        )
        
        def transform(image):
            return preprocessor.preprocess(image, return_tensor=True)
        
        # Determine image_root - use the directory containing the CSV
        image_root = str(self.csv_path.parent)
        print(f"Image root: {image_root}")
        
        # Create dataset
        self.dataset = HandwritingDataset(
            csv_path=str(self.csv_path),
            transform=transform,
            image_root=image_root
        )
        
        # Use model's label mapping for consistent evaluation
        self.dataset.label_to_idx = self.label_to_idx
        self.dataset.idx_to_label = self.idx_to_label
        
        print(f"Test dataset size: {len(self.dataset)}")
        
        # Create data loader with pin_memory=False for Windows compatibility
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False  # Set to False for Windows
        )
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation on the test dataset.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")
        
        if self.loader is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        print("\nRunning evaluation...")
        
        # Track predictions
        all_preds = []
        all_labels = []
        all_original_labels = []
        all_sources = []
        all_fonts = []
        all_confidences = []
        
        for batch_idx, (images, labels, original_labels) in enumerate(self.loader):
            images = images.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_original_labels.extend(original_labels)
            all_confidences.extend(probs.max(dim=1)[0].cpu().numpy())
            
            # Get metadata from dataset
            batch_indices = range(
                batch_idx * self.loader.batch_size,
                min((batch_idx + 1) * self.loader.batch_size, len(self.dataset))
            )
            for idx in batch_indices:
                item = self.dataset.data[idx]
                all_sources.append(item.get('source', 'unknown'))
                all_fonts.append(item.get('font', 'unknown'))
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels, all_preds, all_confidences,
            all_original_labels, all_sources, all_fonts
        )
        
        return metrics
    
    def _calculate_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        confidences: np.ndarray,
        original_labels: List[str],
        sources: List[str],
        fonts: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate all evaluation metrics.
        
        Returns:
            Dictionary with metrics including accuracy, per-class, per-font, etc.
        """
        # Overall metrics
        correct = (preds == labels)
        incorrect = ~correct
        num_correct = np.sum(correct)
        num_incorrect = np.sum(incorrect)
        total = len(labels)
        
        # Top-1 accuracy
        top1_accuracy = 100.0 * num_correct / total
        
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total samples: {total}")
        print(f"Correct: {num_correct}")
        print(f"Incorrect: {num_incorrect}")
        print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
        
        # Per-class accuracy
        unique_classes = np.unique(labels)
        per_class_accuracy = {}
        per_class_correct = {}
        per_class_total = {}
        
        for cls in unique_classes:
            cls_mask = (labels == cls)
            cls_total = np.sum(cls_mask)
            cls_correct = np.sum(correct & cls_mask)
            per_class_total[cls] = cls_total
            per_class_correct[cls] = cls_correct
            per_class_accuracy[cls] = 100.0 * cls_correct / cls_total if cls_total > 0 else 0.0
        
        # Print per-class accuracy
        print(f"\n{'='*60}")
        print(f"PER-CLASS ACCURACY")
        print(f"{'='*60}")
        for cls in sorted(unique_classes):
            label = self.idx_to_label.get(cls, f"class_{cls}")
            acc = per_class_accuracy[cls]
            count = per_class_total[cls]
            print(f"  '{label}': {acc:.2f}% ({count} samples)")
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Font-specific accuracy
        font_accuracy = {}
        font_counts = defaultdict(int)
        font_correct = defaultdict(int)
        
        for i, font in enumerate(fonts):
            font_counts[font] += 1
            if correct[i]:
                font_correct[font] += 1
        
        for font in font_counts:
            font_accuracy[font] = 100.0 * font_correct[font] / font_counts[font]
        
        # Source-specific accuracy
        source_accuracy = {}
        source_counts = defaultdict(int)
        source_correct = defaultdict(int)
        
        for i, source in enumerate(sources):
            source_counts[source] += 1
            if correct[i]:
                source_correct[source] += 1
        
        for source in source_counts:
            source_accuracy[source] = 100.0 * source_correct[source] / source_counts[source]
        
        # Confusing pairs (most common misclassifications)
        confusion_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                if i != j and cm[i][j] > 0:
                    confusion_pairs.append({
                        'true': self.idx_to_label.get(i, f"class_{i}"),
                        'predicted': self.idx_to_label.get(j, f"class_{j}"),
                        'count': int(cm[i][j])
                    })
        
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        # Print font accuracy
        if font_accuracy:
            print(f"\n{'='*60}")
            print(f"FONT-SPECIFIC ACCURACY")
            print(f"{'='*60}")
            for font, acc in sorted(font_accuracy.items()):
                print(f"  {font}: {acc:.2f}% ({font_counts[font]} samples)")
        
        # Print source accuracy
        if source_accuracy:
            print(f"\n{'='*60}")
            print(f"SOURCE-SPECIFIC ACCURACY")
            print(f"{'='*60}")
            for source, acc in sorted(source_accuracy.items()):
                print(f"  {source}: {acc:.2f}% ({source_counts[source]} samples)")
        
        # Print confusing pairs
        if confusion_pairs:
            print(f"\n{'='*60}")
            print(f"TOP CONFUSIONS")
            print(f"{'='*60}")
            for pair in confusion_pairs[:10]:
                print(f"  '{pair['true']}' confused with '{pair['predicted']}': {pair['count']} times")
        
        # Compile results
        results = {
            'total_samples': int(total),
            'correct_predictions': int(num_correct),
            'incorrect_predictions': int(num_incorrect),
            'top1_accuracy': float(top1_accuracy),
            'per_class_accuracy': {
                self.idx_to_label.get(cls, f"class_{cls}"): float(acc)
                for cls, acc in per_class_accuracy.items()
            },
            'font_accuracy': font_accuracy,
            'source_accuracy': source_accuracy,
            'confusion_pairs': confusion_pairs[:10],
            'confusion_matrix': cm.tolist(),
            'label_mapping': self.idx_to_label,
            'checkpoint_info': self.checkpoint_info
        }
        
        return results
    
    def save_confusion_matrix(self, cm: np.ndarray):
        """
        Save confusion matrix visualization.
        """
        try:
            # Create labels
            labels = [self.idx_to_label.get(i, f"{i}") for i in range(len(cm))]
            
            # Plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                square=True
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save
            cm_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to {cm_path}")
            
        except Exception as e:
            print(f"Warning: Could not save confusion matrix: {e}")
    
    def save_results(self, results: Dict[str, Any]):
        """
        Save evaluation results to JSON.
        """
        results_path = self.output_dir / "evaluation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results_copy = results.copy()
        if 'confusion_matrix' in results_copy:
            results_copy['confusion_matrix'] = results_copy['confusion_matrix']
        
        with open(results_path, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        print(f"Results saved to {results_path}")
        
        # Also save confusion matrix as PNG
        if 'confusion_matrix' in results:
            cm = np.array(results['confusion_matrix'])
            self.save_confusion_matrix(cm)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline.
        
        Returns:
            Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("HANDWRITING RECOGNITION EVALUATION")
        print("="*60)
        
        # Load checkpoint
        self.load_checkpoint()
        
        # Load data
        self.load_data()
        
        # Evaluate
        results = self.evaluate()
        
        # Save results
        self.save_results(results)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Results saved to: {self.output_dir}")
        
        return results


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate handwriting recognition model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="C:/Users/Sameer/AI-Smart-Blackboard/app/data/models/handwriting/checkpoint_best.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="C:/Users/Sameer/AI-Smart-Blackboard/app/data/datasets/handwriting/manifest.csv",
        help="Path to test CSV manifest"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="C:/Users/Sameer/AI-Smart-Blackboard/app/data/models/handwriting/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,  # Changed to 0 for Windows compatibility
        help="Number of data loading workers (0 for Windows)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = Evaluator(
            checkpoint_path=args.checkpoint,
            csv_path=args.data,
            output_dir=args.output,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        
        # Run evaluation
        results = evaluator.run()
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {results['top1_accuracy']:.2f}%")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Correct: {results['correct_predictions']}")
        print(f"Incorrect: {results['incorrect_predictions']}")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\nPlease ensure:")
        print("  1. The model is trained first (run train.py)")
        print("  2. The checkpoint path is correct")
        print("  3. The test dataset CSV exists")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()