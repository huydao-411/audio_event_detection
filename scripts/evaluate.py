"""
Evaluation Script for Audio Event Detection Model
Comprehensive evaluation with metrics, visualizations, and analysis
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import yaml

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ast_model import AudioSpectrogramTransformer
from utils.dataset import AudioEventDataset, create_data_loaders
from utils.metrics import MetricsCalculator


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self,
                 model_path: str,
                 config_path: str = "configs/config.yaml",
                 device: str = "cuda"):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_classes = self.config['model']['num_classes']
        self.class_names = [c['name'] for c in self.config['target_classes']]
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(self.num_classes, self.class_names)
        
        print(f"Evaluator initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model"""
        model = AudioSpectrogramTransformer(
            config_path=str(PROJECT_ROOT / "configs" / "config.yaml")
        )
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Strip 'module.' prefix if saved with DataParallel (multi-GPU)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model
    
    def evaluate(self, test_loader) -> dict:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation results
        """
        print("\nEvaluating model...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).squeeze()
                
                # Forward pass
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_prob)
        
        # Get confusion matrix
        cm = self.metrics_calculator.get_confusion_matrix(y_true, y_pred)
        
        # Get classification report
        report = self.metrics_calculator.get_classification_report(y_true, y_pred)
        
        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
        }
        
        return results
    
    def plot_results(self, results: dict, output_dir: str = "results/plots"):
        """
        Create visualization plots
        
        Args:
            results: Evaluation results
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating visualizations...")
        
        # 1. Confusion Matrix
        print("  - Confusion matrix...")
        self.metrics_calculator.plot_confusion_matrix(
            results['predictions']['y_true'],
            results['predictions']['y_pred'],
            save_path=os.path.join(output_dir, 'confusion_matrix.png'),
            normalize=True
        )
        
        # 2. Per-class metrics bar plot
        print("  - Per-class metrics...")
        self._plot_per_class_metrics(results['metrics'], output_dir)
        
        # 3. ROC curves
        print("  - ROC curves...")
        self._plot_roc_curves(results['predictions'], output_dir)
        
        # 4. Precision-Recall curves
        print("  - Precision-Recall curves...")
        self._plot_pr_curves(results['predictions'], output_dir)
        
        print(f"\nPlots saved to {output_dir}")
    
    def _plot_per_class_metrics(self, metrics: dict, output_dir: str):
        """Plot per-class metrics"""
        # Extract per-class metrics
        precision = []
        recall = []
        f1 = []
        
        for class_name in self.class_names:
            precision.append(metrics.get(f'precision_{class_name}', 0))
            recall.append(metrics.get(f'recall_{class_name}', 0))
            f1.append(metrics.get(f'f1_{class_name}', 0))
        
        # Create plot
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300)
        plt.close()
    
    def _plot_roc_curves(self, predictions: dict, output_dir: str):
        """Plot ROC curves"""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        y_true = predictions['y_true']
        y_prob = predictions['y_prob']
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Plot ROC curve for each class
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-class')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
        plt.close()
    
    def _plot_pr_curves(self, predictions: dict, output_dir: str):
        """Plot Precision-Recall curves"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize
        
        y_true = predictions['y_true']
        y_prob = predictions['y_prob']
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Plot PR curve for each class
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            
            plt.plot(recall, precision, lw=2,
                    label=f'{class_name} (AP = {ap:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Multi-class')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=300)
        plt.close()
    
    def save_results(self, results: dict, output_path: str = "results/evaluation_results.json"):
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results
            output_path: Path to save results
        """
        # Prepare results for JSON serialization
        json_results = {
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in results['metrics'].items()},
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        
        # Save to JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate Audio Event Detection Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to processed metadata CSV')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Audio Event Detection - Model Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # Load test data
    print(f"\nLoading and splitting data from CSV: {args.data}")
    metadata_df = pd.read_csv(args.data)
    
    # Drop rows with NaN labels if any
    metadata_df = metadata_df.dropna(subset=['label'])
    metadata_df['label'] = metadata_df['label'].astype(int)
    
    # Split: 80% train, 10% val, 10% test (reproducible split)
    from sklearn.model_selection import train_test_split
    _, temp_df = train_test_split(metadata_df, test_size=0.2, random_state=42, stratify=metadata_df['label'])
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"Test Set Size: {len(test_df)}")
    
    # Create test loader
    config_path = str(PROJECT_ROOT / args.config)
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    batch_size = config_dict['training']['batch_size']
    num_workers = 0 # Use 0 for Windows local testing
    pin_memory = config_dict.get('hardware', {}).get('pin_memory', True)

    test_dataset = AudioEventDataset(test_df, config_path, mode='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    # Run evaluation
    results = evaluator.evaluate(test_loader)
    
    # Print metrics
    evaluator.metrics_calculator.print_metrics(results['metrics'])
    
    # Print classification report
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Generate plots
    evaluator.plot_results(results, os.path.join(args.output, 'plots'))
    
    # Save results
    evaluator.save_results(results, os.path.join(args.output, 'evaluation_results.json'))
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
