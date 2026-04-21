"""
Comprehensive Evaluation Framework for Road Damage Detection
Provides metrics, visualization, and statistical analysis tools
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    cohen_kappa_score, matthews_corrcoef
)
from scipy import stats
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Comprehensive classification metrics calculator"""
    
    @staticmethod
    def compute_all_metrics(y_true, y_pred, y_prob=None, class_names=None):
        """
        Compute all classification metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional, for ROC/AUC)
            class_names: Names of classes (default: D00, D10, D20, D40)
        
        Returns:
            Dictionary of all metrics
        """
        if class_names is None:
            class_names = ['D00 (Longitudinal)', 'D10 (Transverse)', 
                          'D20 (Alligator)', 'D40 (Pothole)']
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Additional metrics
        metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics['per_class'][class_name] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i])
                }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        
        # ROC AUC if probabilities provided
        if y_prob is not None:
            try:
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_auc_score
                
                n_classes = len(np.unique(y_true))
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                
                # Micro-average ROC AUC
                metrics['roc_auc_micro'] = float(roc_auc_score(y_true_bin, y_prob, average='micro'))
                # Macro-average ROC AUC
                metrics['roc_auc_macro'] = float(roc_auc_score(y_true_bin, y_prob, average='macro'))
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
        
        return metrics
    
    @staticmethod
    def print_metrics_summary(metrics, title="Classification Metrics"):
        """Print formatted metrics summary"""
        print("\n" + "="*70)
        print(f"{title:^70}")
        print("="*70)
        
        print(f"\n{'Overall Performance':^70}")
        print("-"*70)
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro):  {metrics['f1_macro']:.4f}")
        print(f"  Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")
        
        if 'roc_auc_macro' in metrics:
            print(f"  ROC AUC (macro):   {metrics['roc_auc_macro']:.4f}")
        
        print(f"\n{'Per-Class Performance':^70}")
        print("-"*70)
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"  {class_name:20s} - P: {class_metrics['precision']:.3f}, "
                  f"R: {class_metrics['recall']:.3f}, F1: {class_metrics['f1']:.3f}")
        print("="*70 + "\n")


class MetricsVisualizer:
    """Visualization tools for metrics"""
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names=None, save_path=None, title="Confusion Matrix"):
        """
        Plot confusion matrix heatmap
        
        Args:
            cm: Confusion matrix array
            class_names: Names of classes
            save_path: Path to save figure
            title: Plot title
        """
        if class_names is None:
            class_names = ['D00', 'D10', 'D20', 'D40']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    @staticmethod
    def plot_roc_curves(y_true, y_prob, class_names=None, save_path=None):
        """
        Plot ROC curves for multi-class classification
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            class_names: Names of classes
            save_path: Path to save figure
        """
        from sklearn.preprocessing import label_binarize
        
        if class_names is None:
            class_names = ['D00', 'D10', 'D20', 'D40']
        
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    @staticmethod
    def plot_precision_recall_curves(y_true, y_prob, class_names=None, save_path=None):
        """Plot Precision-Recall curves"""
        from sklearn.preprocessing import label_binarize
        
        if class_names is None:
            class_names = ['D00', 'D10', 'D20', 'D40']
        
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            plt.plot(recall, precision, lw=2, label=f'{class_names[i]}')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curves saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(results_dict, save_path=None):
        """
        Plot comparison of metrics across different models/methods
        
        Args:
            results_dict: Dict of {method_name: metrics_dict}
            save_path: Path to save figure
        """
        methods = list(results_dict.keys())
        metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            values = [results_dict[method].get(metric, 0) for method in methods]
            
            bars = axes[idx].bar(range(len(methods)), values, color='steelblue', alpha=0.7, edgecolor='black')
           
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}',
                             ha='center', va='bottom', fontsize=10)
            
            axes[idx].set_xticks(range(len(methods)))
            axes[idx].set_xticklabels(methods, rotation=45, ha='right')
            axes[idx].set_ylabel(label, fontsize=11)
            axes[idx].set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        else:
            plt.show()
        plt.close()


class StatisticalTests:
    """Statistical significance testing"""
    
    @staticmethod
    def mcnemar_test(y_true, y_pred1, y_pred2):
        """
        McNemar's test for comparing two classifiers
        
        Returns:
            p-value (< 0.05 means significant difference)
        """
        from statsmodels.stats.contingency_tables import mcnemar
        
        # Create contingency table
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # Both correct, both wrong, 1 correct 2 wrong, 1 wrong 2 correct
        n00 = np.sum(~correct1 & ~correct2)
        n01 = np.sum(~correct1 & correct2)
        n10 = np.sum(correct1 & ~correct2)
        n11 = np.sum(correct1 & correct2)
        
        table = [[n11, n10], [n01, n00]]
        result = mcnemar(table, exact=True)
        
        return {
            'statistic': float(result.statistic),
            'p_value': float(result.pvalue),
            'significant': result.pvalue < 0.05,
            'contingency_table': table
        }
    
    @staticmethod
    def paired_ttest(scores1, scores2):
        """
        Paired t-test for comparing performance across folds
        
        Args:
            scores1: Array of scores for method 1
            scores2: Array of scores for method 2
        
        Returns:
            T-test results
        """
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'mean_diff': float(np.mean(scores1) - np.mean(scores2))
        }
    
    @staticmethod
    def confidence_interval(scores, confidence=0.95):
        """
        Compute confidence interval for performance scores
        
        Args:
            scores: Array of scores
            confidence: Confidence level (default 0.95)
        
        Returns:
            Mean and confidence interval
        """
        mean = np.mean(scores)
        std_err = stats.sem(scores)
        margin = std_err * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
        
        return {
            'mean': float(mean),
            'std': float(np.std(scores)),
            'ci_lower': float(mean - margin),
            'ci_upper': float(mean + margin),
            'confidence': confidence
        }


def save_evaluation_results(metrics, save_dir, method_name="model"):
    """
    Save evaluation results to JSON
    
    Args:
        metrics: Dictionary of metrics
        save_dir: Directory to save results
        method_name: Name of the method being evaluated
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{method_name}_metrics.json')
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {save_path}")
    return save_path


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate dummy data for demonstration
    np.random.seed(42)
    n_samples = 100
    n_classes = 4
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_idx = np.random.choice(n_samples, size=20, replace=False)
    y_pred[error_idx] = np.random.randint(0, n_classes, 20)
    
    # Compute metrics
    metrics = ClassificationMetrics.compute_all_metrics(y_true, y_pred)
    ClassificationMetrics.print_metrics_summary(metrics)
    
    # Visualize
    cm = confusion_matrix(y_true, y_pred)
    MetricsVisualizer.plot_confusion_matrix(cm, save_path='demo_confusion_matrix.png')
    
    print("✓ Evaluation module working correctly")
