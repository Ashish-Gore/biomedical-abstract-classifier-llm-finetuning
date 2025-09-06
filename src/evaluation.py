"""
Evaluation pipeline for model performance assessment.
Generates confusion matrices, metrics, and performance comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, List, Any, Tuple, Optional
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with visualization capabilities."""
    
    def __init__(self):
        self.results = {}
        self.metrics = {}
    
    def evaluate_baseline_vs_finetuned(
        self, 
        baseline_predictions: List[int], 
        baseline_probabilities: List[List[float]],
        finetuned_predictions: List[int], 
        finetuned_probabilities: List[List[float]],
        true_labels: List[int],
        model_names: List[str] = ["Baseline", "Fine-tuned"]
    ) -> Dict[str, Any]:
        """Compare baseline and fine-tuned model performance."""
        
        logger.info("Evaluating baseline vs fine-tuned models...")
        
        # Calculate metrics for both models
        baseline_metrics = self._calculate_metrics(true_labels, baseline_predictions, baseline_probabilities)
        finetuned_metrics = self._calculate_metrics(true_labels, finetuned_predictions, finetuned_probabilities)
        
        # Store results
        self.results = {
            "baseline": {
                "predictions": baseline_predictions,
                "probabilities": baseline_probabilities,
                "metrics": baseline_metrics
            },
            "finetuned": {
                "predictions": finetuned_predictions,
                "probabilities": finetuned_probabilities,
                "metrics": finetuned_metrics
            },
            "true_labels": true_labels
        }
        
        # Calculate improvements
        improvements = self._calculate_improvements(baseline_metrics, finetuned_metrics)
        
        # Generate comparison report
        comparison_report = {
            "baseline_performance": baseline_metrics,
            "finetuned_performance": finetuned_metrics,
            "improvements": improvements,
            "summary": self._generate_summary(baseline_metrics, finetuned_metrics, improvements)
        }
        
        self.metrics = comparison_report
        return comparison_report
    
    def _calculate_metrics(self, true_labels: List[int], predictions: List[int], 
                          probabilities: List[List[float]]) -> Dict[str, float]:
        """Calculate comprehensive metrics for a model."""
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if probabilities and len(probabilities[0]) == 2:
            try:
                roc_auc = roc_auc_score(true_labels, [p[1] for p in probabilities])
            except:
                roc_auc = None
        
        # Per-class metrics
        precision_per_class = precision_score(true_labels, predictions, average=None)
        recall_per_class = recall_score(true_labels, predictions, average=None)
        f1_per_class = f1_score(true_labels, predictions, average=None)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist()
        }
    
    def _calculate_improvements(self, baseline: Dict, finetuned: Dict) -> Dict[str, float]:
        """Calculate performance improvements."""
        improvements = {}
        
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            if metric in baseline and metric in finetuned:
                improvement = finetuned[metric] - baseline[metric]
                improvement_pct = (improvement / baseline[metric]) * 100
                improvements[f"{metric}_improvement"] = improvement
                improvements[f"{metric}_improvement_pct"] = improvement_pct
        
        return improvements
    
    def _generate_summary(self, baseline: Dict, finetuned: Dict, improvements: Dict) -> str:
        """Generate a human-readable summary of the comparison."""
        
        summary_parts = []
        
        # Overall performance
        summary_parts.append("=== MODEL PERFORMANCE COMPARISON ===\n")
        
        summary_parts.append("Baseline Model Performance:")
        summary_parts.append(f"  Accuracy: {baseline['accuracy']:.3f}")
        summary_parts.append(f"  F1-Score: {baseline['f1_score']:.3f}")
        summary_parts.append(f"  Precision: {baseline['precision']:.3f}")
        summary_parts.append(f"  Recall: {baseline['recall']:.3f}\n")
        
        summary_parts.append("Fine-tuned Model Performance:")
        summary_parts.append(f"  Accuracy: {finetuned['accuracy']:.3f}")
        summary_parts.append(f"  F1-Score: {finetuned['f1_score']:.3f}")
        summary_parts.append(f"  Precision: {finetuned['precision']:.3f}")
        summary_parts.append(f"  Recall: {finetuned['recall']:.3f}\n")
        
        summary_parts.append("Performance Improvements:")
        for metric in ["accuracy", "f1_score", "precision", "recall"]:
            if f"{metric}_improvement_pct" in improvements:
                pct = improvements[f"{metric}_improvement_pct"]
                summary_parts.append(f"  {metric.title()}: {pct:+.1f}%")
        
        return "\n".join(summary_parts)
    
    def plot_confusion_matrices(self, save_path: Optional[str] = None) -> go.Figure:
        """Create interactive confusion matrix plots."""
        
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Baseline Model", "Fine-tuned Model"),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # Plot baseline confusion matrix
        baseline_cm = np.array(self.results["baseline"]["metrics"]["confusion_matrix"])
        fig.add_trace(
            go.Heatmap(
                z=baseline_cm,
                x=["Predicted Non-Cancer", "Predicted Cancer"],
                y=["Actual Non-Cancer", "Actual Cancer"],
                text=baseline_cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="Blues",
                showscale=False
            ),
            row=1, col=1
        )
        
        # Plot fine-tuned confusion matrix
        finetuned_cm = np.array(self.results["finetuned"]["metrics"]["confusion_matrix"])
        fig.add_trace(
            go.Heatmap(
                z=finetuned_cm,
                x=["Predicted Non-Cancer", "Predicted Cancer"],
                y=["Actual Non-Cancer", "Actual Cancer"],
                text=finetuned_cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="Greens",
                showscale=False
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Confusion Matrix Comparison: Baseline vs Fine-tuned Model",
            height=400,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self, save_path: Optional[str] = None) -> go.Figure:
        """Create metrics comparison bar chart."""
        
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluation first.")
        
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        baseline_values = [self.metrics["baseline_performance"][m] for m in metrics]
        finetuned_values = [self.metrics["finetuned_performance"][m] for m in metrics]
        
        fig = go.Figure(data=[
            go.Bar(name="Baseline", x=metrics, y=baseline_values, marker_color="lightblue"),
            go.Bar(name="Fine-tuned", x=metrics, y=finetuned_values, marker_color="lightgreen")
        ])
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode="group",
            yaxis=dict(range=[0, 1])
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Metrics comparison plot saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, save_path: Optional[str] = None) -> go.Figure:
        """Plot ROC curves for both models."""
        
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        fig = go.Figure()
        
        true_labels = self.results["true_labels"]
        
        # Plot baseline ROC curve
        baseline_probs = [p[1] for p in self.results["baseline"]["probabilities"]]
        if baseline_probs:
            fpr, tpr, _ = roc_curve(true_labels, baseline_probs)
            auc = roc_auc_score(true_labels, baseline_probs)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'Baseline (AUC = {auc:.3f})',
                line=dict(color='blue')
            ))
        
        # Plot fine-tuned ROC curve
        finetuned_probs = [p[1] for p in self.results["finetuned"]["probabilities"]]
        if finetuned_probs:
            fpr, tpr, _ = roc_curve(true_labels, finetuned_probs)
            auc = roc_auc_score(true_labels, finetuned_probs)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'Fine-tuned (AUC = {auc:.3f})',
                line=dict(color='green')
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="ROC Curves: Baseline vs Fine-tuned Model",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"ROC curves plot saved to {save_path}")
        
        return fig
    
    def generate_report(self, output_dir: str = "./evaluation_results") -> str:
        """Generate comprehensive evaluation report."""
        
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluation first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = output_path / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Generate plots
        self.plot_confusion_matrices(str(output_path / "confusion_matrices.html"))
        self.plot_metrics_comparison(str(output_path / "metrics_comparison.html"))
        self.plot_roc_curves(str(output_path / "roc_curves.html"))
        
        # Generate text report
        report_file = output_path / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(self.metrics["summary"])
            f.write(f"\n\nDetailed metrics saved to: {metrics_file}")
            f.write(f"\nVisualizations saved to: {output_path}")
        
        logger.info(f"Evaluation report generated in {output_path}")
        return str(output_path)


def main():
    """Main function for running evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--baseline_predictions', required=True, help='Baseline model predictions file')
    parser.add_argument('--finetuned_predictions', required=True, help='Fine-tuned model predictions file')
    parser.add_argument('--true_labels', required=True, help='True labels file')
    parser.add_argument('--output_dir', default='./evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    baseline_preds = np.load(args.baseline_predictions)
    finetuned_preds = np.load(args.finetuned_predictions)
    true_labels = np.load(args.true_labels)
    
    # Load probabilities if available
    baseline_probs = None
    finetuned_probs = None
    
    try:
        baseline_probs = np.load(args.baseline_predictions.replace('.npy', '_probs.npy'))
        finetuned_probs = np.load(args.finetuned_predictions.replace('.npy', '_probs.npy'))
    except:
        logger.warning("Probability files not found, skipping ROC analysis")
    
    # Run evaluation
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_baseline_vs_finetuned(
        baseline_preds.tolist(),
        baseline_probs.tolist() if baseline_probs is not None else None,
        finetuned_preds.tolist(),
        finetuned_probs.tolist() if finetuned_probs is not None else None,
        true_labels.tolist()
    )
    
    # Generate report
    output_path = evaluator.generate_report(args.output_dir)
    print(f"Evaluation completed! Results saved to {output_path}")


if __name__ == "__main__":
    main()
