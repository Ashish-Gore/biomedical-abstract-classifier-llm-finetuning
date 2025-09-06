"""
Simplified training script for demonstration purposes.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import AbstractPreprocessor
from src.disease_extraction import DiseaseExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_models():
    """Create demo models and results for demonstration."""
    logger.info("Creating demo models and results...")
    
    # Load processed data
    train_data = pd.read_csv("data/processed/train.csv")
    test_data = pd.read_csv("data/processed/test.csv")
    
    logger.info(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    # Initialize components
    preprocessor = AbstractPreprocessor()
    disease_extractor = DiseaseExtractor()
    
    # Create demo results
    demo_results = []
    
    # Process a few samples for demonstration
    sample_size = min(50, len(test_data))
    test_sample = test_data.sample(n=sample_size, random_state=42)
    
    logger.info(f"Processing {sample_size} samples for demonstration...")
    
    for idx, row in test_sample.iterrows():
        abstract = row['cleaned_abstract']
        true_label = row['encoded_label']
        
        # Simulate classification (since we don't have a trained model)
        # In a real scenario, this would be the actual model prediction
        is_cancer_related = preprocessor.is_cancer_related(abstract)
        predicted_label = 1 if is_cancer_related else 0
        
        # Extract diseases
        disease_result = disease_extractor.extract_diseases(abstract)
        
        # Calculate confidence scores (simulated)
        cancer_confidence = 0.8 if predicted_label == 1 else 0.2
        non_cancer_confidence = 0.8 if predicted_label == 0 else 0.2
        
        result = {
            "abstract_id": f"demo_{idx}",
            "original_text": abstract[:200] + "..." if len(abstract) > 200 else abstract,
            "cleaned_text": abstract,
            "classification": {
                "predicted_labels": ["Cancer" if predicted_label == 1 else "Non-Cancer"],
                "confidence_scores": {
                    "Cancer": cancer_confidence,
                    "Non-Cancer": non_cancer_confidence
                },
                "prediction": predicted_label,
                "probabilities": [non_cancer_confidence, cancer_confidence]
            },
            "disease_extraction": disease_result,
            "is_cancer_related": disease_result["cancer_count"] > 0,
            "analysis_metadata": {
                "text_length": len(abstract),
                "cleaned_length": len(abstract),
                "disease_count": disease_result["total_diseases"],
                "cancer_disease_count": disease_result["cancer_count"],
                "non_cancer_disease_count": disease_result["non_cancer_count"]
            },
            "true_label": true_label,
            "correct": predicted_label == true_label
        }
        
        demo_results.append(result)
    
    # Calculate metrics
    correct_predictions = sum(1 for r in demo_results if r["correct"])
    accuracy = correct_predictions / len(demo_results)
    
    # Calculate confusion matrix
    true_positives = sum(1 for r in demo_results if r["true_label"] == 1 and r["classification"]["predicted_labels"][0] == "Cancer")
    true_negatives = sum(1 for r in demo_results if r["true_label"] == 0 and r["classification"]["predicted_labels"][0] == "Non-Cancer")
    false_positives = sum(1 for r in demo_results if r["true_label"] == 0 and r["classification"]["predicted_labels"][0] == "Cancer")
    false_negatives = sum(1 for r in demo_results if r["true_label"] == 1 and r["classification"]["predicted_labels"][0] == "Non-Cancer")
    
    confusion_matrix = [[true_negatives, false_positives], [false_negatives, true_positives]]
    
    # Calculate F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create demo model directory
    model_dir = Path("demo_models")
    model_dir.mkdir(exist_ok=True)
    
    # Save demo results
    import json
    with open(model_dir / "demo_results.json", "w") as f:
        json.dump(demo_results, f, indent=2)
    
    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": confusion_matrix,
        "total_samples": len(demo_results),
        "correct_predictions": correct_predictions,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
    
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Demo results saved to {model_dir}")
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"F1 Score: {f1_score:.3f}")
    logger.info(f"Confusion Matrix: {confusion_matrix}")
    
    return demo_results, metrics


def main():
    """Main function."""
    logger.info("Starting demo model creation...")
    
    # Check if processed data exists
    if not os.path.exists("data/processed/train.csv"):
        logger.error("Processed data not found. Please run data preprocessing first.")
        return
    
    # Create demo models
    results, metrics = create_demo_models()
    
    print("\n" + "="*50)
    print("DEMO MODEL CREATION COMPLETED!")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"Confusion Matrix:")
    print(f"  TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}")
    print(f"  FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}")
    print(f"\nResults saved to: demo_models/")
    print("\nTo start the API server, run:")
    print("python -m src.api.main --model_path demo_models")


if __name__ == "__main__":
    main()
