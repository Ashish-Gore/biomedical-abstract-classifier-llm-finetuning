"""
Complete training and evaluation script.
Trains both baseline and fine-tuned models and compares their performance.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import AbstractPreprocessor
from src.model_training import CancerClassifier
from src.evaluation import ModelEvaluator
from src.inference import ResearchPaperAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data(output_path: str, num_samples: int = 1000):
    """Create sample data for demonstration purposes."""
    logger.info(f"Creating sample data with {num_samples} abstracts...")
    
    # Sample cancer abstracts
    cancer_abstracts = [
        "This study investigates the efficacy of chemotherapy in treating lung cancer patients. We analyzed 200 patients with stage III non-small cell lung cancer and found significant improvement in survival rates.",
        "Breast cancer is one of the most common malignancies in women. Our research focuses on early detection methods using mammography and genetic screening.",
        "Prostate cancer treatment options include surgery, radiation therapy, and hormone therapy. This paper presents a comparative analysis of treatment outcomes.",
        "Colorectal cancer screening programs have shown significant impact on mortality rates. We evaluated the effectiveness of different screening modalities.",
        "Pancreatic cancer is known for its poor prognosis. This study examines novel therapeutic approaches including immunotherapy and targeted therapy.",
        "Ovarian cancer often presents at advanced stages. Our research investigates biomarkers for early detection and personalized treatment strategies.",
        "Brain cancer treatment remains challenging due to the blood-brain barrier. This paper explores novel drug delivery systems for glioblastoma therapy.",
        "Liver cancer, particularly hepatocellular carcinoma, is a major health concern. We studied the role of viral hepatitis in cancer development.",
        "Kidney cancer, or renal cell carcinoma, has shown promising responses to targeted therapies. This study evaluates the efficacy of new treatment protocols.",
        "Bladder cancer is the most common malignancy of the urinary tract. Our research focuses on non-invasive diagnostic methods and treatment optimization."
    ]
    
    # Sample non-cancer abstracts
    non_cancer_abstracts = [
        "This study examines the relationship between diabetes and cardiovascular disease. We analyzed data from 500 patients over a 5-year period.",
        "Hypertension is a major risk factor for stroke and heart disease. Our research investigates lifestyle interventions for blood pressure management.",
        "Alzheimer's disease is the most common form of dementia. This paper presents findings on early cognitive assessment and intervention strategies.",
        "Parkinson's disease affects motor function and quality of life. We studied the effectiveness of physical therapy and medication combinations.",
        "Depression and anxiety are common mental health conditions. This research evaluates the efficacy of cognitive behavioral therapy approaches.",
        "Asthma is a chronic respiratory condition affecting millions worldwide. Our study examines environmental triggers and treatment optimization.",
        "COPD, or chronic obstructive pulmonary disease, is primarily caused by smoking. This paper presents findings on smoking cessation programs.",
        "Pneumonia is a common respiratory infection. We investigated the effectiveness of different antibiotic treatment protocols.",
        "Tuberculosis remains a global health concern. This study examines drug resistance patterns and treatment outcomes in different populations.",
        "Influenza vaccination programs are crucial for public health. Our research evaluates vaccine effectiveness and coverage rates."
    ]
    
    # Generate more samples by varying the text
    all_abstracts = []
    all_labels = []
    all_pmids = []
    
    # Generate cancer abstracts
    for i in range(num_samples // 2):
        base_abstract = cancer_abstracts[i % len(cancer_abstracts)]
        # Add some variation
        variation = f" Study ID: {i+1}. " + base_abstract + f" Additional analysis shows promising results in cohort {i+1}."
        all_abstracts.append(variation)
        all_labels.append(1)  # Cancer
        all_pmids.append(f"PMID_{i+1:06d}")
    
    # Generate non-cancer abstracts
    for i in range(num_samples // 2, num_samples):
        base_abstract = non_cancer_abstracts[(i - num_samples // 2) % len(non_cancer_abstracts)]
        # Add some variation
        variation = f" Study ID: {i+1}. " + base_abstract + f" Additional analysis shows promising results in cohort {i+1}."
        all_abstracts.append(variation)
        all_labels.append(0)  # Non-cancer
        all_pmids.append(f"PMID_{i+1:06d}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'pmid': all_pmids,
        'abstract': all_abstracts,
        'label': all_labels
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Sample data saved to {output_path}")
    
    return df


def train_baseline_model(data_path: str, output_dir: str) -> str:
    """Train baseline model without fine-tuning."""
    logger.info("Training baseline model...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Preprocess data
    preprocessor = AbstractPreprocessor()
    df['cleaned_abstract'] = df['abstract'].apply(preprocessor.clean_abstract)
    df['encoded_label'] = df['label']
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['encoded_label'])
    
    # Initialize classifier
    classifier = CancerClassifier(
        model_name="microsoft/DialoGPT-medium",
        use_quantization=True
    )
    
    # Load model
    classifier.load_model(num_labels=2)
    
    # Prepare datasets
    train_dataset, test_dataset = classifier.prepare_datasets(train_df, test_df)
    
    # Train with minimal epochs for baseline
    trainer = classifier.train(
        train_dataset, test_dataset,
        output_dir=f"{output_dir}/baseline",
        num_epochs=1,  # Minimal training for baseline
        batch_size=8
    )
    
    # Evaluate baseline
    baseline_results = classifier.evaluate(test_dataset)
    
    # Save baseline results
    baseline_path = f"{output_dir}/baseline_results.json"
    with open(baseline_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    logger.info(f"Baseline model training completed. Results saved to {baseline_path}")
    return baseline_path


def train_finetuned_model(data_path: str, output_dir: str) -> str:
    """Train fine-tuned model with LoRA."""
    logger.info("Training fine-tuned model...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Preprocess data
    preprocessor = AbstractPreprocessor()
    df['cleaned_abstract'] = df['abstract'].apply(preprocessor.clean_abstract)
    df['encoded_label'] = df['label']
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['encoded_label'])
    
    # Initialize classifier
    classifier = CancerClassifier(
        model_name="microsoft/DialoGPT-medium",
        use_quantization=True
    )
    
    # Load model
    classifier.load_model(num_labels=2)
    
    # Prepare datasets
    train_dataset, test_dataset = classifier.prepare_datasets(train_df, test_df)
    
    # Train with more epochs for fine-tuning
    trainer = classifier.train(
        train_dataset, test_dataset,
        output_dir=f"{output_dir}/finetuned",
        num_epochs=3,  # More training for fine-tuning
        batch_size=8
    )
    
    # Evaluate fine-tuned model
    finetuned_results = classifier.evaluate(test_dataset)
    
    # Save fine-tuned results
    finetuned_path = f"{output_dir}/finetuned_results.json"
    with open(finetuned_path, 'w') as f:
        json.dump(finetuned_results, f, indent=2)
    
    logger.info(f"Fine-tuned model training completed. Results saved to {finetuned_path}")
    return finetuned_path


def compare_models(baseline_path: str, finetuned_path: str, output_dir: str):
    """Compare baseline and fine-tuned models."""
    logger.info("Comparing models...")
    
    # Load results
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    
    with open(finetuned_path, 'r') as f:
        finetuned_results = json.load(f)
    
    # Extract data for comparison
    true_labels = baseline_results['predictions']  # Assuming same test set
    baseline_predictions = baseline_results['predictions']
    baseline_probabilities = baseline_results['probabilities']
    finetuned_predictions = finetuned_results['predictions']
    finetuned_probabilities = finetuned_results['probabilities']
    
    # Run evaluation
    evaluator = ModelEvaluator()
    comparison_results = evaluator.evaluate_baseline_vs_finetuned(
        baseline_predictions,
        baseline_probabilities,
        finetuned_predictions,
        finetuned_probabilities,
        true_labels
    )
    
    # Generate comprehensive report
    report_path = evaluator.generate_report(f"{output_dir}/comparison")
    
    logger.info(f"Model comparison completed. Report saved to {report_path}")
    return report_path


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train and evaluate cancer classification models')
    parser.add_argument('--data_path', help='Path to input data CSV file')
    parser.add_argument('--output_dir', default='./training_results', help='Output directory for results')
    parser.add_argument('--create_sample_data', action='store_true', help='Create sample data for demonstration')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of sample abstracts to create')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data if requested
    if args.create_sample_data or not args.data_path:
        data_path = output_dir / "sample_data.csv"
        create_sample_data(str(data_path), args.num_samples)
        args.data_path = str(data_path)
    
    try:
        # Train baseline model
        baseline_path = train_baseline_model(args.data_path, str(output_dir))
        
        # Train fine-tuned model
        finetuned_path = train_finetuned_model(args.data_path, str(output_dir))
        
        # Compare models
        comparison_path = compare_models(baseline_path, finetuned_path, str(output_dir))
        
        print("\n" + "="*50)
        print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Results saved to: {output_dir}")
        print(f"Comparison report: {comparison_path}")
        print("\nTo start the API server, run:")
        print(f"python -m src.api.main --model_path {output_dir}/finetuned")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
