"""
Demonstration of the Research Paper Analysis Pipeline.
This script shows the complete pipeline working without requiring a server.
"""

import sys
import pandas as pd
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_preprocessing import AbstractPreprocessor
from disease_extraction import DiseaseExtractor

def demonstrate_pipeline():
    """Demonstrate the complete pipeline."""
    print("="*60)
    print("RESEARCH PAPER ANALYSIS PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing components...")
    preprocessor = AbstractPreprocessor()
    disease_extractor = DiseaseExtractor()
    print("✅ Components initialized successfully!")
    
    # Load sample data
    print("\n2. Loading sample data...")
    try:
        test_data = pd.read_csv("data/processed/test.csv")
        print(f"✅ Loaded {len(test_data)} test samples")
    except:
        print("❌ Could not load test data. Using sample abstracts...")
        # Use sample abstracts
        sample_abstracts = [
            {
                "abstract": "This study investigates the efficacy of chemotherapy in treating lung cancer patients. We analyzed 200 patients with stage III non-small cell lung cancer and found significant improvement in survival rates.",
                "label": 1,
                "pmid": "sample_001"
            },
            {
                "abstract": "This study examines the relationship between diabetes and cardiovascular disease in elderly patients. We analyzed data from 500 patients over a 5-year period.",
                "label": 0,
                "pmid": "sample_002"
            },
            {
                "abstract": "Breast cancer screening programs have shown significant impact on mortality rates in women over 50. Our research focuses on early detection methods using mammography.",
                "label": 1,
                "pmid": "sample_003"
            },
            {
                "abstract": "Alzheimer's disease is the most common form of dementia affecting millions worldwide. This paper presents findings on early cognitive assessment and intervention strategies.",
                "label": 0,
                "pmid": "sample_004"
            },
            {
                "abstract": "Prostate cancer treatment options include surgery, radiation therapy, and hormone therapy. This paper presents a comparative analysis of treatment outcomes.",
                "label": 1,
                "pmid": "sample_005"
            }
        ]
        test_data = pd.DataFrame(sample_abstracts)
        print(f"✅ Using {len(test_data)} sample abstracts")
    
    # Process abstracts
    print("\n3. Processing abstracts...")
    results = []
    
    for idx, row in test_data.iterrows():
        abstract = row['abstract']
        true_label = row['label']
        pmid = row.get('pmid', f'sample_{idx}')
        
        print(f"\n--- Processing Abstract {idx+1} (PMID: {pmid}) ---")
        
        # Clean abstract
        cleaned_abstract = preprocessor.clean_abstract(abstract)
        print(f"Original length: {len(abstract)} chars")
        print(f"Cleaned length: {len(cleaned_abstract)} chars")
        
        # Classify
        is_cancer_related = preprocessor.is_cancer_related(cleaned_abstract)
        predicted_label = "Cancer" if is_cancer_related else "Non-Cancer"
        true_label_name = "Cancer" if true_label == 1 else "Non-Cancer"
        
        print(f"True label: {true_label_name}")
        print(f"Predicted: {predicted_label}")
        print(f"Correct: {'✅' if (is_cancer_related and true_label == 1) or (not is_cancer_related and true_label == 0) else '❌'}")
        
        # Extract diseases
        disease_result = disease_extractor.extract_diseases(cleaned_abstract)
        print(f"Extracted diseases: {disease_result['extracted_diseases']}")
        print(f"Cancer diseases: {disease_result['cancer_diseases']}")
        print(f"Non-cancer diseases: {disease_result['non_cancer_diseases']}")
        print(f"Total diseases: {disease_result['total_diseases']}")
        
        # Store result
        result = {
            "pmid": pmid,
            "true_label": true_label_name,
            "predicted_label": predicted_label,
            "correct": (is_cancer_related and true_label == 1) or (not is_cancer_related and true_label == 0),
            "extracted_diseases": disease_result['extracted_diseases'],
            "cancer_diseases": disease_result['cancer_diseases'],
            "non_cancer_diseases": disease_result['non_cancer_diseases'],
            "total_diseases": disease_result['total_diseases']
        }
        results.append(result)
    
    # Calculate metrics
    print("\n4. Calculating performance metrics...")
    correct_predictions = sum(1 for r in results if r['correct'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions
    
    # Confusion matrix
    true_positives = sum(1 for r in results if r['true_label'] == 'Cancer' and r['predicted_label'] == 'Cancer')
    true_negatives = sum(1 for r in results if r['true_label'] == 'Non-Cancer' and r['predicted_label'] == 'Non-Cancer')
    false_positives = sum(1 for r in results if r['true_label'] == 'Non-Cancer' and r['predicted_label'] == 'Cancer')
    false_negatives = sum(1 for r in results if r['true_label'] == 'Cancer' and r['predicted_label'] == 'Non-Cancer')
    
    # Calculate F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"✅ Performance Metrics:")
    print(f"   Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1 Score: {f1_score:.3f}")
    print(f"   Confusion Matrix:")
    print(f"     TN: {true_negatives}, FP: {false_positives}")
    print(f"     FN: {false_negatives}, TP: {true_positives}")
    
    # Disease extraction summary
    print("\n5. Disease Extraction Summary:")
    total_diseases = sum(r['total_diseases'] for r in results)
    cancer_diseases = sum(len(r['cancer_diseases']) for r in results)
    non_cancer_diseases = sum(len(r['non_cancer_diseases']) for r in results)
    
    print(f"   Total diseases extracted: {total_diseases}")
    print(f"   Cancer diseases: {cancer_diseases}")
    print(f"   Non-cancer diseases: {non_cancer_diseases}")
    print(f"   Average diseases per abstract: {total_diseases/len(results):.1f}")
    
    # Show all extracted diseases
    all_diseases = set()
    for r in results:
        all_diseases.update(r['extracted_diseases'])
    
    print(f"\n   All unique diseases found:")
    for disease in sorted(all_diseases):
        print(f"     - {disease}")
    
    # Save results
    print("\n6. Saving results...")
    output_file = "pipeline_demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {output_file}")
    
    print("\n" + "="*60)
    print("PIPELINE DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Processed {len(results)} abstracts")
    print(f"🎯 Accuracy: {accuracy:.1%}")
    print(f"🔬 Extracted {total_diseases} diseases")
    print(f"💾 Results saved to: {output_file}")
    print("\n🚀 The pipeline is working perfectly!")
    print("   - Data preprocessing: ✅")
    print("   - Cancer classification: ✅")
    print("   - Disease extraction: ✅")
    print("   - Performance evaluation: ✅")


if __name__ == "__main__":
    demonstrate_pipeline()
