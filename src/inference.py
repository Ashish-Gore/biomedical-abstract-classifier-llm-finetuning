"""
Inference pipeline for cancer classification and disease extraction.
Combines model predictions with disease extraction for complete analysis.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from .disease_extraction import DiseaseExtractor
from .data_preprocessing import AbstractPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchPaperAnalyzer:
    """Complete pipeline for research paper analysis and classification."""
    
    def __init__(self, model_path: str, use_quantization: bool = True):
        self.model_path = model_path
        self.use_quantization = use_quantization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.preprocessor = AbstractPreprocessor()
        self.disease_extractor = DiseaseExtractor()
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"ResearchPaperAnalyzer initialized on {self.device}")
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=2,
                torch_dtype=torch.float16 if self.use_quantization else torch.float32
            )
            
            # Load LoRA weights if they exist
            try:
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                logger.info("Loaded LoRA fine-tuned model")
            except:
                self.model = base_model
                logger.info("Loaded base model (no LoRA weights found)")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text."""
        return self.preprocessor.clean_abstract(text)
    
    def classify_abstract(self, text: str) -> Dict[str, Any]:
        """Classify abstract as cancer or non-cancer."""
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
        
        # Convert to CPU and format
        probabilities = probabilities.cpu().numpy()[0]
        prediction = prediction.cpu().numpy()[0]
        
        # Map prediction to labels
        labels = ["Non-Cancer", "Cancer"]
        predicted_label = labels[prediction]
        
        # Calculate confidence scores
        confidence_scores = {
            "Non-Cancer": float(probabilities[0]),
            "Cancer": float(probabilities[1])
        }
        
        return {
            "predicted_labels": [predicted_label],
            "confidence_scores": confidence_scores,
            "prediction": prediction,
            "probabilities": probabilities.tolist()
        }
    
    def extract_diseases(self, text: str) -> Dict[str, Any]:
        """Extract diseases from abstract text."""
        return self.disease_extractor.extract_diseases(text)
    
    def analyze_abstract(self, text: str, abstract_id: str = None) -> Dict[str, Any]:
        """Complete analysis of a single abstract."""
        if abstract_id is None:
            abstract_id = f"abstract_{hash(text) % 10000}"
        
        logger.info(f"Analyzing abstract: {abstract_id}")
        
        # Clean text
        cleaned_text = self.preprocess_text(text)
        
        # Classify
        classification_result = self.classify_abstract(cleaned_text)
        
        # Extract diseases
        disease_result = self.extract_diseases(cleaned_text)
        
        # Combine results
        result = {
            "abstract_id": abstract_id,
            "original_text": text[:200] + "..." if len(text) > 200 else text,
            "cleaned_text": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
            "classification": classification_result,
            "disease_extraction": disease_result,
            "is_cancer_related": disease_result["cancer_count"] > 0,
            "analysis_metadata": {
                "text_length": len(text),
                "cleaned_length": len(cleaned_text),
                "disease_count": disease_result["total_diseases"],
                "cancer_disease_count": disease_result["cancer_count"],
                "non_cancer_disease_count": disease_result["non_cancer_count"]
            }
        }
        
        return result
    
    def analyze_batch(self, abstracts: List[str], abstract_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Analyze multiple abstracts in batch."""
        if abstract_ids is None:
            abstract_ids = [f"abstract_{i}" for i in range(len(abstracts))]
        
        results = []
        
        for i, (abstract, abstract_id) in enumerate(zip(abstracts, abstract_ids)):
            logger.info(f"Processing batch item {i+1}/{len(abstracts)}")
            result = self.analyze_abstract(abstract, abstract_id)
            results.append(result)
        
        return results
    
    def get_performance_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance summary from batch results."""
        total_abstracts = len(results)
        
        # Classification statistics
        cancer_predictions = sum(1 for r in results if r["classification"]["predicted_labels"][0] == "Cancer")
        non_cancer_predictions = total_abstracts - cancer_predictions
        
        # Disease extraction statistics
        total_diseases = sum(r["disease_extraction"]["total_diseases"] for r in results)
        cancer_diseases = sum(r["disease_extraction"]["cancer_count"] for r in results)
        non_cancer_diseases = sum(r["disease_extraction"]["non_cancer_count"] for r in results)
        
        # Confidence statistics
        cancer_confidences = [r["classification"]["confidence_scores"]["Cancer"] for r in results]
        non_cancer_confidences = [r["classification"]["confidence_scores"]["Non-Cancer"] for r in results]
        
        return {
            "total_abstracts": total_abstracts,
            "classification_summary": {
                "cancer_predictions": cancer_predictions,
                "non_cancer_predictions": non_cancer_predictions,
                "cancer_percentage": (cancer_predictions / total_abstracts) * 100,
                "avg_cancer_confidence": np.mean(cancer_confidences),
                "avg_non_cancer_confidence": np.mean(non_cancer_confidences)
            },
            "disease_extraction_summary": {
                "total_diseases_extracted": total_diseases,
                "cancer_diseases": cancer_diseases,
                "non_cancer_diseases": non_cancer_diseases,
                "avg_diseases_per_abstract": total_diseases / total_abstracts,
                "abstracts_with_diseases": sum(1 for r in results if r["disease_extraction"]["total_diseases"] > 0)
            }
        }


def load_analyzer(model_path: str, use_quantization: bool = True) -> ResearchPaperAnalyzer:
    """Load a pre-trained analyzer."""
    return ResearchPaperAnalyzer(model_path, use_quantization)


def main():
    """Main function for testing the inference pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze research paper abstracts')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--input', help='Input text file or CSV with abstracts')
    parser.add_argument('--text', help='Single abstract text to analyze')
    parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Load analyzer
    analyzer = load_analyzer(args.model_path)
    
    if args.text:
        # Analyze single text
        result = analyzer.analyze_abstract(args.text)
        print(json.dumps(result, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    
    elif args.input:
        # Analyze from file
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
            abstracts = df['cleaned_abstract'].tolist()
            abstract_ids = df.get('pmid', [f"abstract_{i}" for i in range(len(abstracts))]).tolist()
        else:
            with open(args.input, 'r') as f:
                abstracts = [f.read()]
            abstract_ids = ["abstract_0"]
        
        results = analyzer.analyze_batch(abstracts, abstract_ids)
        
        # Generate summary
        summary = analyzer.get_performance_summary(results)
        
        print("Analysis Summary:")
        print(json.dumps(summary, indent=2))
        
        if args.output:
            output_data = {
                "summary": summary,
                "results": results
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
    
    else:
        print("Please provide either --text or --input argument")


if __name__ == "__main__":
    main()
