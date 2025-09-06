"""
Disease extraction module using Named Entity Recognition and LLM-based extraction.
Extracts specific diseases mentioned in research paper abstracts.
"""

import spacy
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from transformers import pipeline
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseaseExtractor:
    """Extracts disease information from research paper abstracts."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = None
        self.ner_pipeline = None
        self.disease_patterns = self._load_disease_patterns()
        self.cancer_specific_patterns = self._load_cancer_patterns()
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Install with: python -m spacy download {model_name}")
        
        # Initialize NER pipeline
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-pubmed",
                aggregation_strategy="simple"
            )
            logger.info("Loaded BERT NER pipeline for medical entities")
        except Exception as e:
            logger.warning(f"Could not load BERT NER pipeline: {e}")
    
    def _load_disease_patterns(self) -> List[Dict[str, str]]:
        """Load comprehensive disease patterns for extraction."""
        return [
            # Cancer types
            {"pattern": r'\b(?:lung|pulmonary)\s+cancer\b', "disease": "Lung Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:breast)\s+cancer\b', "disease": "Breast Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:prostate)\s+cancer\b', "disease": "Prostate Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:colorectal|colon|rectal)\s+cancer\b', "disease": "Colorectal Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:pancreatic)\s+cancer\b', "disease": "Pancreatic Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:ovarian)\s+cancer\b', "disease": "Ovarian Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:brain|cerebral)\s+cancer\b', "disease": "Brain Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:liver|hepatic)\s+cancer\b', "disease": "Liver Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:kidney|renal)\s+cancer\b', "disease": "Kidney Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:bladder)\s+cancer\b', "disease": "Bladder Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:cervical)\s+cancer\b', "disease": "Cervical Cancer", "category": "Cancer"},
            {"pattern": r'\b(?:endometrial)\s+cancer\b', "disease": "Endometrial Cancer", "category": "Cancer"},
            
            # Blood cancers
            {"pattern": r'\b(?:leukemia|leukaemia)\b', "disease": "Leukemia", "category": "Cancer"},
            {"pattern": r'\b(?:lymphoma)\b', "disease": "Lymphoma", "category": "Cancer"},
            {"pattern": r'\b(?:myeloma)\b', "disease": "Myeloma", "category": "Cancer"},
            
            # Other cancers
            {"pattern": r'\b(?:melanoma)\b', "disease": "Melanoma", "category": "Cancer"},
            {"pattern": r'\b(?:sarcoma)\b', "disease": "Sarcoma", "category": "Cancer"},
            {"pattern": r'\b(?:carcinoma)\b', "disease": "Carcinoma", "category": "Cancer"},
            
            # Non-cancer diseases
            {"pattern": r'\b(?:diabetes|diabetic)\b', "disease": "Diabetes", "category": "Metabolic"},
            {"pattern": r'\b(?:hypertension|high blood pressure)\b', "disease": "Hypertension", "category": "Cardiovascular"},
            {"pattern": r'\b(?:cardiovascular disease|heart disease)\b', "disease": "Cardiovascular Disease", "category": "Cardiovascular"},
            {"pattern": r'\b(?:stroke|cerebrovascular)\b', "disease": "Stroke", "category": "Cardiovascular"},
            {"pattern": r'\b(?:alzheimer|alzheimer\'s)\b', "disease": "Alzheimer's Disease", "category": "Neurological"},
            {"pattern": r'\b(?:parkinson|parkinson\'s)\b', "disease": "Parkinson's Disease", "category": "Neurological"},
            {"pattern": r'\b(?:dementia)\b', "disease": "Dementia", "category": "Neurological"},
            {"pattern": r'\b(?:depression|depressive)\b', "disease": "Depression", "category": "Mental Health"},
            {"pattern": r'\b(?:anxiety)\b', "disease": "Anxiety", "category": "Mental Health"},
            {"pattern": r'\b(?:asthma)\b', "disease": "Asthma", "category": "Respiratory"},
            {"pattern": r'\b(?:copd|chronic obstructive pulmonary)\b', "disease": "COPD", "category": "Respiratory"},
            {"pattern": r'\b(?:pneumonia)\b', "disease": "Pneumonia", "category": "Respiratory"},
            {"pattern": r'\b(?:tuberculosis|tb)\b', "disease": "Tuberculosis", "category": "Infectious"},
            {"pattern": r'\b(?:influenza|flu)\b', "disease": "Influenza", "category": "Infectious"},
            {"pattern": r'\b(?:covid-19|coronavirus|sars-cov-2)\b', "disease": "COVID-19", "category": "Infectious"},
        ]
    
    def _load_cancer_patterns(self) -> List[str]:
        """Load cancer-specific patterns for validation."""
        return [
            'cancer', 'carcinoma', 'tumor', 'neoplasm', 'malignancy', 
            'oncology', 'metastasis', 'chemotherapy', 'radiotherapy',
            'surgery', 'biopsy', 'prognosis', 'survival'
        ]
    
    def extract_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """Extract diseases using regex patterns."""
        diseases = []
        text_lower = text.lower()
        
        for pattern_info in self.disease_patterns:
            matches = re.finditer(pattern_info["pattern"], text_lower, re.IGNORECASE)
            for match in matches:
                diseases.append({
                    "disease": pattern_info["disease"],
                    "category": pattern_info["category"],
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,  # High confidence for regex matches
                    "method": "regex"
                })
        
        return diseases
    
    def extract_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract diseases using spaCy NER."""
        diseases = []
        
        if self.nlp is None:
            return diseases
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            # Check if entity is disease-related
            if ent.label_ in ["DISEASE", "MEDICAL_CONDITION", "SYMPTOM"]:
                # Additional filtering for disease-related terms
                if any(keyword in ent.text.lower() for keyword in 
                      ['cancer', 'disease', 'syndrome', 'disorder', 'condition']):
                    diseases.append({
                        "disease": ent.text.title(),
                        "category": "Medical Condition",
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.7,
                        "method": "spacy"
                    })
        
        return diseases
    
    def extract_with_bert(self, text: str) -> List[Dict[str, Any]]:
        """Extract diseases using BERT-based NER."""
        diseases = []
        
        if self.ner_pipeline is None:
            return diseases
        
        try:
            # Process in chunks if text is too long
            max_length = 512
            if len(text) > max_length:
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            else:
                chunks = [text]
            
            for chunk in chunks:
                entities = self.ner_pipeline(chunk)
                
                for entity in entities:
                    if entity['entity_group'] in ['DISEASE', 'MEDICAL_CONDITION']:
                        diseases.append({
                            "disease": entity['word'].title(),
                            "category": "Medical Condition",
                            "start": entity['start'],
                            "end": entity['end'],
                            "confidence": entity['score'],
                            "method": "bert"
                        })
        
        except Exception as e:
            logger.warning(f"BERT extraction failed: {e}")
        
        return diseases
    
    def extract_diseases(self, text: str, methods: List[str] = None) -> Dict[str, Any]:
        """Extract diseases using multiple methods and combine results."""
        if methods is None:
            methods = ["regex", "spacy", "bert"]
        
        all_diseases = []
        
        # Extract using different methods
        if "regex" in methods:
            regex_diseases = self.extract_with_regex(text)
            all_diseases.extend(regex_diseases)
        
        if "spacy" in methods and self.nlp is not None:
            spacy_diseases = self.extract_with_spacy(text)
            all_diseases.extend(spacy_diseases)
        
        if "bert" in methods and self.ner_pipeline is not None:
            bert_diseases = self.extract_with_bert(text)
            all_diseases.extend(bert_diseases)
        
        # Remove duplicates and merge similar diseases
        unique_diseases = self._merge_duplicates(all_diseases)
        
        # Categorize diseases
        cancer_diseases = [d for d in unique_diseases if d["category"] == "Cancer"]
        non_cancer_diseases = [d for d in unique_diseases if d["category"] != "Cancer"]
        
        return {
            "extracted_diseases": [d["disease"] for d in unique_diseases],
            "cancer_diseases": [d["disease"] for d in cancer_diseases],
            "non_cancer_diseases": [d["disease"] for d in non_cancer_diseases],
            "disease_details": unique_diseases,
            "total_diseases": len(unique_diseases),
            "cancer_count": len(cancer_diseases),
            "non_cancer_count": len(non_cancer_diseases)
        }
    
    def _merge_duplicates(self, diseases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate diseases and keep the one with highest confidence."""
        disease_map = {}
        
        for disease in diseases:
            disease_name = disease["disease"].lower()
            
            if disease_name not in disease_map:
                disease_map[disease_name] = disease
            else:
                # Keep the one with higher confidence
                if disease["confidence"] > disease_map[disease_name]["confidence"]:
                    disease_map[disease_name] = disease
        
        return list(disease_map.values())
    
    def is_cancer_related(self, text: str) -> bool:
        """Determine if text is cancer-related based on extracted diseases."""
        extraction_result = self.extract_diseases(text, methods=["regex"])
        return extraction_result["cancer_count"] > 0
    
    def process_abstracts(self, abstracts: List[str], abstract_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Process multiple abstracts and extract diseases."""
        if abstract_ids is None:
            abstract_ids = [f"abstract_{i}" for i in range(len(abstracts))]
        
        results = []
        
        for i, (abstract_id, abstract) in enumerate(zip(abstract_ids, abstracts)):
            logger.info(f"Processing abstract {i+1}/{len(abstracts)}: {abstract_id}")
            
            extraction_result = self.extract_diseases(abstract)
            
            result = {
                "abstract_id": abstract_id,
                "extracted_diseases": extraction_result["extracted_diseases"],
                "cancer_diseases": extraction_result["cancer_diseases"],
                "non_cancer_diseases": extraction_result["non_cancer_diseases"],
                "is_cancer_related": extraction_result["cancer_count"] > 0,
                "disease_count": extraction_result["total_diseases"]
            }
            
            results.append(result)
        
        return results


def main():
    """Main function for testing disease extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract diseases from abstracts')
    parser.add_argument('--input', required=True, help='Input CSV file with abstracts')
    parser.add_argument('--output', required=True, help='Output JSON file for results')
    parser.add_argument('--abstract_col', default='cleaned_abstract', help='Column name for abstracts')
    parser.add_argument('--id_col', default='pmid', help='Column name for abstract IDs')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    
    # Initialize extractor
    extractor = DiseaseExtractor()
    
    # Process abstracts
    results = extractor.process_abstracts(
        df[args.abstract_col].tolist(),
        df[args.id_col].tolist()
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Disease extraction completed! Results saved to {args.output}")
    print(f"Processed {len(results)} abstracts")


if __name__ == "__main__":
    main()
