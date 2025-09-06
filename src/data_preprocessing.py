"""
Data preprocessing pipeline for research paper abstracts.
Handles PubMed data cleaning, normalization, and preparation for model training.
"""

import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbstractPreprocessor:
    """Handles preprocessing of research paper abstracts."""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.disease_patterns = self._load_disease_patterns()
    
    def _load_disease_patterns(self) -> List[str]:
        """Load common disease patterns for extraction."""
        return [
            r'\b(?:cancer|carcinoma|tumor|neoplasm|malignancy|oncology)\b',
            r'\b(?:lung|breast|prostate|colorectal|pancreatic|ovarian|brain)\s+cancer\b',
            r'\b(?:leukemia|lymphoma|melanoma|sarcoma)\b',
            r'\b(?:diabetes|hypertension|cardiovascular|stroke|heart disease)\b',
            r'\b(?:alzheimer|parkinson|dementia|depression|anxiety)\b',
            r'\b(?:asthma|copd|pneumonia|tuberculosis|influenza)\b'
        ]
    
    def clean_abstract(self, text: str) -> str:
        """Clean and normalize abstract text."""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Normalize citations (remove reference numbers)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d{4}\)', '', text)
        
        return text.strip()
    
    def extract_diseases(self, text: str) -> List[str]:
        """Extract disease mentions from abstract text."""
        diseases = []
        text_lower = text.lower()
        
        for pattern in self.disease_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            diseases.extend(matches)
        
        # Remove duplicates and clean up
        diseases = list(set([d.strip() for d in diseases if d.strip()]))
        return diseases
    
    def is_cancer_related(self, text: str) -> bool:
        """Determine if abstract is cancer-related based on keywords."""
        cancer_keywords = [
            'cancer', 'carcinoma', 'tumor', 'neoplasm', 'malignancy', 
            'oncology', 'metastasis', 'chemotherapy', 'radiotherapy'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in cancer_keywords)
    
    def preprocess_dataset(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """Preprocess the entire dataset."""
        logger.info(f"Loading data from {data_path}")
        
        # Load data (assuming CSV format with columns: pmid, abstract, label)
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        logger.info(f"Loaded {len(df)} abstracts")
        
        # Clean abstracts
        df['cleaned_abstract'] = df['abstract'].apply(self.clean_abstract)
        
        # Remove abstracts that are too short or empty
        df = df[df['cleaned_abstract'].str.len() > 50]
        
        # Extract diseases
        df['extracted_diseases'] = df['cleaned_abstract'].apply(self.extract_diseases)
        
        # Create binary labels if not present
        if 'label' not in df.columns:
            df['label'] = df['cleaned_abstract'].apply(self.is_cancer_related)
            df['label'] = df['label'].astype(int)
        
        # Encode labels
        df['encoded_label'] = self.label_encoder.fit_transform(df['label'])
        
        # Split data
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['encoded_label']
        )
        
        # Save processed data
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_dir / 'train.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        # Save metadata
        metadata = {
            'total_abstracts': int(len(df)),
            'train_size': int(len(train_df)),
            'test_size': int(len(test_df)),
            'cancer_abstracts': int(df['label'].sum()),
            'non_cancer_abstracts': int((~df['label'].astype(bool)).sum()),
            'label_mapping': {str(k): int(v) for k, v in zip(
                self.label_encoder.classes_, 
                self.label_encoder.transform(self.label_encoder.classes_)
            )}
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Preprocessing complete. Saved to {output_path}")
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        return metadata


def main():
    """Main preprocessing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess research paper abstracts')
    parser.add_argument('--input', required=True, help='Input data path')
    parser.add_argument('--output', required=True, help='Output directory path')
    
    args = parser.parse_args()
    
    preprocessor = AbstractPreprocessor()
    metadata = preprocessor.preprocess_dataset(args.input, args.output)
    
    print("Preprocessing completed successfully!")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    main()
