"""
Script to process the provided dataset and convert it to CSV format for our pipeline.
"""

import os
import pandas as pd
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_abstract_from_file(file_path):
    """Extract abstract from a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract ID
        id_match = re.search(r'<ID:(\d+)>', content)
        abstract_id = id_match.group(1) if id_match else os.path.basename(file_path).replace('.txt', '')
        
        # Extract title
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', content)
        title = title_match.group(1).strip() if title_match else ""
        
        # Extract abstract
        abstract_match = re.search(r'Abstract:\s*(.+?)(?:\n\n|\Z)', content, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else ""
        
        return {
            'pmid': abstract_id,
            'title': title,
            'abstract': abstract,
            'file_path': file_path
        }
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def process_dataset(cancer_dir, non_cancer_dir, output_file):
    """Process the entire dataset and create a CSV file."""
    logger.info("Processing dataset...")
    
    all_data = []
    
    # Process cancer files
    logger.info("Processing cancer files...")
    cancer_files = list(Path(cancer_dir).glob("*.txt"))
    for i, file_path in enumerate(cancer_files):
        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(cancer_files)} cancer files")
        
        data = extract_abstract_from_file(file_path)
        if data and data['abstract']:  # Only include if abstract exists
            data['label'] = 1  # Cancer
            all_data.append(data)
    
    # Process non-cancer files
    logger.info("Processing non-cancer files...")
    non_cancer_files = list(Path(non_cancer_dir).glob("*.txt"))
    for i, file_path in enumerate(non_cancer_files):
        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(non_cancer_files)} non-cancer files")
        
        data = extract_abstract_from_file(file_path)
        if data and data['abstract']:  # Only include if abstract exists
            data['label'] = 0  # Non-cancer
            all_data.append(data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Remove duplicates based on PMID
    df = df.drop_duplicates(subset=['pmid'], keep='first')
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    logger.info(f"Dataset processed successfully!")
    logger.info(f"Total abstracts: {len(df)}")
    logger.info(f"Cancer abstracts: {len(df[df['label'] == 1])}")
    logger.info(f"Non-cancer abstracts: {len(df[df['label'] == 0])}")
    logger.info(f"Saved to: {output_file}")
    
    return df


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process the research paper dataset')
    parser.add_argument('--cancer_dir', default='Dataset__1_/Dataset/Cancer', 
                       help='Directory containing cancer abstracts')
    parser.add_argument('--non_cancer_dir', default='Dataset__1_/Dataset/Non-Cancer', 
                       help='Directory containing non-cancer abstracts')
    parser.add_argument('--output', default='data/raw/dataset.csv', 
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Process dataset
    df = process_dataset(args.cancer_dir, args.non_cancer_dir, args.output)
    
    # Show sample
    print("\nSample data:")
    print(df.head())
    
    print(f"\nDataset statistics:")
    print(f"Total abstracts: {len(df)}")
    print(f"Cancer: {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)")
    print(f"Non-cancer: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
