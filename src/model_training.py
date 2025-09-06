"""
Model training pipeline for cancer classification using LoRA fine-tuning.
Supports Gemma, Phi, and other small language models.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbstractDataset(Dataset):
    """Dataset class for abstract classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class CancerClassifier:
    """Cancer classification model with LoRA fine-tuning support."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", use_quantization: bool = True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model: {model_name}")
    
    def _setup_quantization(self):
        """Setup quantization configuration for memory efficiency."""
        if self.use_quantization and torch.cuda.is_available():
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return None
    
    def load_model(self, num_labels: int = 2):
        """Load the base model and tokenizer."""
        logger.info("Loading tokenizer and model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if enabled
        quantization_config = self._setup_quantization()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Setup LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model loaded successfully!")
    
    def prepare_datasets(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                        text_col: str = 'cleaned_abstract', label_col: str = 'encoded_label'):
        """Prepare training and test datasets."""
        logger.info("Preparing datasets...")
        
        train_dataset = AbstractDataset(
            train_data[text_col].tolist(),
            train_data[label_col].tolist(),
            self.tokenizer
        )
        
        test_dataset = AbstractDataset(
            test_data[text_col].tolist(),
            test_data[label_col].tolist(),
            self.tokenizer
        )
        
        return train_dataset, test_dataset
    
    def train(self, train_dataset: AbstractDataset, test_dataset: AbstractDataset,
              output_dir: str = "./models", num_epochs: int = 3, batch_size: int = 8):
        """Train the model with LoRA fine-tuning."""
        logger.info("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Custom metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'f1': f1
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed! Model saved to {output_dir}")
        
        return trainer
    
    def evaluate(self, test_dataset: AbstractDataset) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        logger.info("Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in DataLoader(test_dataset, batch_size=8):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions, 
            target_names=['Non-Cancer', 'Cancer'],
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
        
        logger.info(f"Evaluation completed - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train cancer classification model')
    parser.add_argument('--data_dir', required=True, help='Path to processed data directory')
    parser.add_argument('--model_name', default='microsoft/DialoGPT-medium', 
                       help='Hugging Face model name')
    parser.add_argument('--output_dir', default='./models', help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--use_quantization', action='store_true', 
                       help='Use 4-bit quantization for memory efficiency')
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_csv(f"{args.data_dir}/train.csv")
    test_data = pd.read_csv(f"{args.data_dir}/test.csv")
    
    # Load metadata
    with open(f"{args.data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Initialize classifier
    classifier = CancerClassifier(
        model_name=args.model_name,
        use_quantization=args.use_quantization
    )
    
    # Load model
    classifier.load_model(num_labels=2)
    
    # Prepare datasets
    train_dataset, test_dataset = classifier.prepare_datasets(train_data, test_data)
    
    # Train model
    trainer = classifier.train(
        train_dataset, test_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate model
    results = classifier.evaluate(test_dataset)
    
    # Save results
    results_path = Path(args.output_dir) / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training completed successfully!")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
