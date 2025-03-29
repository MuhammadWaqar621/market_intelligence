import os
import logging
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

from config.settings import MODEL_CACHE_DIR

# Configure logging
logger = logging.getLogger(__name__)


class SentimentModel:
    """Class for training and fine-tuning sentiment analysis models."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 3):
        """
        Initialize the sentiment model.
        
        Args:
            model_name: Name of the pre-trained model to use.
            num_labels: Number of sentiment labels (3 for positive, neutral, negative).
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                cache_dir=MODEL_CACHE_DIR
            )
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        return self.tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
    
    def fine_tune(self, texts: List[str], labels: List[int], output_dir: str = None) -> Dict[str, Any]:
        """
        Fine-tune the sentiment model on custom data.
        
        Args:
            texts: List of text examples.
            labels: List of corresponding labels.
            output_dir: Directory to save the fine-tuned model.
            
        Returns:
            Dictionary with training results.
        """
        if self.model is None or self.tokenizer is None:
            return {"status": "error", "message": "Model or tokenizer not initialized"}
            
        if len(texts) != len(labels):
            return {"status": "error", "message": "Number of texts and labels must match"}
            
        try:
            # Prepare dataset
            df = pd.DataFrame({"text": texts, "label": labels})
            train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
            
            train_dataset = Dataset.from_pandas(train_df)
            eval_dataset = Dataset.from_pandas(eval_df)
            
            # Tokenize datasets
            tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
            tokenized_eval = eval_dataset.map(self.tokenize_function, batched=True)
            
            # Set up training arguments
            if output_dir is None:
                output_dir = os.path.join(MODEL_CACHE_DIR, "fine_tuned_sentiment")
                
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval
            )
            
            # Train model
            trainer.train()
            
            # Evaluate model
            eval_results = trainer.evaluate()
            
            # Save model
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            return {
                "status": "success",
                "eval_results": eval_results,
                "model_path": output_dir
            }
            
        except Exception as e:
            logger.error(f"Error fine-tuning model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze.
            
        Returns:
            List of dictionaries with prediction results.
        """
        if self.model is None or self.tokenizer is None:
            return [{"error": "Model or tokenizer not initialized"}]
            
        results = []
        
        try:
            # Tokenize texts
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Process results
            for i, text in enumerate(texts):
                probs = probabilities[i].tolist()
                predicted_class = torch.argmax(probabilities[i]).item()
                
                # Map class to label
                if self.num_labels == 3:
                    label = ["Negative", "Neutral", "Positive"][predicted_class]
                else:
                    label = str(predicted_class)
                
                results.append({
                    "text": text,
                    "label": label,
                    "confidence": probs[predicted_class],
                    "probabilities": {
                        "negative": probs[0] if self.num_labels == 3 else None,
                        "neutral": probs[1] if self.num_labels == 3 else None,
                        "positive": probs[2] if self.num_labels == 3 else None
                    }
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return [{"error": str(e)}]
    
    def save_model(self, path: str) -> Dict[str, Any]:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model.
            
        Returns:
            Dictionary with saving results.
        """
        if self.model is None or self.tokenizer is None:
            return {"status": "error", "message": "Model or tokenizer not initialized"}
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            return {
                "status": "success",
                "model_path": path
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def load_model(self, path: str) -> Dict[str, Any]:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Dictionary with loading results.
        """
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            
            # Update num_labels
            self.num_labels = self.model.config.num_labels
            
            return {
                "status": "success",
                "model_path": path,
                "num_labels": self.num_labels
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return {"status": "error", "message": str(e)}