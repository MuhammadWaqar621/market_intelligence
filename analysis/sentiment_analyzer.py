import logging
import re
import nltk
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import Counter

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from config.settings import SENTIMENT_MODEL, MODEL_CACHE_DIR
from database.connection import get_db_session
from database.models import Product, SentimentAnalysis

# Configure logging
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Class for analyzing sentiment in product reviews."""
    
    def __init__(self, model_name: str = SENTIMENT_MODEL, use_transformers: bool = True):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the pre-trained transformer model or 'logistic' for traditional ML.
            use_transformers: Whether to use transformer models or traditional ML.
        """
        self.model_name = model_name
        self.use_transformers = use_transformers
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.model = None
        self.tokenizer = None
        
        # Load model
        if use_transformers:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
                logger.info(f"Loaded transformer model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading transformer model: {str(e)}")
                self.use_transformers = False
                
        # Initialize traditional ML model as backup
        if not use_transformers:
            self.model = LogisticRegression()
            self.vectorizer = CountVectorizer(max_features=5000)
            logger.info("Using Logistic Regression model for sentiment analysis")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Join tokens back into text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def train_traditional_model(self, reviews: List[str], labels: List[int]) -> None:
        """
        Train the traditional ML model on labeled data.
        
        Args:
            reviews: List of preprocessed review texts.
            labels: List of sentiment labels (0: negative, 1: neutral, 2: positive).
        """
        if len(reviews) != len(labels):
            logger.error("Number of reviews and labels must match")
            return
            
        try:
            # Vectorize text
            X = self.vectorizer.fit_transform(reviews)
            
            # Train model
            self.model.fit(X, labels)
            
            logger.info(f"Trained Logistic Regression model on {len(reviews)} reviews")
            
        except Exception as e:
            logger.error(f"Error training traditional model: {str(e)}")
    
    def predict_sentiment_transformer(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment using transformer model.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Tuple of (sentiment_label, sentiment_score).
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Map class to label
            if predicted_class == 0:
                label = "Negative"
                score = -confidence
            elif predicted_class == 1:
                label = "Neutral"
                score = 0.0
            else:
                label = "Positive"
                score = confidence
            
            return label, score
            
        except Exception as e:
            logger.error(f"Error predicting sentiment with transformer: {str(e)}")
            return "Neutral", 0.0
    
    def predict_sentiment_traditional(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment using traditional ML model.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Tuple of (sentiment_label, sentiment_score).
        """
        try:
            # Vectorize text
            X = self.vectorizer.transform([text])
            
            # Predict
            predicted_class = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = probabilities[predicted_class]
            
            # Map class to label
            if predicted_class == 0:
                label = "Negative"
                score = -confidence
            elif predicted_class == 1:
                label = "Neutral"
                score = 0.0
            else:
                label = "Positive"
                score = confidence
            
            return label, score
            
        except Exception as e:
            logger.error(f"Error predicting sentiment with traditional model: {str(e)}")
            return "Neutral", 0.0
    
    def predict_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment of text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Tuple of (sentiment_label, sentiment_score).
        """
        if not text or not isinstance(text, str):
            return "Neutral", 0.0
            
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        if not preprocessed_text:
            return "Neutral", 0.0
            
        # Predict sentiment
        if self.use_transformers and self.model is not None and self.tokenizer is not None:
            return self.predict_sentiment_transformer(preprocessed_text)
        else:
            return self.predict_sentiment_traditional(preprocessed_text)
    
    def extract_topics(self, text: str, n_topics: int = 3) -> List[str]:
        """
        Extract common topics from text.
        
        Args:
            text: Text to analyze.
            n_topics: Number of topics to extract.
            
        Returns:
            List of extracted topics.
        """
        if not text or not isinstance(text, str):
            return []
            
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        if not preprocessed_text:
            return []
            
        # Tokenize
        tokens = word_tokenize(preprocessed_text)
        
        # Count word frequencies
        word_counts = Counter(tokens)
        
        # Remove single-character words
        word_counts = {word: count for word, count in word_counts.items() if len(word) > 1}
        
        # Get most common words
        most_common = word_counts.most_common(n_topics)
        
        # Extract topics
        topics = [word for word, count in most_common]
        
        return topics
    
    def analyze_product_reviews(self, product_id: int) -> Dict[str, Any]:
        """
        Analyze sentiment of product reviews.
        
        Args:
            product_id: Database ID of the product.
            
        Returns:
            Dictionary with sentiment analysis results.
        """
        session = get_db_session()
        if not session:
            return {"status": "error", "message": "Database connection failed"}
        
        try:
            # Get product and reviews
            product = session.query(Product).filter(Product.id == product_id).first()
            
            if not product:
                return {"status": "error", "message": f"Product with ID {product_id} not found"}
                
            reviews = product.reviews
            
            if not reviews:
                return {
                    "status": "success",
                    "product_name": product.product_name,
                    "sentiment": "No reviews available",
                    "sentiment_score": 0.0,
                    "top_issues": [],
                    "top_praises": []
                }
            
            # Split multiple reviews if they're in a single string
            review_list = reviews.split(" | ") if " | " in reviews else [reviews]
            
            # Analyze sentiment for each review
            sentiments = []
            scores = []
            all_text = ""
            
            for review in review_list:
                label, score = self.predict_sentiment(review)
                sentiments.append(label)
                scores.append(score)
                all_text += " " + review
            
            # Calculate overall sentiment
            positive_count = sentiments.count("Positive")
            neutral_count = sentiments.count("Neutral")
            negative_count = sentiments.count("Negative")
            
            if positive_count > negative_count and positive_count >= neutral_count:
                overall_sentiment = "Positive"
            elif negative_count > positive_count and negative_count >= neutral_count:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"
                
            overall_score = sum(scores) / len(scores) if scores else 0.0
            
            # Extract topics
            positive_reviews = " ".join([review for review, label in zip(review_list, sentiments) if label == "Positive"])
            negative_reviews = " ".join([review for review, label in zip(review_list, sentiments) if label == "Negative"])
            
            top_praises = self.extract_topics(positive_reviews, 5)
            top_issues = self.extract_topics(negative_reviews, 5)
            
            # Save analysis to database
            existing_analysis = session.query(SentimentAnalysis).filter(
                SentimentAnalysis.product_id == product_id
            ).first()
            
            if existing_analysis:
                existing_analysis.sentiment_score = overall_score
                existing_analysis.sentiment_label = overall_sentiment
                existing_analysis.top_issues = ",".join(top_issues)
                existing_analysis.top_praises = ",".join(top_praises)
            else:
                new_analysis = SentimentAnalysis(
                    product_id=product_id,
                    sentiment_score=overall_score,
                    sentiment_label=overall_sentiment,
                    top_issues=",".join(top_issues),
                    top_praises=",".join(top_praises)
                )
                session.add(new_analysis)
                
            session.commit()
            
            return {
                "status": "success",
                "product_name": product.product_name,
                "sentiment": overall_sentiment,
                "sentiment_score": round(overall_score, 2),
                "sentiment_distribution": {
                    "positive": positive_count,
                    "neutral": neutral_count,
                    "negative": negative_count
                },
                "top_issues": top_issues,
                "top_praises": top_praises
            }
            
        except Exception as e:
            logger.error(f"Error analyzing product reviews: {str(e)}")
            session.rollback()
            return {"status": "error", "message": str(e)}
            
        finally:
            session.close()
    
    def batch_analyze_products(self, product_ids: List[int]) -> Dict[str, Any]:
        """
        Batch analyze sentiment of multiple products.
        
        Args:
            product_ids: List of product IDs to analyze.
            
        Returns:
            Dictionary with batch analysis results.
        """
        results = []
        
        for product_id in product_ids:
            result = self.analyze_product_reviews(product_id)
            if result.get("status") == "success":
                results.append({
                    "product_id": product_id,
                    "product_name": result.get("product_name"),
                    "sentiment": result.get("sentiment"),
                    "sentiment_score": result.get("sentiment_score"),
                    "top_issues": result.get("top_issues", [])
                })
        
        return {
            "status": "success",
            "results": results,
            "count": len(results)
        }