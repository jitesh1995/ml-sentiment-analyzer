"""
Sentiment Analyzer - ML-powered text sentiment classification tool.

This module provides sentiment analysis capabilities using both
traditional ML (scikit-learn) and transformer-based approaches.
Supports positive, negative, and neutral sentiment detection.

Author: jitesh1995
License: MIT
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentiment labels
LABELS = {0: "negative", 1: "neutral", 2: "positive"}


class TextPreprocessor:
      """Handles text cleaning and preprocessing for sentiment analysis."""

    @staticmethod
    def clean_text(text: str) -> str:
              """Remove noise from text data."""
              text = text.lower()
              text = re.sub(r"http\S+|www\S+|https\S+", "", text)
              text = re.sub(r"@\w+", "", text)
              text = re.sub(r"#\w+", "", text)
              text = re.sub(r"[^a-zA-Z\s]", "", text)
              text = re.sub(r"\s+", " ", text).strip()
              return text

    @staticmethod
    def preprocess_batch(texts: List[str]) -> List[str]:
              """Clean a batch of text samples."""
              return [TextPreprocessor.clean_text(t) for t in texts]


class SentimentAnalyzer:
      """
          ML-based Sentiment Analyzer using TF-IDF + Logistic Regression.

              Supports training on custom datasets and predicting sentiment
                  for new text inputs with confidence scores.
                      """

    def __init__(self, max_features: int = 10000, ngram_range: Tuple = (1, 2)):
              self.preprocessor = TextPreprocessor()
              self.pipeline = Pipeline([
                  ("tfidf", TfidfVectorizer(
                      max_features=max_features,
                      ngram_range=ngram_range,
                      stop_words="english",
                      sublinear_tf=True,
                  )),
                  ("classifier", LogisticRegression(
                      max_iter=1000,
                      C=1.0,
                      solver="lbfgs",
                      multi_class="multinomial",
                      random_state=42,
                  )),
              ])
              self.is_trained = False
              logger.info("SentimentAnalyzer initialized.")

    def train(
              self,
              texts: List[str],
              labels: List[int],
              test_size: float = 0.2,
    ) -> Dict[str, float]:
              """
                      Train the sentiment model on labeled data.

                              Args:
                                          texts: List of text samples.
                                                      labels: List of integer labels (0=negative, 1=neutral, 2=positive).
                                                                  test_size: Fraction of data to use for validation.

                                                                          Returns:
                                                                                      Dictionary with training and validation metrics.
                                                                                              """
              logger.info(f"Training on {len(texts)} samples...")
              cleaned = self.preprocessor.preprocess_batch(texts)

        X_train, X_val, y_train, y_val = train_test_split(
                      cleaned, labels, test_size=test_size, random_state=42, stratify=labels
        )

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        train_acc = accuracy_score(y_train, self.pipeline.predict(X_train))
        val_acc = accuracy_score(y_val, self.pipeline.predict(X_val))

        val_preds = self.pipeline.predict(X_val)
        report = classification_report(y_val, val_preds, target_names=list(LABELS.values()))

        logger.info(f"Train Accuracy: {train_acc:.4f}")
        logger.info(f"Validation Accuracy: {val_acc:.4f}")
        logger.info(f"\n{report}")

        return {"train_accuracy": train_acc, "val_accuracy": val_acc}

    def predict(self, text: str) -> Dict[str, any]:
              """
                      Predict sentiment for a single text input.

                              Args:
                                          text: Input text string.

                                                  Returns:
                                                              Dictionary with predicted label, confidence, and probabilities.
                                                                      """
              if not self.is_trained:
                            raise RuntimeError("Model not trained. Call train() first.")

              cleaned = self.preprocessor.clean_text(text)
              prediction = self.pipeline.predict([cleaned])[0]
              probabilities = self.pipeline.predict_proba([cleaned])[0]

        return {
                      "text": text,
                      "sentiment": LABELS[prediction],
                      "confidence": float(np.max(probabilities)),
                      "probabilities": {
                                        LABELS[i]: float(p) for i, p in enumerate(probabilities)
                      },
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
              """Predict sentiment for multiple texts."""
              return [self.predict(t) for t in texts]


def load_sample_data() -> Tuple[List[str], List[int]]:
      """Load a small built-in sample dataset for demonstration."""
      samples = [
          ("I absolutely love this product, it works great!", 2),
          ("This is the best purchase I have ever made.", 2),
          ("Wonderful experience, highly recommend!", 2),
          ("The quality exceeded my expectations.", 2),
          ("Amazing service and fast delivery.", 2),
          ("The product is okay, nothing special.", 1),
          ("It works as expected, average quality.", 1),
          ("Decent product for the price.", 1),
          ("Neither good nor bad, just average.", 1),
          ("It does what it says, nothing more.", 1),
          ("Terrible product, waste of money.", 0),
          ("Very disappointed with the quality.", 0),
          ("Would not recommend to anyone.", 0),
          ("Broken on arrival, worst purchase.", 0),
          ("Customer service was horrible.", 0),
      ]
      texts, labels = zip(*samples)
      return list(texts), list(labels)


def main():
      """Main entry point for demonstration."""
      print("=" * 60)
      print("  ML Sentiment Analyzer - Demo")
      print("=" * 60)

    # Load sample data
      texts, labels = load_sample_data()
    print(f"\nLoaded {len(texts)} sample texts for training.\n")

    # Initialize and train
    analyzer = SentimentAnalyzer(max_features=5000)
    metrics = analyzer.train(texts, labels, test_size=0.2)

    print(f"\nTraining Accuracy:   {metrics['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")

    # Test predictions
    test_texts = [
              "This product is fantastic and I love it!",
              "Meh, it was just okay I guess.",
              "Absolutely terrible, never buying again.",
    ]

    print("\n" + "-" * 60)
    print("  Sample Predictions")
    print("-" * 60)

    for text in test_texts:
              result = analyzer.predict(text)
              print(f"\n  Text: {result['text']}")
              print(f"  Sentiment: {result['sentiment'].upper()}")
              print(f"  Confidence: {result['confidence']:.2%}")
              for label, prob in result["probabilities"].items():
                            print(f"    {label}: {prob:.2%}")


if __name__ == "__main__":
      main()
