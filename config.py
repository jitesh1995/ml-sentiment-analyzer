"""
Configuration settings for the ML Sentiment Analyzer.

Centralizes all hyperparameters, paths, and environment-specific
settings for easy tuning and deployment.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
      """ML model hyperparameters."""
      max_features: int = 10000
      ngram_range: Tuple[int, int] = (1, 2)
      max_iter: int = 1000
      regularization_c: float = 1.0
      solver: str = "lbfgs"
      multi_class: str = "multinomial"
      random_state: int = 42
      test_size: float = 0.2


@dataclass
class PreprocessingConfig:
      """Text preprocessing settings."""
      lowercase: bool = True
      remove_urls: bool = True
      remove_mentions: bool = True
      remove_hashtags: bool = True
      remove_special_chars: bool = True
      min_token_length: int = 2
      stop_words: str = "english"


@dataclass
class TrainingConfig:
      """Training pipeline settings."""
      batch_size: int = 32
      epochs: int = 10
      learning_rate: float = 0.001
      early_stopping_patience: int = 3
      validation_split: float = 0.2
      shuffle: bool = True


@dataclass
class PathConfig:
      """File and directory paths."""
      data_dir: str = os.getenv("DATA_DIR", "data")
      model_dir: str = os.getenv("MODEL_DIR", "models")
      log_dir: str = os.getenv("LOG_DIR", "logs")
      output_dir: str = os.getenv("OUTPUT_DIR", "output")

    def __post_init__(self):
              """Create directories if they don't exist."""
              for directory in [self.data_dir, self.model_dir, self.log_dir, self.output_dir]:
                            os.makedirs(directory, exist_ok=True)


@dataclass
class LoggingConfig:
      """Logging configuration."""
      level: str = os.getenv("LOG_LEVEL", "INFO")
      format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
      date_format: str = "%Y-%m-%d %H:%M:%S"
      log_to_file: bool = True
      log_filename: str = "sentiment_analyzer.log"


@dataclass
class AppConfig:
      """Main application configuration aggregator."""
      model: ModelConfig = field(default_factory=ModelConfig)
      preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
      training: TrainingConfig = field(default_factory=TrainingConfig)
      paths: PathConfig = field(default_factory=PathConfig)
      logging: LoggingConfig = field(default_factory=LoggingConfig)

    # App metadata
      app_name: str = "ML Sentiment Analyzer"
      version: str = "1.0.0"
      debug: bool = os.getenv("DEBUG", "false").lower() == "true"


# Singleton config instance
config = AppConfig()


if __name__ == "__main__":
      print(f"App: {config.app_name} v{config.version}")
      print(f"Debug: {config.debug}")
      print(f"Model max_features: {config.model.max_features}")
      print(f"Preprocessing lowercase: {config.preprocessing.lowercase}")
      print(f"Training batch_size: {config.training.batch_size}")
      print(f"Log level: {config.logging.level}")
