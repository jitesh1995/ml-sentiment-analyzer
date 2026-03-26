# ML Sentiment Analyzer

A Python-based sentiment analysis tool using NLP and Machine Learning. Supports text classification with scikit-learn (TF-IDF + Logistic Regression) and transformers for positive, negative, and neutral sentiment detection.

## Features

- **Text Preprocessing**: Cleans URLs, mentions, hashtags, and special characters
- - **TF-IDF Vectorization**: Converts text to numerical features using n-gram TF-IDF
  - - **Logistic Regression Classifier**: Multi-class sentiment classification with confidence scores
    - - **Batch Prediction**: Analyze multiple texts at once
      - - **Built-in Sample Data**: Demo dataset included for quick testing
       
        - ## Project Structure
       
        - ```
          ml-sentiment-analyzer/
          ├── sentiment_analyzer.py   # Main ML pipeline and analyzer classes
          ├── requirements.txt        # Python dependencies
          ├── .gitignore             # Python gitignore template
          ├── LICENSE                # MIT License
          └── README.md              # This file
          ```

          ## Quick Start

          ### Installation

          ```bash
          git clone https://github.com/jitesh1995/ml-sentiment-analyzer.git
          cd ml-sentiment-analyzer
          pip install -r requirements.txt
          ```

          ### Run Demo

          ```bash
          python sentiment_analyzer.py
          ```

          ### Usage in Code

          ```python
          from sentiment_analyzer import SentimentAnalyzer, load_sample_data

          # Load data and train
          texts, labels = load_sample_data()
          analyzer = SentimentAnalyzer(max_features=5000)
          metrics = analyzer.train(texts, labels)

          # Predict sentiment
          result = analyzer.predict("This product is amazing!")
          print(result["sentiment"])    # "positive"
          print(result["confidence"])   # 0.95
          ```

          ## Sentiment Labels

          | Label | Value | Description |
          |-------|-------|-------------|
          | Negative | 0 | Negative sentiment (complaints, dissatisfaction) |
          | Neutral | 1 | Neutral sentiment (factual, balanced) |
          | Positive | 2 | Positive sentiment (praise, satisfaction) |

          ## Tech Stack

          - **Python 3.9+**
          - - **scikit-learn** - ML pipeline (TF-IDF + Logistic Regression)
            - - **NumPy** - Numerical operations
              - - **Transformers** - Optional: HuggingFace transformer models
                - - **NLTK** - Natural language processing utilities
                 
                  - ## License
                 
                  - This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
                 
                  - ## Author
                 
                  - **Jitesh Khemchandani** - [GitHub](https://github.com/jitesh1995)
