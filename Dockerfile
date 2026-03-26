# ---- ML Sentiment Analyzer ----
# Multi-stage Docker build for lightweight production image

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim AS production

LABEL maintainer="jitesh1995"
LABEL description="ML Sentiment Analyzer - NLP-powered text classification"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY sentiment_analyzer.py .
COPY config.py .

# Create non-root user for security
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
    USER appuser

    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
        CMD python -c "import sentiment_analyzer; print('OK')" || exit 1

        # Environment variables
        ENV PYTHONUNBUFFERED=1
        ENV PYTHONDONTWRITEBYTECODE=1
        ENV LOG_LEVEL=INFO

        EXPOSE 8000

        # Default command: run the demo
        CMD ["python", "sentiment_analyzer.py"]
