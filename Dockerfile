# Use AMD64 Python base image
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# System & Python dependencies
RUN apt-get update && \
    apt-get install -y git && \
    pip install --no-cache-dir \
        PyPDF2 \
        numpy \
        nltk \
        scikit-learn \
        sentence-transformers \
        transformers \
        reportlab

# Download NLTK tokenizer
RUN python -c "import nltk; nltk.download('punkt')"

# Preload models for offline use
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from transformers import pipeline; pipeline('summarization', model='sshleifer/distilbart-cnn-6-6')" && \
    python -c "from transformers import pipeline; pipeline('text-generation', model='gpt2-medium')"

# Optional: clean temp huggingface download folders
RUN rm -rf /root/.cache/huggingface/hub/*_tmp*

# Offline mode for HuggingFace Transformers
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Run your main script
CMD ["python", "main.py"]
