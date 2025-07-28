import PyPDF2
import io
import re
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline
import os
import json
from datetime import datetime

# --- Function for PDF Text Extraction and Chunking ---
def process_pdfs_and_chunk(pdf_paths, chunk_method="sentence", sentences_per_chunk=3):
    all_chunks = []
    # NLTK data is pre-downloaded in the Dockerfile.

    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        if chunk_method == "sentence":
                            sentences = nltk.sent_tokenize(page_text)
                            for i in range(0, len(sentences), sentences_per_chunk):
                                chunk_content = " ".join(sentences[i:i + sentences_per_chunk])
                                if chunk_content.strip():
                                    all_chunks.append({
                                        "content": chunk_content.strip(),
                                        "section_title": f"Page {page_num + 1}",
                                        "page": page_num + 1,
                                        "source_file": os.path.basename(pdf_path)
                                    })
        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")
            continue
    return all_chunks

# --- Main Processing Function ---
def process_documents(pdf_files, persona, job_to_be_done):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

    document_chunks = process_pdfs_and_chunk(pdf_files)
    if not document_chunks:
        return {}

    query = f"As a {persona}, I need to: {job_to_be_done}"
    query_embedding = embedding_model.encode([query])
    chunk_contents = [chunk['content'] for chunk in document_chunks]
    chunk_embeddings = embedding_model.encode(chunk_contents)

    similarities = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    ranked_chunks = []
    for i, chunk in enumerate(document_chunks):
        ranked_chunks.append({
            "score": similarities[i],
            "chunk": chunk
        })
    ranked_chunks.sort(key=lambda x: x['score'],
