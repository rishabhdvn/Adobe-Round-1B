import PyPDF2
import os
import json
import nltk
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# --- Chunking PDF Text ---
def process_pdfs_and_chunk(pdf_paths, chunk_method="sentence", sentences_per_chunk=3):
    all_chunks = []
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
    return all_chunks

# --- Main Logic ---
def main():
    input_dir = "/app/input"
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)

    # Load files from /app/input
    pdf_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDFs found in input folder.")
        return

    persona = "Investment Analyst"
    job_to_be_done = "Analyze revenue trends, R&D investments, and market positioning strategies."
    query = f"As a {persona}, I need to: {job_to_be_done}"

    print("Loading models...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

    print("Processing documents...")
    document_chunks = process_pdfs_and_chunk(pdf_files)

    if not document_chunks:
        print("No chunks generated.")
        return

    # Embedding and similarity
    print("Ranking content by semantic relevance...")
    query_embedding = embedding_model.encode([query])
    chunk_contents = [chunk['content'] for chunk in document_chunks]
    chunk_embeddings = embedding_model.encode(chunk_contents)

    similarities = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    ranked_chunks = sorted([
        {"score": similarities[i], "chunk": chunk}
        for i, chunk in enumerate(document_chunks)
    ], key=lambda x: x["score"], reverse=True)

    # Generate summary from top chunks
    top_n = 3
    top_chunks = ranked_chunks[:top_n]
    combined_text = " ".join([item["chunk"]["content"] for item in top_chunks])

    try:
        summary = summarizer(combined_text, max_length=180, min_length=40, do_sample=False)[0]["summary_text"]
    except Exception as e:
        print(f"Summarization failed: {e}")
        summary = "Summary generation failed."

    # Prepare output JSON
    output_data = {
        "metadata": {
            "input_documents": [os.path.basename(f) for f in pdf_files],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "timestamp": datetime.now().isoformat()
        },
        "top_chunks": [
            {
                "document": item['chunk']['source_file'],
                "page": item['chunk']['page'],
                "section_title": item['chunk']['section_title'],
                "importance_rank": i + 1
            }
            for i, item in enumerate(top_chunks)
        ],
        "refined_text": summary
    }

    # Save output
    output_path = os.path.join(output_dir, "final_output.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✔ Output written to: {output_path}")

if __name__ == "__main__":
    nltk.download("punkt")
    main()
