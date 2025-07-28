# Adobe Hackathon - Round 1B Submission

This project is a solution for Round 1B of the Adobe Hackathon, "Persona-Driven Document Intelligence". It's a Python-based pipeline that analyzes a collection of PDF documents to extract the most relevant sections based on a given user persona and their specific job-to-be-done.

### Approach

The solution uses a multi-stage pipeline to process and rank information:
1.  **PDF Processing & Chunking**: The system ingests multiple PDF documents and uses `PyPDF2` to extract raw text, which is then chunked using `NLTK`.
2.  **Semantic Embedding**: A query is constructed from the persona and job. The `all-MiniLM-L6-v2` model converts the query and chunks into vector embeddings.
3.  **Relevance Ranking**: Cosine similarity is used to rank chunks by relevance to the user's task.
4.  **Summarization & Analysis**: The top-ranked chunks are summarized using the `distilbart-cnn-6-6` model to generate refined text for the final output.

### Models and Libraries Used

* **PDF Processing**: `PyPDF2`
* **Text Processing**: `nltk`
* **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Summarization**: `transformers` (`sshleifer/distilbart-cnn-6-6`)
* **Backend**: `torch` (CPU version)

### How to Build and Run

**1. Build the Docker Image**
```bash
docker build -t round1b-solution .

docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none round1b-solution