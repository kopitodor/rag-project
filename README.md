# RAG Chatbot with Evaluation Pipeline

A retrieval-augmented generation (RAG) chatbot that answers questions 
grounded strictly in your own documents, with no hallucination outside 
the provided knowledge base. Currently loaded with NVIDIA's FY2025 
annual report (10-K) and the FIBA Official Basketball Rules 2024.

## What it does
- Loads multiple PDF documents automatically from the project folder
- Cleans and chunks documents with configurable size and overlap
- Embeds chunks using `BAAI/bge-large-en-v1.5` for high-quality retrieval
- Stores embeddings in a FAISS vector index with disk caching for fast restarts
- Retrieves candidates using hybrid search — dense (cosine) + sparse (BM25)
- Reranks combined candidates with a CrossEncoder for precision
- Passes top-k reranked chunks to GPT-3.5-turbo with a grounding prompt
- Maintains conversation memory across turns
- Serves a chat UI via Gradio

## Stack
- LangChain · sentence-transformers · FAISS · BM25 · CrossEncoder · OpenAI API · Gradio

## Architecture
```
PDFs → clean → chunk → embed (bge-large) → FAISS index (cached to disk)
                                  ↓
Query → dense retrieval (FAISS top-10)  ─┐
      → sparse retrieval (BM25 top-10)  ─┴→ reranker → top-3 → LLM → answer
```

## Quickstart
```bash
conda create -n rag-project python=3.11
conda activate rag-project
pip install -r requirements.txt
```

Add your OpenAI key to a `.env` file:
```
OPENAI_API_KEY=sk-...
```

Add any PDF files to the project folder, then run:
```bash
python rag_basic.py
```

The vector index is built on first run and cached — subsequent runs load instantly.

## Key design decisions
- **Hybrid search over dense-only:** BM25 catches exact keyword matches 
  (names, numbers, technical terms) that embedding models can miss
- **Reranker after retrieval:** CrossEncoder re-scores candidates more 
  accurately than cosine similarity, fixing query phrasing sensitivity
- **Prompt grounding:** LLM is explicitly constrained to context only — 
  tested to refuse out-of-context questions even when it knows the answer
- **High chunk overlap (400 tokens):** Prevents relevant sentences from 
  falling on chunk boundaries and becoming unretrievable

## Project status
- [x] Week 1: end-to-end pipeline — PDF loading, FAISS, GPT-3.5, Gradio UI
- [x] Week 2: hybrid search, reranker, multi-document, memory, disk caching
- [ ] Week 3: evaluation pipeline with RAGAS — faithfulness, relevancy scores
- [ ] Week 4: FastAPI serving, Docker, Hugging Face Spaces deployment