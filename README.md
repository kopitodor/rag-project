# RAG Chatbot with Evaluation Pipeline

A retrieval-augmented generation (RAG) chatbot that answers questions 
grounded strictly in your own documents, with no hallucination outside 
the provided knowledge base.

## What it does
- Loads PDF documents and splits them into chunks
- Embeds chunks using sentence-transformers (all-MiniLM-L6-v2)
- Stores embeddings in a FAISS vector index for fast similarity search
- Retrieves the top-k most relevant chunks for any user query
- Passes retrieved context to GPT-3.5-turbo with a grounding prompt
- Serves a chat UI via Gradio

## Stack
- LangChain · sentence-transformers · FAISS · OpenAI API · Gradio

## Quickstart
```bash
conda create -n rag-project python=3.11
conda activate rag-project
pip install -r requirements.txt
```
Add your OpenAI key to a `.env` file:

OPENAI_API_KEY=sk-...

Then run:
```bash
python rag_basic.py
```
## Project status
- [x] Week 1: working end-to-end pipeline with Gradio UI
- [ ] Week 2: improved retrieval — reranker, hybrid search, memory
- [ ] Week 3: evaluation pipeline with RAGAS
- [ ] Week 4: FastAPI serving, Docker, deployment