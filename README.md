---
title: RAG Project
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: rag_basic.py
pinned: false
python_version: 3.11
---

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
- LangChain · sentence-transformers · FAISS · BM25 · CrossEncoder · OpenAI API · Gradio · RAGAS

## Architecture
```
PDFs -> clean -> chunk -> embed (bge-large) -> FAISS index (cached to disk)
                                    |
Query -> dense retrieval (FAISS)  --+
      -> sparse retrieval (BM25)  --+--> reranker -> top-3 -> LLM -> answer
```

## Quickstart
```bash
conda create -n rag-project python=3.11
conda activate rag-project
pip install -r requirements.txt
```

Add your OpenAI key to a `.env` file:
OPENAI_API_KEY=sk-...

Add any PDF files to the project folder, then run:
```bash
python rag_basic.py
```

The vector index is built on first run and cached — subsequent runs load instantly.

## Evaluation

The pipeline is evaluated using RAGAS with a hand-crafted golden set of 
20 question/answer pairs (12 NVIDIA, 8 FIBA), verified directly against 
the source documents.

### Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Are all answer claims supported by retrieved chunks? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are the most relevant chunks ranked highest? |
| Context Recall | Do retrieved chunks contain everything needed to answer? |

### Results

| Metric | Run 1 (baseline) | Run 2 (tighter prompt) |
|---|---|---|
| Faithfulness | 0.695 | 0.732 |
| Answer Relevancy | 0.927 | 0.942 |
| Context Precision | 0.925 | 0.958 |
| Context Recall | 0.950 | 0.950 |

Run the evaluation harness:
```bash
python eval/run_eval.py
```

Results are saved to `eval/results/` as timestamped JSON files.

### Evaluation design decisions
- **Hand-crafted golden set over auto-generated:** Ground truths were 
  verified directly against the source PDFs to ensure the evaluator and 
  pipeline share the same knowledge source
- **Source attribution removed from answers:** Phrases like "as stated in 
  the 10-K" introduce unverifiable claims that hurt faithfulness scores. 
  In a RAG system, provenance is implicit in the architecture — it does 
  not need to be stated in the answer
- **Metric limitations acknowledged:** ContextRecall can produce false 
  negatives on questions requiring implicit computation (e.g. calculating 
  a percentage from two raw figures). Automated metrics are signals, 
  not verdicts — human spot-checks remain necessary

## Key design decisions
- **Hybrid search over dense-only:** BM25 catches exact keyword matches 
  (names, numbers, technical terms) that embedding models can miss due 
  to lexical gaps
- **Reranker after retrieval:** CrossEncoder re-scores candidates more 
  accurately than cosine similarity, fixing query phrasing sensitivity
- **Prompt grounding:** LLM is explicitly constrained to context only — 
  tested to refuse out-of-context questions even when it knows the answer
- **High chunk overlap (400 tokens):** Prevents relevant sentences from 
  falling on chunk boundaries and becoming unretrievable

## Project status
- [x] Week 1: end-to-end pipeline — PDF loading, FAISS, GPT-3.5, Gradio UI
- [x] Week 2: hybrid search, reranker, multi-document, memory, disk caching
- [x] Week 3: RAGAS eval pipeline — 20-question golden set, 4 metrics, two runs
- [ ] Week 4: FastAPI serving, Docker, Hugging Face Spaces deployment