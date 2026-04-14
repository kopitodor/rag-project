from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import gradio as gr
import time
import os
from pathlib import Path


load_dotenv()

# --- load multiple PDFs
print("Loading PDFs...")

pdf_files = list(Path(".").glob("*.pdf"))
print(f"Found {len(pdf_files)} PDFs: {[f.name for f in pdf_files]}")

pages = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    doc_pages = loader.load()
    # add filename to metadata so we know which doc each chunk came from
    for page in doc_pages:
        page.metadata["source"] = pdf_file.name
    pages.extend(doc_pages)
    print(f"Loaded {pdf_file.name}: {len(doc_pages)} pages")

print(f"Total pages loaded: {len(pages)}")

# clean up spacing issues from PDF extraction
for page in pages:
    import re
    # add spaces between sentences that got merged
    page.page_content = re.sub(r'([a-z])([A-Z])', r'\1 \2', page.page_content)
    # normalize whitespace
    page.page_content = re.sub(r'\s+', ' ', page.page_content).strip()

print(f"Loaded {len(pages)} pages")

# --- chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=400)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# --- embed and build vector store
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

if os.path.exists("faiss_index"):
    print("Loading vector store from disk...")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("Building vector store from scratch...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("Vector store saved to disk")

print("Vector store ready\n")

# --- build BM25 sparse index
print("Building BM25 index...")
chunk_texts = [doc.page_content for doc in chunks]
tokenized_chunks = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)
print("BM25 index ready")


# --- LLM and prompt

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable assistant with expertise in both technology companies 
and sports regulations. You answer questions based strictly on two documents:
1. NVIDIA's official annual report (10-K) for fiscal year 2025
2. FIBA Official Basketball Rules 2024

Guidelines:
- Be conversational and clear, not robotic
- Structure longer answers with short paragraphs
- When citing numbers or facts, always mention which document they come from
- Always attribute basketball rules to "FIBA Official Basketball Rules 2024", never to NBA
- If the answer is not in the context, say "I don't have that information in the provided documents"
- Keep answers concise unless the question needs detail
- Never open with labels like "Answer:" or "Assistant:"
- Never add information from your own knowledge, only use the provided context

Context:
{context}

Conversation so far:
{history}

Question: {question}
""")

# --- RAG function
def ask(question, history):
    # --- dense retrieval
    dense_chunks = vectorstore.similarity_search(question, k=10)
    
    # --- sparse BM25 retrieval
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    sparse_chunks = [chunks[i] for i in top_bm25_indices]
    
    # --- combine and deduplicate
    seen = set()
    combined = []
    for doc in dense_chunks + sparse_chunks:
        key = doc.page_content[:50]
        if key not in seen:
            seen.add(key)
            combined.append(doc)
    
    # --- rerank combined results
    pairs = [[question, doc.page_content] for doc in combined]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, combined), key=lambda x: x[0], reverse=True)
    top_chunks = [doc for _, doc in ranked[:3]]
    
    context = "\n\n".join([doc.page_content for doc in top_chunks])
    
    history_text = ""
    if history:
        for h in history[-3:]:
            role = h.get('role', '')
            content = h.get('content', '')
            if isinstance(content, str):
                if role == 'user':
                    history_text += f"User: {content}\n"
                elif role == 'assistant':
                    content_clean = content.split('\n\n**Sources:**')[0]
                    history_text += f"Assistant: {content_clean}\n"

    chain = prompt | llm

    for attempt in range(3):
        try:
            response = chain.invoke({
                "context": context,
                "question": question,
                "history": history_text
            })
            sources = "\n\n**Sources:**\n" + "\n".join(
                [f"- {doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')+1}: {doc.page_content[:80]}..." 
                 for doc in top_chunks]
            )
            return response.content + sources
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < 2:
                print(f"Rate limit hit, waiting 20 seconds...")
                time.sleep(20)
            else:
                return f"Error: {str(e)}"

# --- launch Gradio UI
print("Launching UI at http://127.0.0.1:7860")
gr.ChatInterface(
    fn=ask,
    title="RAG Chatbot",
    description="Ask anything about your document.",
).launch()