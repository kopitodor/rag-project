from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from pathlib import Path
import re
import os
import time

load_dotenv()

# --- Load PDFs
print("Loading PDFs...")
pdf_files = list(Path(".").glob("*.pdf"))
print(f"Found {len(pdf_files)} PDFs: {[f.name for f in pdf_files]}")

pages = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    doc_pages = loader.load()
    for page in doc_pages:
        page.metadata["source"] = pdf_file.name
    pages.extend(doc_pages)
    print(f"Loaded {pdf_file.name}: {len(doc_pages)} pages")

for page in pages:
    page.page_content = re.sub(r'([a-z])([A-Z])', r'\1 \2', page.page_content)
    page.page_content = re.sub(r'\s+', ' ', page.page_content).strip()

print(f"Total pages loaded: {len(pages)}")

# --- Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=400)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# --- Embed + FAISS
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

# --- BM25
print("Building BM25 index...")
chunk_texts = [doc.page_content for doc in chunks]
tokenized_chunks = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)
print("BM25 index ready")

# --- LLM + Prompt
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
- Only make claims that are directly and explicitly stated in the context above
- Do not add explanatory context, comparisons, or background knowledge even if true
- If you find yourself writing something not in the context, stop and omit it

Context:
{context}

Conversation so far:
{history}

Question: {question}
""")

# --- Core retrieval function
def retrieve_and_answer(question: str, history_text: str = "") -> tuple[str, list]:
    """
    Runs the full hybrid retrieval + rerank + LLM pipeline.
    Returns (answer_text, top_chunks).
    """
    # Dense retrieval
    dense_chunks = vectorstore.similarity_search(question, k=10)

    # Sparse BM25 retrieval
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    sparse_chunks = [chunks[i] for i in top_bm25_indices]

    # Combine and deduplicate
    seen = set()
    combined = []
    for doc in dense_chunks + sparse_chunks:
        key = doc.page_content[:50]
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    # Rerank
    pairs = [[question, doc.page_content] for doc in combined]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, combined), key=lambda x: x[0], reverse=True)
    top_chunks = [doc for _, doc in ranked[:3]]

    # Generate answer
    context = "\n\n".join([doc.page_content for doc in top_chunks])
    chain = prompt | llm

    for attempt in range(3):
        try:
            response = chain.invoke({
                "context": context,
                "question": question,
                "history": history_text
            })
            return response.content, top_chunks
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < 2:
                print(f"Rate limit hit, waiting 20 seconds...")
                time.sleep(20)
            else:
                return f"Error: {str(e)}", []

def get_answer(question: str) -> tuple[str, list[str]]:
    """For the eval harness. Returns (answer, context_strings)."""
    answer, top_chunks = retrieve_and_answer(question)
    return answer, [doc.page_content for doc in top_chunks]