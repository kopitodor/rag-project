from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import gradio as gr
import time
import os

load_dotenv()

# --- load and clean the PDF
print("Loading PDF...")
loader = PyPDFLoader("document.pdf")
pages = loader.load()

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

if os.path.exists("faiss_index"):
    print("Loading vector store from disk...")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("Building vector store from scratch...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("Vector store saved to disk")

print("Vector store ready\n")

# debug — search for employee count directly
test = vectorstore.similarity_search("approximately 36000 employees 38 countries", k=1)
print(repr(test[0].page_content))
for i, doc in enumerate(test):
    print(f"[{i+1}] Page {doc.metadata.get('page', '?')+1}: {doc.page_content[:150]}")
print("---")

# --- LLM and prompt

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable financial analyst assistant with deep expertise 
in technology companies. You answer questions about NVIDIA based on their 
official filings and reports.

Guidelines:
- Be conversational and clear, not robotic
- Structure longer answers with short paragraphs
- When citing numbers or facts, mention they come from the document
- If the answer is not in the context, say "I don't have that information 
  in the provided documents"
- Keep answers concise unless the question needs detail
- In the begginig of an answer, answer fluently. No need openings
like "Answer:" or "Asistant:"

Context:
{context}

Conversation so far:
{history}

Question: {question}
""")

# --- RAG function
def ask(question, history):
    retrieved_chunks = vectorstore.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])
    
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
                [f"- Page {doc.metadata.get('page', '?')+1}: {doc.page_content[:80]}..." 
                 for doc in retrieved_chunks]
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