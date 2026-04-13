from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import gradio as gr
import time

load_dotenv()

# --- load the PDF
print("Loading PDF...")
loader = PyPDFLoader("document.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# --- chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# --- embed and build vector store
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store ready\n")

# --- LLM and prompt
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
""")

# --- RAG function
def ask(question, history):
    retrieved_chunks = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])
    chain = prompt | llm
    
    for attempt in range(3):
        try:
            response = chain.invoke({"context": context, "question": question})
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