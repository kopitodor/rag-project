from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import retrieve_and_answer

app = FastAPI(
    title="RAG Chatbot API",
    description="REST API for the RAG pipeline over NVIDIA 10-K and FIBA rules",
    version="1.0.0"
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    answer, top_chunks = retrieve_and_answer(request.question)
    
    sources = [
        {
            "source": doc.metadata.get("source", "?"),
            "page": doc.metadata.get("page", 0) + 1,
            "snippet": doc.page_content[:120]
        }
        for doc in top_chunks
    ]
    
    return AskResponse(answer=answer, sources=sources)