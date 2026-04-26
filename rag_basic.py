import gradio as gr
from pipeline import retrieve_and_answer

def ask(question, history):
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

    answer, top_chunks = retrieve_and_answer(question, history_text)

    sources = "\n\n**Sources:**\n" + "\n".join(
        [f"- {doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')+1}: {doc.page_content[:80]}..."
         for doc in top_chunks]
    )
    return answer + sources

if __name__ == "__main__":
    print("Launching UI at http://127.0.0.1:7860")
    gr.ChatInterface(
        fn=ask,
        title="RAG Chatbot",
        description="Ask anything about your document.",
    ).launch()