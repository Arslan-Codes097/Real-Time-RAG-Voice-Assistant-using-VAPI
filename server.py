import os
import io
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from gtts import gTTS

from rag_pipeline import (
    query_rag,
    list_stored_documents,
    delete_document_by_source,
    process_and_add_uploaded_file
)

load_dotenv()

app = FastAPI(title="Arslan.AI - Real-Time Voice RAG Assistant (VAPI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model: str = "llama-3.3-70b-versatile"

class VapiMessage(BaseModel):
    role: str
    content: str

class VapiChatCompletionRequest(BaseModel):
    model: Optional[str] = "llama-3.3-70b-versatile"
    messages: List[Dict[str, Any]] = []

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "vapi_public_key": os.getenv("VAPI_PUBLIC_KEY", ""),
        "vapi_assistant_id": os.getenv("VAPI_ASSISTANT_ID", "")
    }

# ---------- Vapi Custom LLM / Server URL Endpoint ----------

@app.post("/vapi/chat/completions")
@app.post("/api/vapi/chat/completions")
async def vapi_custom_llm_endpoint(request: Request):
    """
    OpenAI-compatible Chat Completion endpoint for Vapi Custom LLM Integration.
    Vapi Dashboard -> Assistant -> Model Provider -> Custom LLM -> Set Server URL to this endpoint.
    """
    body = await request.json()
    messages = body.get("messages", [])

    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        user_query = "Hello"

    # Execute LangChain RAG pipeline
    rag_result = query_rag(user_query)
    reply_text = rag_result.get("reply", "I am unable to answer based on the available documents.")

    return {
        "id": f"vapi-rag-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply_text
                },
                "finish_reason": "stop"
            }
        ]
    }

# ---------- Direct REST Chat & Speech APIs ----------

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Direct RAG query endpoint for text & voice UI."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = query_rag(request.message, model_name=request.model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts")
def tts_endpoint(request: ChatRequest):
    """Generate natural voice audio stream (MP3) from text reply using gTTS."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message empty.")

    try:
        tts = gTTS(text=request.message, lang="en", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(fp, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")

# ---------- RAG Document Management Endpoints ----------

@app.get("/api/documents")
def get_documents():
    try:
        docs = list_stored_documents()
        return {"documents": docs}
    except Exception as e:
        return {"documents": [], "error": str(e)}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        res = process_and_add_uploaded_file(contents, file.filename)
        return {"status": "success", "message": f"Successfully ingested {file.filename}", "details": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    try:
        delete_document_by_source(filename)
        return {"status": "success", "message": f"Deleted document '{filename}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Bulletproof Static Asset Handlers ----------

def find_file(filename: str):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_dir)
    search_paths = [
        os.path.join(curr_dir, "public", filename),
        os.path.join(curr_dir, "static", filename),
        os.path.join(parent_dir, "public", filename),
        os.path.join(parent_dir, "static", filename),
        os.path.join(curr_dir, filename),
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    return None

@app.get("/")
def read_root():
    index_path = find_file("index.html")
    if index_path:
        return FileResponse(index_path)
    return JSONResponse({"message": "Arslan.AI Voice RAG (VAPI) API Server Operating."})

@app.get("/style.css")
def get_style():
    style_path = find_file("style.css")
    if style_path:
        return FileResponse(style_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")

@app.get("/app.js")
def get_script():
    script_path = find_file("app.js")
    if script_path:
        return FileResponse(script_path, media_type="text/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")

@app.get("/favicon.ico")
@app.get("/favicon.png")
def get_favicon():
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
