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
    try:
        body = await request.json()
        messages = body.get("messages", [])

        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query:
            user_query = "Hello"

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
    except Exception as e:
        print(f"Vapi Custom LLM Error: {e}")
        return {
            "id": f"vapi-error-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I am experiencing a temporary connection issue reaching the document database."
                    },
                    "finish_reason": "stop"
                }
            ]
        }

# ---------- Direct REST Chat & Speech APIs & Vapi Tool Endpoints ----------

@app.post("/api/chat")
@app.post("/api/vapi/tool")
async def chat_endpoint(request: Request):
    """Direct RAG query endpoint supporting Web UI, Vapi Custom LLM, and Vapi Tool Calls."""
    try:
        body = await request.json()
        
        tool_call_id = None
        user_query = ""

        # Check if Vapi sent a Tool Call payload
        message_obj = body.get("message")
        if isinstance(message_obj, dict) and message_obj.get("type") == "tool-calls":
            tool_calls = message_obj.get("toolCalls", [])
            if tool_calls:
                tc = tool_calls[0]
                tool_call_id = tc.get("id")
                func_args = tc.get("function", {}).get("arguments", {})
                if isinstance(func_args, dict):
                    user_query = func_args.get("query") or func_args.get("message") or func_args.get("input") or ""
                elif isinstance(func_args, str):
                    user_query = func_args

        # Fallback extraction for direct message, query, or input
        if not user_query:
            user_query = body.get("message") or body.get("query") or body.get("input") or ""
            if isinstance(user_query, dict):
                user_query = user_query.get("content") or user_query.get("text") or ""
            elif not str(user_query).strip() and "message" in body and isinstance(body["message"], str):
                user_query = body["message"]

        if not str(user_query).strip():
            user_query = "What is in the company policy document?"

        model_name = body.get("model", "llama-3.3-70b-versatile")
        result = query_rag(str(user_query), model_name=model_name)
        reply_text = result.get("reply", "No relevant information found in documents.")

        # If Vapi invoked us as a Tool, return Vapi's required tool response format!
        if tool_call_id:
            return {
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "result": reply_text
                    }
                ]
            }

        # Otherwise return standard JSON response format for web UI
        return result
    except Exception as e:
        print(f"Chat Endpoint Error: {e}")
        return {
            "reply": "I am having trouble accessing documents right now.",
            "sources": [],
            "retrieved_chunks": [],
            "error": str(e)
        }

@app.post("/api/tts")
async def tts_endpoint(request: Request):
    """Generate natural voice audio stream (MP3) from text reply using gTTS."""
    try:
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            return JSONResponse({"error": "Message empty"}, status_code=400)

        tts = gTTS(text=message, lang="en", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(fp, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)

@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    try:
        delete_document_by_source(filename)
        return {"status": "success", "message": f"Deleted document '{filename}'"}
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

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
