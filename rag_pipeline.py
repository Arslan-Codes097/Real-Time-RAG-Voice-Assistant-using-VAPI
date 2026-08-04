import os
import sys
import uuid
import tempfile
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastembed import TextEmbedding
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Safe FastEmbed initialization with /tmp cache path
os.environ["FASTEMBED_CACHE_PATH"] = os.path.join(tempfile.gettempdir(), "fastembed_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_fastembed_instance = None

def get_embeddings():
    global _fastembed_instance
    if _fastembed_instance is None:
        class FastEmbedWrapper:
            def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
                self.model = TextEmbedding(model_name=model_name, cache_dir=os.environ["FASTEMBED_CACHE_PATH"])

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                results = []
                batch_size = 16
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    embeddings = list(self.model.embed(batch))
                    for vec in embeddings:
                        results.append([float(v) for v in vec])
                return results

            def embed_query(self, text: str) -> List[float]:
                return [float(v) for v in next(self.model.embed([text]))]

        _fastembed_instance = FastEmbedWrapper()
    return _fastembed_instance

def get_supabase_client():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if url and key and "your_supabase" not in url:
        try:
            from supabase import create_client
            return create_client(url, key)
        except Exception as e:
            print(f"Supabase Client Connection Error: {e}")
    return None

def get_groq_llm(model_name: str = "llama-3.3-70b-versatile"):
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is missing in environment variables!")
    return ChatGroq(model=model_name, groq_api_key=groq_key, temperature=0.3)

VOICE_SYSTEM_PROMPT = """You are Arslan.AI, an intelligent real-time voice assistant.
Answer the user's question using ONLY the retrieved document context below.

CRITICAL VOICE RULES:
1. Keep your reply concise, natural, and conversational (1 to 3 short sentences max).
2. Do NOT use markdown, bullet points, asterisks, bold text, code blocks, or special formatting.
3. Speak directly as a human assistant would in a spoken phone call.
4. If the context does not contain the answer, say "I don't have that specific information in my documents."

Context:
{context}

Question:
{question}
"""

def query_rag(question: str, model_name: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
    """Execute LangChain RAG pipeline against Supabase Vector Database with hybrid fallback."""
    if not question.strip():
        return {"reply": "I didn't hear a question.", "sources": []}

    retrieved_chunks = []
    sources = []
    supabase_client = get_supabase_client()
    embeddings = get_embeddings()

    if supabase_client:
        try:
            query_vector = embeddings.embed_query(question)
            rpc_params = {
                "query_embedding": query_vector,
                "match_count": 3,
                "filter": {}
            }
            res = supabase_client.rpc("match_documents", rpc_params).execute()
            rows = res.data or []

            # If RPC returns zero rows, use keyword content search fallback
            if not rows:
                keywords = [w for w in question.split() if len(w) > 3]
                if keywords:
                    kw = keywords[0]
                    text_res = supabase_client.table("documents").select("content, metadata").ilike("content", f"%{kw}%").limit(3).execute()
                    rows = text_res.data or []

            for r in rows:
                snippet = r.get("content", "").strip()
                if snippet:
                    retrieved_chunks.append(snippet)
                    meta = r.get("metadata") or {}
                    src = meta.get("source_file") or meta.get("source") or "Document"
                    if src not in sources:
                        sources.append(src)
            context_text = "\n\n".join(retrieved_chunks)
        except Exception as e:
            print(f"Supabase search error: {e}")
            context_text = ""
    else:
        context_text = "No document database connection configured."

    if not context_text:
        context_text = "No relevant document context found."

    prompt = ChatPromptTemplate.from_template(VOICE_SYSTEM_PROMPT)
    llm = get_groq_llm(model_name)

    chain = (
        {"context": lambda x: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    reply = chain.invoke(question)

    return {
        "reply": reply,
        "sources": sources,
        "retrieved_chunks": [c[:200] + "..." for c in retrieved_chunks]
    }

# ---------- Document Management Functions ----------

def list_stored_documents() -> List[Dict[str, Any]]:
    supabase_client = get_supabase_client()
    if supabase_client:
        try:
            res = supabase_client.table("documents").select("metadata").execute()
            sources_map = {}
            for row in res.data or []:
                meta = row.get("metadata") or {}
                src = meta.get("source_file") or meta.get("source")
                if src:
                    sources_map[src] = sources_map.get(src, 0) + 1
            return [{"source": src, "chunk_count": count} for src, count in sources_map.items()]
        except Exception as e:
            print(f"Error listing Supabase documents: {e}")
            return []
    return []

def delete_document_by_source(source_name: str) -> bool:
    supabase_client = get_supabase_client()
    if supabase_client:
        try:
            supabase_client.table("documents").delete().eq("metadata->>source_file", source_name).execute()
            return True
        except Exception as e:
            print(f"Error deleting from Supabase: {e}")
            return False
    return False

def process_and_add_uploaded_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    from ingest import load_pdf, load_docx, load_txt

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        raw_docs = load_pdf(temp_path)
    elif ext == ".docx":
        raw_docs = load_docx(temp_path)
    elif ext in [".txt", ".md"]:
        raw_docs = load_txt(temp_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Allowed: .pdf, .docx, .txt")

    if not raw_docs:
        raise ValueError("File appears to be empty or unreadable.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    chunk_texts = [c.page_content for c in chunks]
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(chunk_texts)

    supabase_client = get_supabase_client()
    if supabase_client:
        try:
            delete_document_by_source(filename)
        except Exception:
            pass

        records = []
        for i in range(len(chunks)):
            records.append({
                "id": str(uuid.uuid4()),
                "content": chunk_texts[i],
                "metadata": {"source_file": filename, "chunk_index": i},
                "embedding": vectors[i]
            })

        batch_size = 50
        for idx in range(0, len(records), batch_size):
            batch = records[idx:idx + batch_size]
            supabase_client.table("documents").upsert(batch).execute()

    try:
        os.remove(temp_path)
        os.rmdir(temp_dir)
    except Exception:
        pass

    return {"filename": filename, "chunks_added": len(chunks)}
