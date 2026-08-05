import os
import sys
import uuid
import tempfile
from typing import Dict, List, Any
from dotenv import load_dotenv

# Environment configuration for serverless performance
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.vectorstores import SupabaseVectorStore

_hf_embeddings_instance = None

def get_embeddings():
    global _hf_embeddings_instance
    if _hf_embeddings_instance is None:
        try:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            hf_token = os.getenv("HF_TOKEN", "")
            if not hf_token:
                print("HF_TOKEN is missing in environment variables!")
                return None
            _hf_embeddings_instance = HuggingFaceEndpointEmbeddings(
                model="BAAI/bge-small-en-v1.5",
                huggingfacehub_api_token=hf_token
            )
        except Exception as e:
            print(f"HuggingFace Embeddings Initialization Note: {e}")
            _hf_embeddings_instance = False
    return _hf_embeddings_instance

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
    """Execute LangChain RAG pipeline against Supabase Vector Database using standard vector store."""
    if not question.strip():
        return {"reply": "I didn't hear a question.", "sources": []}

    retrieved_chunks = []
    sources = []
    supabase_client = get_supabase_client()
    embeddings = get_embeddings()

    if supabase_client and embeddings:
        try:
            vector_store = SupabaseVectorStore(
                client=supabase_client,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents",
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(question)
            
            for doc in docs:
                retrieved_chunks.append(doc.page_content)
                src = doc.metadata.get("source_file") or doc.metadata.get("source") or "Document"
                if src not in sources:
                    sources.append(src)
        except Exception as e:
            print(f"Retrieval error: {e}")

    if not retrieved_chunks:
        context_text = "No relevant document context found."
    else:
        context_text = "\n\n".join(retrieved_chunks)

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

    supabase_client = get_supabase_client()
    embeddings = get_embeddings()
    
    if supabase_client and embeddings:
        try:
            delete_document_by_source(filename)
        except Exception:
            pass
            
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents",
        )
        
        # Add a chunk_index to metadata
        for i, chunk in enumerate(chunks):
            if "source_file" not in chunk.metadata:
                chunk.metadata["source_file"] = filename
            chunk.metadata["chunk_index"] = i
            
        vector_store.add_documents(chunks)

    try:
        os.remove(temp_path)
        os.rmdir(temp_dir)
    except Exception:
        pass

    return {"filename": filename, "chunks_added": len(chunks)}
