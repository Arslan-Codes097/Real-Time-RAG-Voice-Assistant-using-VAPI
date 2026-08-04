import os
import sys
import uuid
import glob
from typing import List
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from pypdf import PdfReader
import docx

DOCS_DIRECTORY = os.path.join(os.path.dirname(__file__), "sample_docs")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

class FastEmbedWrapper:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

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

embeddings = FastEmbedWrapper()

def load_pdf(file_path: str) -> List[Document]:
    reader = PdfReader(file_path)
    docs = []
    filename = os.path.basename(file_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "source_file": filename, "page": i + 1}
                )
            )
    return docs

def load_docx(file_path: str) -> List[Document]:
    doc = docx.Document(file_path)
    full_text = []
    filename = os.path.basename(file_path)
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    content = "\n".join(full_text)
    if content.strip():
        return [Document(page_content=content, metadata={"source": filename, "source_file": filename})]
    return []

def load_txt(file_path: str) -> List[Document]:
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if content.strip():
        return [Document(page_content=content, metadata={"source": filename, "source_file": filename})]
    return []

def load_all_documents(folder_path: str) -> List[Document]:
    documents = []
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            print(f"📄 Loading PDF: {os.path.basename(file_path)}")
            documents.extend(load_pdf(file_path))
        elif ext == ".docx":
            print(f"📝 Loading DOCX: {os.path.basename(file_path)}")
            documents.extend(load_docx(file_path))
        elif ext in [".txt", ".md"]:
            print(f"📑 Loading Text: {os.path.basename(file_path)}")
            documents.extend(load_txt(file_path))
    return documents

def ingest_documents():
    print("🚀 Starting Standard LangChain Document Ingestion...")
    raw_docs = load_all_documents(DOCS_DIRECTORY)
    print(f"✅ Loaded {len(raw_docs)} document sections/pages.")

    if not raw_docs:
        print("⚠️ No valid documents found in sample_docs directory.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"🧩 Split into {len(chunks)} text chunks.")

    print("🧠 Generating Embeddings in Batches...")
    chunk_texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(chunk_texts)

    if SUPABASE_URL and SUPABASE_KEY and "your_supabase" not in SUPABASE_URL:
        print("⚡ Upserting vector chunks into Supabase Cloud Vector DB...")
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        records = []
        for i in range(len(chunks)):
            source_file = chunks[i].metadata.get("source_file") or chunks[i].metadata.get("source")
            records.append({
                "id": str(uuid.uuid4()),
                "content": chunk_texts[i],
                "metadata": {"source_file": source_file, "chunk_index": i},
                "embedding": vectors[i]
            })

        batch_size = 50
        for idx in range(0, len(records), batch_size):
            batch = records[idx:idx + batch_size]
            supabase.table("documents").upsert(batch).execute()
        print("🎉 Ingestion into Supabase Vector DB Complete!")

if __name__ == "__main__":
    ingest_documents()
