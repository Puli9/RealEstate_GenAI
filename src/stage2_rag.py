"""
Stage 2: RAG System with Lazy Loading
API-safe - always gets fresh collection state
"""

import os
import PyPDF2
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from google import genai

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG_DOCS_DIR = os.path.join(BASE_DIR, "data", "docs", "regulatory_pdfs")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "vector_store")
COLLECTION_NAME = "realestate_docs"

os.makedirs(RAG_DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ==============================
# GEMINI CLIENT
# ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=GEMINI_API_KEY)
EMBED_MODEL = "models/text-embedding-004"

# ==============================
# LAZY CHROMADB CONNECTION
# ==============================
# Global variables for lazy initialization (must declare before use)
_chroma_client = None
_collection = None

def get_chroma_client():
    """Get or create ChromaDB client (lazy initialization)"""
    global _chroma_client
    if _chroma_client is None:
        print(f"📂 Connecting to ChromaDB at: {VECTOR_DB_DIR}")
        _chroma_client = chromadb.PersistentClient(
            path=VECTOR_DB_DIR
        )
    return _chroma_client

def get_collection(force_refresh: bool = False):
    """
    Get or create collection (lazy initialization)
    
    Args:
        force_refresh: Force reconnection to get latest state
    
    Returns:
        ChromaDB collection
    """
    global _collection
    
    if _collection is None or force_refresh:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(COLLECTION_NAME)
        count = _collection.count()
        print(f"✅ Collection '{COLLECTION_NAME}': {count} chunks")
    
    return _collection

# ==============================
# DOCUMENT LOADING
# ==============================
def list_pdf_and_text_files(folder: str) -> List[str]:
    """List all PDF/TXT/MD files recursively"""
    if not os.path.isdir(folder):
        return []
    
    files = []
    for root, dirs, files_in_dir in os.walk(folder):
        for f in files_in_dir:
            if f.lower().endswith(('.pdf', '.txt', '.md')):
                files.append(os.path.join(root, f))
    return files

def pdf_to_text(pdf_path: str) -> str:
    """Extract text from PDF"""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"⚠️ Error reading PDF {pdf_path}: {e}")
    return text.strip()

def load_docs() -> List[Dict[str, str]]:
    """Load all documents from RAG_DOCS_DIR"""
    docs = []
    file_paths = list_pdf_and_text_files(RAG_DOCS_DIR)
    
    if not file_paths:
        print(f"⚠️ No files found in {RAG_DOCS_DIR}")
        return []
    
    for path in file_paths:
        try:
            if path.lower().endswith('.pdf'):
                text = pdf_to_text(path)
                doc_id = os.path.basename(path).replace('.pdf', '')
            else:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                doc_id = os.path.basename(path)
            
            if text.strip():
                docs.append({
                    "id": doc_id,
                    "text": text,
                    "source_path": path
                })
                print(f"✅ Loaded: {os.path.basename(path)} ({len(text)} chars)")
            else:
                print(f"⚠️ Empty: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Failed: {path}: {e}")
    
    return docs

# ==============================
# CHUNKING & EMBEDDING
# ==============================
def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """Fixed-size chunking with overlap"""
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Gemini"""
    all_embeddings = []
    BATCH_SIZE = 50
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f"📦 Embedding batch {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1}...")
        
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
        )
        batch_embeds = [e.values for e in response.embeddings]
        all_embeddings.extend(batch_embeds)
    
    print(f"✅ Generated {len(all_embeddings)} embeddings")
    return all_embeddings

# ==============================
# INDEX BUILDING
# ==============================
def build_rag_index():
    """Build RAG index from documents"""
    print("\n" + "="*60)
    print("BUILDING RAG INDEX")
    print("="*60)
    
    # Get fresh collection
    collection = get_collection(force_refresh=True)
    
    # Load documents
    docs = load_docs()
    if not docs:
        print(f"❌ No documents in {RAG_DOCS_DIR}")
        return
    
    print(f"\n📚 Loaded {len(docs)} documents")
    
    # Prepare chunks
    all_ids, all_texts, all_metadatas = [], [], []
    
    for d in docs:
        chunks = chunk_text(d["text"], max_chars=800, overlap=100)
        print(f"  {d['id']}: {len(chunks)} chunks")
        
        for i, ch in enumerate(chunks):
            cid = f"{d['id']}_chunk_{i}"
            all_ids.append(cid)
            all_texts.append(ch)
            all_metadatas.append({
                "source": d["id"],
                "source_path": d.get("source_path", ""),
                "chunk_id": i
            })
    
    print(f"\n📊 Total chunks: {len(all_texts)}")
    
    # Generate embeddings
    print("\n🔢 Generating embeddings...")
    embeddings = embed_texts(all_texts)
    
    # Store in ChromaDB
    print("\n💾 Storing in ChromaDB...")
    collection.add(
        ids=all_ids,
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )
    
    print("\n" + "="*60)
    print("✅ INDEX BUILD COMPLETE")
    print("="*60)
    print(f"📁 Vector store: {VECTOR_DB_DIR}")
    print(f"📊 Total chunks: {collection.count()}")

# ==============================
# RETRIEVAL FUNCTION (API-SAFE)
# ==============================
def rag_query(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query RAG system (always gets fresh collection state)
    
    Args:
        query: Search query
        top_k: Number of results
    
    Returns:
        List of relevant chunks
    """
    # Always get fresh collection to avoid stale state
    collection = get_collection(force_refresh=False)
    
    # Check if empty
    count = collection.count()
    if count == 0:
        print(f"⚠️ WARNING: Collection is empty ({count} chunks)")
        print(f"   Run: python -m src.stage2_rag")
        return []
    
    # Generate query embedding
    q_emb = embed_texts([query])[0]
    
    # Query
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    # Format results
    docs = []
    if res["documents"] and res["documents"][0]:
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0]
        ):
            docs.append({
                "text": doc,
                "source": meta["source"],
                "source_path": meta.get("source_path", ""),
                "chunk_id": meta["chunk_id"],
                "distance": dist,
            })
    
    return docs

# ==============================
# STATUS CHECK
# ==============================
def check_rag_status():
    """Check RAG system status"""
    print("\n" + "="*60)
    print("RAG SYSTEM STATUS")
    print("="*60)
    
    print(f"📂 Vector DB: {VECTOR_DB_DIR}")
    print(f"   Exists: {os.path.exists(VECTOR_DB_DIR)}")
    
    print(f"\n📂 Documents: {RAG_DOCS_DIR}")
    print(f"   Exists: {os.path.exists(RAG_DOCS_DIR)}")
    if os.path.exists(RAG_DOCS_DIR):
        files = list_pdf_and_text_files(RAG_DOCS_DIR)
        print(f"   Files: {len(files)}")
        for f in files[:5]:
            print(f"     - {os.path.basename(f)}")
    
    # Get fresh collection
    collection = get_collection(force_refresh=True)
    print(f"\n🗂️ Collection: {COLLECTION_NAME}")
    print(f"   Chunks: {collection.count()}")
    
    if collection.count() > 0:
        sample = collection.get(limit=3)
        print(f"   Sample IDs: {sample['ids']}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("="*60)
    print("STAGE 2: RAG SYSTEM")
    print("="*60)
    
    check_rag_status()
    
    collection = get_collection()
    
    if collection.count() == 0:
        print("\n⚠️ Collection empty. Building index...")
        build_rag_index()
    else:
        print(f"\n✅ Collection already has {collection.count()} chunks")
        print("   Skipping rebuild. To rebuild, delete collection first.")
        # rebuild = input("\n🔄 Rebuild? (y/n): ").strip().lower()
        # if rebuild == 'y':
        #     client = get_chroma_client()
        #     client.delete_collection(COLLECTION_NAME)
        #     build_rag_index()
    
    # Test retrieval
    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60)
    
    test_query = (
        "Telangana RERA rules real estate regulation "
        "Hyderabad Madhapur residential property "
        "rental yield investment"
    )
    
    print(f"\nQuery: {test_query}")
    results = rag_query(test_query, top_k=5)
    print(f"\n✅ Retrieved {len(results)} chunks")
    
    if results:
        print("\n" + "="*60)
        print("TOP RESULTS")
        print("="*60)
        for i, chunk in enumerate(results, 1):
            print(f"\n📄 RESULT {i} (distance={chunk['distance']:.4f})")
            print(f"Source: {chunk['source']} (chunk {chunk['chunk_id']})")
            print(f"Preview: {chunk['text'][:200]}...")