import re
import chromadb
from pathlib import Path
from typing import List, Dict, Any
# CẤU HÌNH
DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

CHUNK_SIZE = 400       # tokens ước lượng
CHUNK_OVERLAP = 80     # tokens overlap

# STEP 1: PREPROCESS - Trích xuất Metadata
def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": Path(filepath).name,
        "section": "General",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    
    content_start_idx = 0
    for i, line in enumerate(lines):
        match = re.match(r"^(Source|Department|Effective Date|Access):\s*(.*)$", line, re.IGNORECASE)
        if match:
            key = match.group(1).lower().replace(" ", "_")
            value = match.group(2).strip()
            metadata[key] = value
            content_start_idx = i + 1
        elif line.startswith("==="):
            content_start_idx = i
            break
            
    content_lines = lines[content_start_idx:]
    cleaned_text = "\n".join(content_lines).strip()
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text) 

    return {"text": cleaned_text, "metadata": metadata}

# STEP 2: CHUNK - Cắt văn bản thông minh (Paragraph Chunking)
def _split_by_size(text: str, base_metadata: Dict, section: str) -> List[Dict[str, Any]]:
    chunk_chars = CHUNK_SIZE * 4
    overlap_chars = CHUNK_OVERLAP * 4
    
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk_text = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        
        if len(current_chunk_text) + len(para) > chunk_chars and current_chunk_text:
            chunks.append({
                "text": current_chunk_text.strip(),
                "metadata": {**base_metadata, "section": section}
            })
            current_chunk_text = current_chunk_text[-overlap_chars:] + "\n\n" + para
        else:
            if current_chunk_text:
                current_chunk_text += "\n\n" + para
            else:
                current_chunk_text = para

    if current_chunk_text:
        chunks.append({
            "text": current_chunk_text.strip(),
            "metadata": {**base_metadata, "section": section}
        })
        
    return chunks

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    parts = re.split(r"(===.*?===)", text)
    current_section = "General"
    
    for i in range(len(parts)):
        part = parts[i].strip()
        if not part: continue
        
        if re.match(r"===.*?===", part):
            current_section = part.replace("=", "").strip()
        else:
            section_chunks = _split_by_size(part, base_metadata, current_section)
            chunks.extend(section_chunks)
            
    return chunks

# STEP 3: EMBED + STORE
def get_embedding(text: str) -> List[float]:
    from sentence_transformers import SentenceTransformer
    if not hasattr(get_embedding, "_model"):
        get_embedding._model = SentenceTransformer("all-MiniLM-L6-v2")
    return get_embedding._model.encode(text, normalize_embeddings=True).tolist()

def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    print(f"Bắt đầu build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(db_dir))
    # Xóa collection cũ nếu tồn tại (tránh lỗi dimension mismatch khi đổi model)
    try:
        client.delete_collection("rag_lab")
    except Exception:
        pass
    collection = client.create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"}
    )

    doc_files = list(docs_dir.glob("*.txt"))
    total_chunks = 0

    for filepath in doc_files:
        print(f"Đang xử lý: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")
        
        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}"
            embedding = get_embedding(chunk["text"])
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            
        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            total_chunks += len(ids)

    print(f"\nHoàn thành! Đã index {total_chunks} chunks.")

# STEP 4: INSPECT / KIỂM TRA (Phục vụ DoD số 3)
def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 3) -> None:
    print("\n" + "="*50)
    print("KIỂM TRA CHẤT LƯỢNG CHUNK (Definition of Done 3)")
    print("="*50)
    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"\n[Chunk {i+1}]")
            print(f"  Source: {meta.get('source')} | Section: {meta.get('section')}")
            print(f"  Date: {meta.get('effective_date')}")
            print(f"  Text preview: {doc[:150]}...\n")
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")

# MAIN
if __name__ == "__main__":
    build_index()
    list_chunks()