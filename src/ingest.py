import os
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any

import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR    = Path("data/raw")
CHROMA_DIR  = Path("chroma_db")
COLLECTION  = "nasa_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE  = 512
OVERLAP     = 80

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text(pdf_path: Path) -> Dict:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        if text.strip():
            pages.append({"page": page_num, "text": text.strip()})
    meta = {
        "source": pdf_path.name,
        "title": doc.metadata.get("title", pdf_path.stem) or pdf_path.stem,
        "author": doc.metadata.get("author", "NASA") or "NASA",
        "total_pages": len(doc),
    }
    doc.close()
    return {"pages": pages, "metadata": meta}


def chunk_text(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    chunks, current, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) > CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            overlap, olen = [], 0
            for sent in reversed(current):
                if olen + len(sent) <= OVERLAP:
                    overlap.insert(0, sent)
                    olen += len(sent)
                else:
                    break
            current, cur_len = overlap, olen
        current.append(s)
        cur_len += len(s)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) > 50]


def ingest_all(reset: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION)
            logger.info("Deleted existing collection")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer(EMBED_MODEL)
    logger.info(f"Embedding model loaded: {EMBED_MODEL}")

    pdfs = list(DATA_DIR.glob("**/*.pdf"))
    if not pdfs:
        logger.error(f"No PDFs found in {DATA_DIR}")
        return
    logger.info(f"Found {len(pdfs)} PDF files")

    existing = set()
    if collection.count() > 0:
        try:
            offset = 0
            while True:
                result = collection.get(
                    limit=500,
                    offset=offset,
                    include=["metadatas"]
                )
                if not result["metadatas"]:
                    break
                for m in result["metadatas"]:
                    existing.add(m["source"])
                offset += 500
                if len(result["metadatas"]) < 500:
                    break
        except Exception:
            existing = set()

    total = 0
    for pdf_path in tqdm(pdfs, desc="Ingesting"):
        if pdf_path.name in existing:
            continue
        try:
            data = extract_text(pdf_path)
            chunks = []
            for p in data["pages"]:
                for i, chunk in enumerate(chunk_text(p["text"])):
                    cid = hashlib.md5(
                        f"{pdf_path.name}_{p['page']}_{i}".encode()
                    ).hexdigest()
                    chunks.append({
                        "id": cid,
                        "text": chunk,
                        "meta": {
                            "source": data["metadata"]["source"],
                            "title": data["metadata"]["title"],
                            "author": data["metadata"]["author"],
                            "page": p["page"],
                            "total_pages": data["metadata"]["total_pages"],
                        }
                    })
            for i in range(0, len(chunks), 64):
                batch = chunks[i:i+64]
                embeddings = model.encode(
                    [c["text"] for c in batch],
                    show_progress_bar=False
                ).tolist()
                collection.add(
                    ids=[c["id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    embeddings=embeddings,
                    metadatas=[c["meta"] for c in batch],
                )
            total += len(chunks)
            logger.info(f"  Done: {pdf_path.name} — {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"  Failed: {pdf_path.name} — {e}")

    print(f"\nDone! {collection.count()} total chunks in database.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()
    ingest_all(reset=args.reset)


