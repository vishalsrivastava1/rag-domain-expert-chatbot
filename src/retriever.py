import re, math, logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR  = Path("chroma_db")
COLLECTION  = "nasa_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25:
    """Keyword search using BM25 scoring (like Google but simple)."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.avg_dl = 0
        self.doc_ids = []

    def tokenize(self, text):
        return re.findall(r"\b[a-zA-Z][a-zA-Z0-9]{2,}\b", text.lower())

    def fit(self, documents, doc_ids):
        self.doc_ids = doc_ids
        tokenized = [self.tokenize(d) for d in documents]
        self.corpus = tokenized
        df = defaultdict(int)
        total_len = 0
        for tokens in tokenized:
            total_len += len(tokens)
            for w in set(tokens): df[w] += 1
        N = len(documents)
        self.avg_dl = total_len / N if N else 1
        for w, freq in df.items():
            self.idf[w] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        self.doc_freqs = [dict(defaultdict(int,
            {t: tokens.count(t) for t in set(tokens)}))
            for tokens in tokenized]

    def score(self, query, top_k=20):
        scores = defaultdict(float)
        for token in self.tokenize(query):
            if token not in self.idf: continue
            idf = self.idf[token]
            for idx, doc_id in enumerate(self.doc_ids):
                tf = self.doc_freqs[idx].get(token, 0)
                dl = len(self.corpus[idx])
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[doc_id] += idf * num / den
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class HybridRetriever:
    """Combines BM25 + semantic search via Reciprocal Rank Fusion."""

    def __init__(self, top_k=5, rrf_k=60):
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(COLLECTION)
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(EMBED_MODEL)
        self._build_bm25()

    def _build_bm25(self):
        """Load all docs from ChromaDB and build BM25 index."""
        logger.info("Building BM25 index...")
        total = self.collection.count()
        all_docs, all_ids = [], []
        for offset in range(0, total, 1000):
            r = self.collection.get(
                limit=1000, offset=offset, include=["documents"]
            )
            all_docs.extend(r["documents"])
            all_ids.extend(r["ids"])
        self.bm25 = BM25()
        self.bm25.fit(all_docs, all_ids)
        logger.info(f"BM25 index ready: {len(all_ids)} chunks")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Run hybrid search and return top results with metadata."""
        k = top_k or self.top_k
        candidates = max(k * 4, 20)

        # 1. BM25 keyword search
        bm25_results = self.bm25.score(query, top_k=candidates)

        # 2. Semantic vector search
        q_emb = self.model.encode(query).tolist()
        sem = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(candidates, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        sem_ranking = [
            sid for sid in sem["ids"][0]
      ]

        # 3. Reciprocal Rank Fusion (RRF)
        scores = defaultdict(float)

        for rank, (doc_id, _) in enumerate(bm25_results, 1):
            scores[doc_id] += 1.0 / (self.rrf_k + rank)

        for rank, doc_id in enumerate(sem_ranking, 1):
            scores[doc_id] += 1.0 / (self.rrf_k + rank)

        top_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]

        if not top_ids:
            return []

        # 4. Fetch full data for top results
        fetched = self.collection.get(
            ids=top_ids, include=["documents", "metadatas"]
        )
        results = []
        for doc_id, text, meta in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"]
        ):
            results.append({
                "text": text,
                "source": meta.get("source", "Unknown"),
                "title": meta.get("title", "Unknown"),
                "author": meta.get("author", "NASA"),
                "page": meta.get("page", 1),
                "total_pages": meta.get("total_pages", "?"),
                "score": round(scores[doc_id], 5),
            })
        # return in score order
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def get_stats(self):
        return {"total_chunks": self.collection.count()}


# Singleton — load once, reuse across requests
_retriever: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    retriever = get_retriever()

    query = "What is NASA's Artemis program?"
    results = retriever.retrieve(query, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Title:", r["title"])
        print("Source:", r["source"])
        print("Page:", r["page"])
        print("Score:", r["score"])
        print("Text preview:", r["text"][:200])