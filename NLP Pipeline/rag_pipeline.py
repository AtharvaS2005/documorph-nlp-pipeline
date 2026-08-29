"""
RAG pipeline: chunking + embeddings + Qdrant indexing/retrieval.

Chunking: LangChain's RecursiveCharacterTextSplitter (chunk_size=1024,
chunk_overlap=100) -- content-agnostic, works on any extracted text
regardless of document type.

Embeddings: BAAI/bge-m3 via sentence-transformers -- a general multilingual
embedding model, not tuned to any specific document domain.

Vector DB: Qdrant, expected to be running separately (e.g. via Docker),
reachable at QDRANT_URL (default http://localhost:6333).

Retrieval: cosine similarity, top-k configurable (default 5).
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1024") or "1024")
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100") or "100")
TOP_K = int(os.getenv("RAG_TOP_K", "5") or "5")
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5") or "0.5")

_MODEL_CACHE: Dict[str, Any] = {}
_CLIENT_CACHE: Dict[str, Any] = {}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks. Content-agnostic -- works on any
    extracted text regardless of document type."""
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [c for c in splitter.split_text(text) if c.strip()]


def get_embedding_model():
    model = _MODEL_CACHE.get(EMBEDDING_MODEL_NAME)
    if model is not None:
        return model
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    _MODEL_CACHE[EMBEDDING_MODEL_NAME] = model
    return model


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()


def get_qdrant_client(url: str = QDRANT_URL):
    client = _CLIENT_CACHE.get(url)
    if client is not None:
        return client
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url)
    _CLIENT_CACHE[url] = client
    return client


def _ensure_collection(client, collection_name: str, vector_size: int) -> None:
    from qdrant_client.models import Distance, VectorParams

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic point id from doc_id + chunk_index, so re-indexing the
    same document overwrites its old chunks instead of duplicating them."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))


def index_document(
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: str = DEFAULT_COLLECTION,
    client_url: str = QDRANT_URL,
) -> int:
    """Chunk, embed, and upsert a document's text into Qdrant. Returns the
    number of chunks indexed. Re-indexing the same doc_id overwrites its
    previous chunks (deterministic point ids) rather than duplicating them."""
    from qdrant_client.models import PointStruct

    chunks = chunk_text(text)
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    client = get_qdrant_client(client_url)
    _ensure_collection(client, collection_name, vector_size=len(vectors[0]))

    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {"doc_id": doc_id, "chunk_index": idx, "text": chunk}
        if metadata:
            payload.update(metadata)
        points.append(PointStruct(id=_point_id(doc_id, idx), vector=vector, payload=payload))

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def retrieve(
    query: str,
    top_k: int = TOP_K,
    collection_name: str = DEFAULT_COLLECTION,
    client_url: str = QDRANT_URL,
) -> List[Dict[str, Any]]:
    """Retrieve the top_k most relevant chunks for query. Combines cosine
    similarity with a literal-substring match bonus so exact terms (names,
    IDs, invoice numbers) that pure semantic embeddings are weak at still
    rank correctly above semantically-similar-but-unrelated chunks."""
    if not query or not query.strip():
        return []

    client = get_qdrant_client(client_url)
    if not client.collection_exists(collection_name):
        return []

    query_vector = embed_texts([query])[0]
    candidate_limit = max(top_k * 4, 20)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=candidate_limit,
        with_payload=True,
    )

    query_lower = query.strip().lower()
    LITERAL_MATCH_BONUS = 0.35

    scored: List[Dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        text = payload.get("text", "")
        score = float(point.score)
        if query_lower and query_lower in text.lower():
            score += LITERAL_MATCH_BONUS
        scored.append(
            {
                "text": text,
                "score": score,
                "doc_id": payload.get("doc_id", ""),
                "chunk_index": payload.get("chunk_index", -1),
                "session_id": payload.get("session_id", ""),
                "file_name": payload.get("file_name", ""),
            }
        )

    scored.sort(key=lambda r: r["score"], reverse=True)
    filtered = [r for r in scored if r["score"] >= MIN_SCORE]
    return filtered[:top_k]

def delete_documents(
    doc_ids: List[str],
    collection_name: str = DEFAULT_COLLECTION,
    client_url: str = QDRANT_URL,
) -> int:
    """Remove every chunk belonging to the given doc_ids from a workspace's
    collection. Returns the number of doc_ids requested (Qdrant's filtered
    delete does not report a per-point count)."""
    if not doc_ids:
        return 0

    from qdrant_client.models import FieldCondition, Filter, MatchAny

    client = get_qdrant_client(client_url)
    if not client.collection_exists(collection_name):
        return 0

    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchAny(any=list(doc_ids)))]
        ),
    )
    return len(doc_ids)