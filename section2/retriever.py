from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME
from .embeddings import embed_query, embed_texts
from .ingest import ChunkRecord, extract_chunks


STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "is",
    "with",
    "what",
    "which",
    "how",
    "under",
    "in",
    "for",
    "on",
    "and",
    "this",
    "that",
}


@dataclass(slots=True)
class RetrievedChunk:
    document: str
    page: int
    clause: str
    chunk: str
    score: float


class ContractRetriever:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self._sync_index()

    def retrieve(self, question: str, top_k: int = 3, candidate_k: int = 8) -> list[RetrievedChunk]:
        query_embedding = embed_query(question)
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        results: list[RetrievedChunk] = []
        for doc_text, metadata, distance in zip(docs, metadatas, distances, strict=True):
            semantic = 1.0 / (1.0 + float(distance))
            rerank = _rerank_score(question, doc_text, metadata["document"], metadata["clause"])
            final_score = 0.7 * semantic + 0.3 * rerank
            results.append(
                RetrievedChunk(
                    document=str(metadata["document"]),
                    page=int(metadata["page"]),
                    clause=str(metadata["clause"]),
                    chunk=doc_text,
                    score=final_score,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def all_chunks(self) -> list[RetrievedChunk]:
        payload = self.collection.get(include=["documents", "metadatas"])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        items: list[RetrievedChunk] = []
        for doc_text, metadata in zip(documents, metadatas, strict=True):
            items.append(
                RetrievedChunk(
                    document=str(metadata["document"]),
                    page=int(metadata["page"]),
                    clause=str(metadata["clause"]),
                    chunk=doc_text,
                    score=0.0,
                )
            )
        return items

    def _sync_index(self) -> None:
        chunks = extract_chunks()
        manifest_path = CHROMA_DIR / "index_manifest.json"
        signature = _chunk_signature(chunks)
        existing = self.collection.count()
        manifest = _load_manifest(manifest_path)
        if existing == len(chunks) and manifest.get("signature") == signature:
            return
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self._add_chunks(chunks)
        manifest_path.write_text(
            json.dumps({"chunk_count": len(chunks), "signature": signature}, indent=2),
            encoding="utf-8",
        )

    def _add_chunks(self, chunks: Iterable[ChunkRecord]) -> None:
        chunk_list = list(chunks)
        embeddings = embed_texts(record.text for record in chunk_list)
        self.collection.add(
            ids=[record.chunk_id for record in chunk_list],
            documents=[record.text for record in chunk_list],
            metadatas=[
                {
                    "document": record.document,
                    "page": record.page,
                    "clause": record.clause,
                }
                for record in chunk_list
            ],
            embeddings=embeddings,
        )


def _rerank_score(question: str, text: str, document: str, clause: str) -> float:
    query_tokens = _tokenize(question)
    text_tokens = _tokenize(text)
    metadata_tokens = _tokenize(f"{document} {clause}".replace("_", " "))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens) / len(query_tokens)
    metadata_overlap = len(query_tokens & metadata_tokens) / len(query_tokens)
    phrase_bonus = _metadata_phrase_bonus(question, document, clause)
    return min(1.0, 0.65 * overlap + 0.25 * metadata_overlap + phrase_bonus)


def _tokenize(text: str) -> set[str]:
    parts = re.findall(r"[a-zA-Z0-9₹]+", text.lower())
    return {part for part in parts if part not in STOPWORDS}


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _chunk_signature(chunks: Iterable[ChunkRecord]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        payload = json.dumps(
            {
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "page": chunk.page,
                "clause": chunk.clause,
                "text": chunk.text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def _metadata_phrase_bonus(question: str, document: str, clause: str) -> float:
    metadata = _normalized_phrase_space(f"{document} {clause}".replace("_", " "))
    question_tokens = [token for token in _normalized_phrase_space(question).split() if token]
    bonus = 0.0
    for size, weight in ((3, 0.18), (2, 0.12)):
        for index in range(0, max(0, len(question_tokens) - size + 1)):
            phrase = " ".join(question_tokens[index : index + size])
            if phrase and phrase in metadata:
                bonus = max(bonus, weight)
    return bonus


def _normalized_phrase_space(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9₹]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()
