"""
Semantic Tool Retriever Module for inu-agent-core
In-Memory Tool Vector Store with L2 Cosine Similarity and Dynamic Hybrid Scoring (Zero-Hardcoding Tool RAG).
Replaces static keyword dictionaries and magic number scoring heuristics.
"""
from typing import List, Dict, Any, Optional
import asyncio
import re
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.tools.base import BaseTool
from app.llm.embeddings import get_embeddings, FastEmbedBackend, MockEmbeddingBackend


def serialize_tool(tool: BaseTool) -> str:
    """
    Serialize a BaseTool instance into a clean semantic document for vector embedding.
    Uses the tool's natural human-readable description.
    """
    return (tool.description or tool.name).strip()


def extract_tool_keywords(tool: BaseTool) -> str:
    """
    Extract searchable text for dynamic lexical matching (name, description, parameter descriptions).
    """
    parts = [tool.name, tool.description or ""]
    if hasattr(tool, "detail_desc") and getattr(tool, "detail_desc", None):
        parts.append(str(tool.detail_desc))
    schema = tool.get_schema()
    fn = schema.get("function") or schema
    params = (fn.get("parameters") or {}).get("properties") or {}
    for k, v in params.items():
        parts.append(k)
        desc = v.get("description", "")
        if desc:
            parts.append(desc)
    return " ".join(parts).lower()


class SemanticToolRetriever:
    """
    High-performance In-Memory Semantic Tool Retriever (Tool RAG).
    Indexes ~120 tools into normalized embedding vectors at startup.
    At runtime, performs hybrid cosine similarity + dynamic lexical matching in < 5ms.
    """
    def __init__(self):
        self.tools: List[BaseTool] = []
        self.tool_docs: List[str] = []
        self.tool_lexical_texts: List[str] = []
        self.vectors: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.is_indexed: bool = False

    async def index_tools(self, tools: List[BaseTool]) -> int:
        """
        Asynchronously index candidate tools into normalized embedding vectors.
        Called during application lifespan startup or when tools are updated.
        """
        if not tools:
            self.tools = []
            self.tool_docs = []
            self.tool_lexical_texts = []
            self.vectors = np.empty((0, 384), dtype=np.float32)
            self.is_indexed = False
            return 0

        self.tools = list(tools)
        self.tool_docs = [serialize_tool(t) for t in self.tools]
        self.tool_lexical_texts = [extract_tool_keywords(t) for t in self.tools]

        logger.info(f"Indexing {len(self.tools)} tools into SemanticToolRetriever...")
        raw_embeddings = await get_embeddings(self.tool_docs)

        self.vectors = np.array(raw_embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-9)
        self.is_indexed = True

        logger.info(
            f"Successfully indexed {len(self.tools)} tools into in-memory vector index "
            f"(Shape: {self.vectors.shape}, Memory: {self.vectors.nbytes / 1024:.1f} KB)"
        )
        return len(self.tools)

    def index_tools_sync(self, tools: List[BaseTool]) -> int:
        """
        Synchronous tool indexing helper for synchronous callers (e.g. unit tests or legacy facade).
        """
        if not tools:
            self.tools = []
            self.tool_docs = []
            self.tool_lexical_texts = []
            self.vectors = np.empty((0, 384), dtype=np.float32)
            self.is_indexed = False
            return 0

        self.tools = list(tools)
        self.tool_docs = [serialize_tool(t) for t in self.tools]
        self.tool_lexical_texts = [extract_tool_keywords(t) for t in self.tools]

        target_backend = settings.EMBEDDING_BACKEND.lower()
        if target_backend == "mock":
            raw_embeddings = MockEmbeddingBackend().embed(self.tool_docs)
        else:
            raw_embeddings = FastEmbedBackend.get_instance().embed(self.tool_docs)

        self.vectors = np.array(raw_embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-9)
        self.is_indexed = True
        return len(self.tools)

    def _build_augmented_query(self, query: str, history: Optional[List[Any]] = None) -> str:
        """
        Synthesize current query with recent user and assistant turns for multi-turn context inheritance.
        e.g., '20학번은?', '전화번호 알아?' will inherit previous turn context including assistant-discovered entities.
        """
        q = query.strip()
        if not history:
            return q

        prev_turns = []
        for h in history[-4:]:
            role = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else None)
            content = getattr(h, "content", None) or (h.get("content") if isinstance(h, dict) else "")
            if content:
                # Clean and limit assistant message length to preserve entities without vector noise
                clean_content = content.replace("\n", " ").strip()
                if role == "assistant" and len(clean_content) > 250:
                    clean_content = clean_content[:250]
                prev_turns.append(clean_content)

        if prev_turns:
            return f"{' '.join(prev_turns)} {q}"
        return q

    async def retrieve(
        self,
        query: str,
        history: Optional[List[Any]] = None,
        top_k: int = 4,
        threshold: Optional[float] = None,
    ) -> List[BaseTool]:
        """
        Retrieve the top_k most semantically relevant tools for a user query.
        Combines dense embedding cosine similarity with zero-hardcoding dynamic lexical match.
        Returns empty list if no tools meet the similarity threshold (treated as general conversation).
        """
        if not self.tools or len(self.vectors) == 0:
            return []

        min_threshold = threshold if threshold is not None else settings.EMBEDDING_SIMILARITY_THRESHOLD
        augmented_query = self._build_augmented_query(query, history)

        query_embs = await get_embeddings([augmented_query])
        query_vec = np.array(query_embs[0], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        # 1. Dense Cosine Similarity via NumPy dot product
        dense_scores = np.dot(self.vectors, query_vec)

        # 2. Dynamic Lexical Match against tool metadata (Zero-Hardcoding)
        q_words = [w.strip() for w in re.findall(r"[a-zA-Z0-9가-힣]+", query.lower()) if len(w.strip()) >= 2]

        scored_tools = []
        for idx, tool in enumerate(self.tools):
            dense_sim = float(dense_scores[idx])

            doc_text = self.tool_lexical_texts[idx]
            lexical_hits = sum(1 for w in q_words if w in doc_text) if q_words else 0
            lexical_boost = 0.5 * (lexical_hits / max(len(q_words), 1)) if q_words else 0.0

            total_score = dense_sim + lexical_boost
            if total_score >= min_threshold:
                scored_tools.append((total_score, tool))

        # Sort by total score descending
        scored_tools.sort(key=lambda x: x[0], reverse=True)

        # Dynamic Relevance Cutoff: 1위 도구 점수가 높을 때(>= 0.6), 격차가 큰 저점수 도구 제외
        if scored_tools:
            top_score = scored_tools[0][0]
            cutoff = max(min_threshold, top_score * 0.55) if top_score >= 0.6 else min_threshold
            scored_tools = [item for item in scored_tools if item[0] >= cutoff]

        selected = [t for _, t in scored_tools[:top_k]]

        logger.debug(
            f"SemanticToolRetriever retrieved {len(selected)} tools "
            f"({[t.name for t in selected]}) for query: '{query[:30]}...' (Threshold: {min_threshold})"
        )
        return selected

    def retrieve_sync(
        self,
        query: str,
        history: Optional[List[Any]] = None,
        top_k: int = 4,
        threshold: Optional[float] = None,
    ) -> List[BaseTool]:
        """
        Synchronous retrieval helper for legacy or sync callers (e.g. unit tests).
        """
        if not self.tools or len(self.vectors) == 0:
            return []

        min_threshold = threshold if threshold is not None else settings.EMBEDDING_SIMILARITY_THRESHOLD
        augmented_query = self._build_augmented_query(query, history)

        target_backend = settings.EMBEDDING_BACKEND.lower()
        if target_backend == "mock":
            embs = MockEmbeddingBackend().embed([augmented_query])
        else:
            embs = FastEmbedBackend.get_instance().embed([augmented_query])

        query_vec = np.array(embs[0], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        dense_scores = np.dot(self.vectors, query_vec)
        q_words = [w.strip() for w in re.findall(r"[a-zA-Z0-9가-힣]+", query.lower()) if len(w.strip()) >= 2]

        scored_tools = []
        for idx, tool in enumerate(self.tools):
            dense_sim = float(dense_scores[idx])

            doc_text = self.tool_lexical_texts[idx]
            lexical_hits = sum(1 for w in q_words if w in doc_text) if q_words else 0
            lexical_boost = 0.5 * (lexical_hits / max(len(q_words), 1)) if q_words else 0.0

            total_score = dense_sim + lexical_boost
            if total_score >= min_threshold:
                scored_tools.append((total_score, tool))

        scored_tools.sort(key=lambda x: x[0], reverse=True)

        if scored_tools:
            top_score = scored_tools[0][0]
            cutoff = max(min_threshold, top_score * 0.55) if top_score >= 0.6 else min_threshold
            scored_tools = [item for item in scored_tools if item[0] >= cutoff]

        return [t for _, t in scored_tools[:top_k]]


# Global Singleton Retriever
tool_retriever = SemanticToolRetriever()
