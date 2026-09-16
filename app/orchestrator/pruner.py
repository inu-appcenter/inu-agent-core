"""
[DEPRECATED] Legacy Tool Pruning Module
This module is maintained exclusively for backward compatibility with existing tests/callers.
All static keyword dictionaries (CATEGORY_KEYWORDS) and magic-number scoring rules have been eliminated.
Core routing is now handled by the In-Memory Semantic Tool Retriever (app.orchestrator.retriever.tool_retriever).
"""
from typing import List, Dict, Any, Optional
import warnings

from app.tools.base import BaseTool
from app.core.logging import logger
from app.orchestrator.retriever import tool_retriever, SemanticToolRetriever


class ToolPruner:
    """
    Deprecated ToolPruner facade.
    Forwards requests directly to the zero-hardcoding SemanticToolRetriever.
    """
    @classmethod
    def prune(
        cls,
        query: str,
        tools: List[BaseTool],
        history: Optional[List[Any]] = None,
        client: str = "INTIP",
        max_tools: int = 4,
    ) -> List[BaseTool]:
        """
        Backward-compatible facade delegating to SemanticToolRetriever.
        """
        if not tools:
            return []

        # If global tool_retriever has been initialized with the exact toolset, use it directly
        if tool_retriever.is_indexed and tool_retriever.tools == tools:
            return tool_retriever.retrieve_sync(
                query=query,
                history=history,
                top_k=max_tools,
            )

        # For ad-hoc tool lists (e.g. isolated unit tests), index dynamically
        temp_retriever = SemanticToolRetriever()
        temp_retriever.index_tools_sync(tools)
        return temp_retriever.retrieve_sync(
            query=query,
            history=history,
            top_k=max_tools,
        )
