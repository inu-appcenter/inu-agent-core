"""
Central Tool Registry for inu-agent-core
Manages both OpenAPI-derived tools and custom extension tools.
"""
from typing import Dict, List, Optional
from pathlib import Path
import json

from app.core.logging import logger
from app.tools.base import BaseTool
from app.tools.openapi import OpenApiConnector, OpenApiTool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (Category: {tool.category})")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    async def sync_inu_portal_tools(self) -> int:
        """
        Synchronize tools from inu-portal-server /v3/api-docs.
        If the server is unavailable, load fallback sample schema.
        """
        connector = OpenApiConnector()
        spec = await connector.fetch_spec()

        if not spec:
            logger.warning("inu-portal-server is offline or unreachable. Loading fallback sample schema...")
            sample_path = Path(__file__).parent / "schemas" / "inu_portal_sample.json"
            if sample_path.exists():
                with open(sample_path, "r", encoding="utf-8") as f:
                    spec = json.load(f)

        if not spec:
            logger.error("No OpenAPI spec available to sync.")
            return 0

        parsed_tools = connector.parse_spec(spec)
        for tool in parsed_tools:
            self.register(tool)

        logger.info(f"Successfully synced {len(parsed_tools)} tools from INU Portal OpenAPI spec")
        return len(parsed_tools)


tool_registry = ToolRegistry()
