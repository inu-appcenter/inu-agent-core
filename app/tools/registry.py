"""
Central Tool Registry for inu-agent-core
Manages both OpenAPI-derived tools and Client Action tools (Coocon model).
"""
from typing import Dict, List, Optional
from pathlib import Path
import json

from app.core.logging import logger
from app.tools.base import BaseTool
from app.tools.openapi import OpenApiConnector, OpenApiTool
from app.tools.action_tool import ClientActionTool
from app.rules.registry import action_rule_registry

from app.tools.inuchat import InuAiKnowledgeTool
from app.tools.library import LibrarySeatTool
from app.tools.campus_watch import CampusWatchTool

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (Category: {tool.category})")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        cat_upper = category.upper()
        return [t for t in self._tools.values() if t.category.upper() == cat_upper]

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def register_client_action_rules(self) -> int:
        """
        Register external action rules (LMS, Library, Portal) as callable LLM tools.
        """
        rules = action_rule_registry.list_rules()
        registered_count = 0
        for rule in rules:
            # Library real-time seats are handled directly by server tool LibrarySeatTool.
            # Only personal LMS and PORTAL ERP rules use on-device client actions.
            if rule.domain == "LIBRARY":
                continue
            action_tool = ClientActionTool(rule)
            self.register(action_tool)
            registered_count += 1
        logger.info(f"Registered {registered_count} Client Action tools (LMS, Portal).")
        return registered_count

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

    async def initialize_all_tools(self) -> None:
        """Initialize internal OpenAPI tools, INUChat RAG knowledge tool, Library real-time tool, Campus Watch tool, and external Client Action tools."""
        # 1. Register InuChat Knowledge Tool
        self.register(InuAiKnowledgeTool())
        # 2. Register Library Real-time Public Seat Tool
        self.register(LibrarySeatTool())
        # 3. Register Campus Watch / Seat Sniper Tool
        self.register(CampusWatchTool())
        # 4. Register Client Action Tools
        self.register_client_action_rules()
        # 5. Register OpenAPI Tools
        await self.sync_inu_portal_tools()


tool_registry = ToolRegistry()
