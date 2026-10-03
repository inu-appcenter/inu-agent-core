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
from app.tools.notification_tools import (
    ManageReminderTool,
    DailyBriefTool,
    NoticeKeywordTool,
    MySettingsTool,
)
from app.tools.mcp_connector import McpConnector, McpRemoteTool

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (Category: {tool.category})")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        if name in self._tools:
            return self._tools[name]
        # Case-insensitive and prefix-tolerant alias matching
        clean_name = name.replace("api_", "").replace("action_", "").replace("-", "_").lower()
        # 1. Exact match on stripped tool name
        for t_name, t in self._tools.items():
            t_clean = t_name.replace("api_", "").replace("action_", "").replace("-", "_").lower()
            if t_clean == clean_name:
                return t

        # 2. Canonical domain alias mappings (prioritize primary search/query tools)
        if "contact" in clean_name or "directory" in clean_name:
            for preferred in ["api_searchDirectory", "api_getDirectory", "searchDirectory", "getDirectory", "directory"]:
                if preferred in self._tools:
                    return self._tools[preferred]
            for t_name, t in self._tools.items():
                if "directory" in t_name.lower():
                    return t
        elif "search" in clean_name:
            for preferred in ["api_unifiedSearch", "api_search", "unifiedSearch", "search"]:
                if preferred in self._tools:
                    return self._tools[preferred]
            for t_name, t in self._tools.items():
                if "search" in t_name.lower():
                    return t
        elif "club" in clean_name:
            for preferred in ["api_getAllClubs", "getAllClubs", "api_getClubs", "api_club", "getClubs"]:
                if preferred in self._tools:
                    return self._tools[preferred]
            for t_name, t in self._tools.items():
                if "club" in t_name.lower():
                    return t
        elif "lost" in clean_name:
            for preferred in ["api_getLostProperties", "api_getList_1", "getLostProperties", "api_lostProperty"]:
                if preferred in self._tools:
                    return self._tools[preferred]
            for t_name, t in self._tools.items():
                if "lost" in t_name.lower():
                    return t
        elif "syllabus" in clean_name:
            for preferred in ["api_getSyllabus", "getSyllabus"]:
                if preferred in self._tools:
                    return self._tools[preferred]
            for t_name, t in self._tools.items():
                if "syllabus" in t_name.lower():
                    return t
        elif "tuition" in clean_name:
            for preferred in ["action_portal_get_tuition", "PORTAL_GET_TUITION"]:
                if preferred in self._tools:
                    return self._tools[preferred]
        elif "scholarship" in clean_name:
            for preferred in ["action_portal_get_scholarship", "PORTAL_GET_SCHOLARSHIP"]:
                if preferred in self._tools:
                    return self._tools[preferred]
        elif "assignment" in clean_name:
            for preferred in ["action_lms_get_upcoming_assignments", "LMS_GET_UPCOMING_ASSIGNMENTS"]:
                if preferred in self._tools:
                    return self._tools[preferred]
        elif "reminder" in clean_name:
            for preferred in ["action_manage_reminder", "manage_reminder"]:
                if preferred in self._tools:
                    return self._tools[preferred]
        elif "setting" in clean_name:
            for preferred in ["action_my_settings", "my_settings", "action_daily_brief"]:
                if preferred in self._tools:
                    return self._tools[preferred]
        elif "council" in clean_name:
            for preferred in ["api_getCouncilNotices", "api_getAllPost_3", "api_getCouncilNotice_1"]:
                if preferred in self._tools:
                    return self._tools[preferred]
        elif "department" in clean_name and "notice" in clean_name:
            for preferred in ["api_getDepartmentNotices", "getDepartmentNotices"]:
                if preferred in self._tools:
                    return self._tools[preferred]

        # 3. Fallback to category match only if no name matched
        for t_name, t in self._tools.items():
            if t.category.lower() == clean_name:
                return t
        return None

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
        Synchronize tools from inu-portal-server.
        Tries remote MCP (Model Context Protocol /mcp) first, and falls back to OpenAPI /v3/api-docs if unreachable.
        """
        # 1. Try modern MCP Server first
        try:
            mcp_connector = McpConnector()
            mcp_tools = await mcp_connector.fetch_tools()
            if mcp_tools:
                for tool in mcp_tools:
                    self.register(tool)
                logger.info(f"Successfully mounted {len(mcp_tools)} tools from INU Portal MCP server")
                return len(mcp_tools)
        except Exception as e:
            logger.warning(f"Failed to sync tools via MCP: {e}. Falling back to OpenAPI...")

        # 2. Fallback to OpenAPI Connector
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
        # 4. Register Notification & Settings Tools
        self.register(ManageReminderTool())
        self.register(DailyBriefTool())
        self.register(NoticeKeywordTool())
        self.register(MySettingsTool())
        # 5. Register Client Action Tools
        self.register_client_action_rules()
        # 6. Register OpenAPI Tools
        await self.sync_inu_portal_tools()


tool_registry = ToolRegistry()
