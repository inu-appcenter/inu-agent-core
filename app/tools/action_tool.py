"""
Client Action Tool implementation
Wraps Action Rules into LLM-callable tools that dispatch P2P ClientActionInstructions to mobile apps.
"""
from typing import Dict, Any
from uuid import uuid4

from app.tools.base import BaseTool
from app.rules.models import ActionRule
from app.llm.schemas import ClientActionInstruction, ClientActionRequest


class ClientActionTool(BaseTool):
    """
    A tool that emits a ClientActionInstruction for on-device scraping (Coocon model).
    """
    def __init__(self, rule: ActionRule):
        self.rule = rule
        self.name = f"action_{rule.action_id.lower()}"
        self.description = f"[{rule.domain}] {rule.title} - {rule.description}"
        self.category = rule.domain

    def get_schema(self) -> Dict[str, Any]:
        """
        Produce function calling schema for LLM
        """
        # Dynamic parameter properties based on rule target params
        properties = {}
        for k, v in self.rule.target.params.items():
            properties[k] = {
                "type": "string" if isinstance(v, str) else "number",
                "description": f"Parameter: {k}",
            }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [],
                },
            },
        }

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> ClientActionInstruction:
        """
        Produce instruction packet for the mobile app Action Runner
        """
        unique_action_id = f"act_{self.rule.action_id.lower()}_{uuid4().hex[:8]}"

        # Merge arguments with default params
        final_params = dict(self.rule.target.params)
        final_params.update(arguments)

        return ClientActionInstruction(
            action_id=unique_action_id,
            auth_domain=self.rule.domain,
            protocol=self.rule.protocol,
            request=ClientActionRequest(
                url=self.rule.target.url,
                method=self.rule.target.method,
                headers=self.rule.target.headers if self.rule.target.headers else None,
                params=final_params if final_params else None,
                body=self.rule.target.body_template,
            ),
        )
