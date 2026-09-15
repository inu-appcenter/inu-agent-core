"""
Base interfaces and definitions for inu-agent-core Tools
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel


class ToolParameter(BaseModel):
    type: str
    description: str
    required: bool = False
    enum: Optional[list] = None


class BaseTool(ABC):
    """
    Abstract base class for all inu-agent-core Tools
    """
    name: str
    description: str
    category: str = "GENERAL"  # "INTIP", "LMS", "LIBRARY", "PORTAL", "GENERAL"

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return standard OpenAI Function Calling tool definition
        """
        pass

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Execute the tool action or return ClientActionInstruction
        """
        pass
