"""
Remote OpenAPI Connector for inu-portal-server
Dynamically syncs endpoints from Spring Boot /v3/api-docs and transforms them into callable AI Tools.
"""
from typing import Dict, Any, List, Optional
import re
import httpx

from app.core.config import settings
from app.core.logging import logger
from app.tools.base import BaseTool


class OpenApiTool(BaseTool):
    """
    A dynamic tool instance backed by a remote OpenAPI REST endpoint.
    """
    def __init__(
        self,
        name: str,
        description: str,
        method: str,
        path: str,
        parameters_schema: Dict[str, Any],
        category: str = "GENERAL",
        requires_auth: bool = False,
    ):
        self.name = name
        self.description = description
        self.method = method.upper()
        self.path = path
        self.parameters_schema = parameters_schema
        self.category = category
        self.requires_auth = requires_auth

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Execute HTTP request against inu-portal-server with Token Relay.
        """
        base_url = settings.INU_PORTAL_SERVER_URL.rstrip("/")
        url_path = self.path

        # 1. Path parameter substitution (e.g. /api/timetables/{timeTableId})
        path_vars = re.findall(r"\{([a-zA-Z0-9_]+)\}", self.path)
        query_params = {}
        for k, v in arguments.items():
            if k in path_vars:
                url_path = url_path.replace(f"{{{k}}}", str(v))
            else:
                query_params[k] = v

        full_url = f"{base_url}{url_path}"

        # 2. Token Relay: propagate incoming Auth & Authorization headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "inu-agent-core/0.1.0",
        }
        raw_token = context.get("auth") or context.get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""
        if clean_token:
            headers["Auth"] = clean_token
            headers["Authorization"] = f"Bearer {clean_token}"

        if settings.INU_INTERNAL_S2S_SECRET:
            headers["X-Internal-Secret"] = settings.INU_INTERNAL_S2S_SECRET

        logger.info(f"Executing OpenApiTool [{self.name}]: {self.method} {full_url} (Auth: {bool(clean_token)})")

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                if self.method == "GET":
                    response = await client.get(full_url, params=query_params, headers=headers)
                elif self.method == "POST":
                    response = await client.post(full_url, json=query_params, headers=headers)
                else:
                    response = await client.request(self.method, full_url, params=query_params, headers=headers)

                if response.status_code >= 400:
                    logger.warning(f"OpenApiTool [{self.name}] failed with status {response.status_code}: {response.text}")
                    return {
                        "error": f"API request failed with status {response.status_code}",
                        "status_code": response.status_code,
                    }

                data = response.json()
                # If wrapped in standard ResponseDto(data=..., message=..., status=...)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]

                # Extract list if inside ListResponseDto (e.g. {"pages": 1, "total": 2, "contents": [...]})
                list_payload = None
                if isinstance(data, list):
                    list_payload = data
                elif isinstance(data, dict):
                    if "contents" in data and isinstance(data["contents"], list):
                        list_payload = data["contents"]
                    elif "items" in data and isinstance(data["items"], list):
                        list_payload = data["items"]

                # Guardrail: truncate large arrays to prevent context window exhaustion
                if list_payload is not None:
                    if len(list_payload) > 6:
                        return {
                            "total_count": len(list_payload),
                            "items": list_payload[:6],
                            "contents": list_payload[:6],
                            "notice": f"결과가 많아 상위 6개 항목만 표시합니다. (전체 {len(list_payload)}건)",
                        }
                    elif isinstance(data, dict) and "contents" in data:
                        data["items"] = list_payload

                return data
            except Exception as e:
                logger.error(f"Failed to execute OpenApiTool [{self.name}]: {e}", exc_info=True)
                return {"error": f"통신 오류가 발생했습니다: {str(e)}"}


class OpenApiConnector:
    """
    Parser and manager that pulls /v3/api-docs from inu-portal-server
    and registers tools dynamically.
    """
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.INU_PORTAL_SERVER_URL).rstrip("/")

    async def fetch_spec(self) -> Optional[Dict[str, Any]]:
        """Fetch OpenAPI 3.0 spec from remote /v3/api-docs"""
        url = f"{self.base_url}/v3/api-docs"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e:
            logger.warning(f"Could not reach {url} to fetch OpenAPI spec: {e}")
        return None

    def parse_spec(self, spec: Dict[str, Any]) -> List[OpenApiTool]:
        """Convert OpenAPI 3.0 paths into OpenApiTool objects"""
        tools: List[OpenApiTool] = []
        paths = spec.get("paths", {})

        for path, path_item in paths.items():
            for method, op in path_item.items():
                if method.lower() not in ["get"]:
                    # Guardrail: Only safe GET methods for automatic agent exposure
                    continue

                op_id = op.get("operationId") or self._generate_op_id(method, path)
                summary = (op.get("summary") or op.get("description") or f"API for {path}").strip()
                detail_desc = (op.get("description") or "").strip()

                tags = op.get("tags", [])
                category = self._infer_category(path, tags)

                # Skip internal or deprecated APIs
                if any(x in path for x in ["/admin/", "/actuator/", "/error", "/agent/"]):
                    continue

                tool_name = self._sanitize_tool_name(op_id, path)
                params_schema = self._build_params_schema(op.get("parameters", []))
                requires_auth = self._check_auth_required(op)

                tool = OpenApiTool(
                    name=tool_name,
                    description=summary,
                    method=method,
                    path=path,
                    parameters_schema=params_schema,
                    category=category,
                    requires_auth=requires_auth,
                )
                tool.detail_desc = detail_desc
                tools.append(tool)

        logger.info(f"Parsed {len(tools)} OpenApiTools from OpenAPI specification")
        return tools

    def _generate_op_id(self, method: str, path: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]", "_", path).strip("_")
        return f"{method}_{cleaned}"

    def _sanitize_tool_name(self, op_id: str, path: str) -> str:
        # Create clear tool name e.g. api_cafeterias_getMenu
        name = re.sub(r"[^a-zA-Z0-9_]", "_", op_id)
        if not name.startswith("api_"):
            name = f"api_{name}"
        return name[:64]

    def _infer_category(self, path: str, tags: List[str]) -> str:
        p = path.lower()
        if "/cafeteria" in p:
            return "CAFETERIA"
        if "/bus" in p or "buses" in p or "shuttle" in p:
            return "BUS"
        if "/timetable" in p or "syllabus" in p:
            return "TIMETABLE"
        if "/notice" in p or "councilnotice" in p:
            return "NOTICE"
        if "/schedule" in p or "/calendar" in p or "/semester" in p:
            return "SCHEDULE"
        if "/reservation" in p:
            return "RESERVATION"
        if "/weather" in p:
            return "WEATHER"
        if "/directory" in p or "contact" in p or "/department" in p:
            return "DIRECTORY"
        return "INTIP"

    def _check_auth_required(self, op: Dict[str, Any]) -> bool:
        security = op.get("security", [])
        return len(security) > 0

    def _build_params_schema(self, parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
        properties = {}
        required = []

        for p in parameters:
            p_name = p.get("name")
            if not p_name or p_name in ["member"]:
                continue

            p_schema = p.get("schema", {})
            p_type = p_schema.get("type", "string")
            p_desc = p.get("description") or f"Parameter {p_name}"

            prop_def: Dict[str, Any] = {
                "type": p_type,
                "description": p_desc,
            }
            if "enum" in p_schema:
                prop_def["enum"] = p_schema["enum"]
            if "default" in p_schema:
                prop_def["default"] = p_schema["default"]

            properties[p_name] = prop_def
            if p.get("required"):
                required.append(p_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
