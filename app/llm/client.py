"""
LLM Client integration module for LiteLLM / vLLM (OpenAI-compatible) via HTTPX
Bypasses Cloudflare / SDK-level header blocking and handles streaming natively.
"""
from typing import AsyncGenerator, Dict, Any, List, Optional
import json
import httpx

from app.core.config import settings
from app.core.logging import logger


class LLMClient:
    def __init__(self):
        self._base_url = settings.LLM_BASE_URL.rstrip("/")
        self._api_key = settings.LLM_API_KEY
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=float(settings.LLM_TIMEOUT_SECONDS),
            write=10.0,
            pool=10.0,
        )
        logger.info(
            f"Initialized LLMClient (HTTPX) pointing to {self._base_url} (Model: {settings.LLM_MODEL_NAME})"
        )

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "inu-agent-core/0.1.0",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM tokens using standard OpenAI-compatible SSE over HTTPX.
        """
        target_model = model or settings.LLM_MODEL_NAME
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens or settings.LLM_MAX_TOKENS

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "stream": True,
        }

        url = f"{self._base_url}/chat/completions"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        err_body = await response.aread()
                        logger.error(
                            f"LLM request failed with status {response.status_code}: {err_body.decode('utf-8', errors='ignore')}"
                        )
                        raise RuntimeError(f"LLM Server Error ({response.status_code}): {err_body.decode('utf-8', errors='ignore')[:200]}")

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        if line == "data: [DONE]":
                            break

                        try:
                            chunk_data = json.loads(line[6:])
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error during LLM streaming: {e}", exc_info=True)
                raise e

    async def create_guided_completion(
        self,
        messages: List[Dict[str, Any]],
        response_schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate structured JSON strictly conforming to response_schema using vLLM/LiteLLM.
        """
        target_model = model or settings.LLM_MODEL_NAME
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "response_format": {
                "type": "json_object",
                "schema": response_schema,
            },
        }

        url = f"{self._base_url}/chat/completions"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise RuntimeError(f"LLM Server Error ({response.status_code}): {response.text[:200]}")

                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "{}")
                return "{}"
            except Exception as e:
                logger.error(f"Error during guided JSON completion: {e}", exc_info=True)
                raise e


    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute OpenAI / vLLM tool calling completion.
        Returns dict with keys:
            - "content": Optional text response / reasoning
            - "thought": Optional reasoning thought
            - "tool_calls": List[Dict] with id, function: {name, arguments: Dict}
            - "raw_message": Full message dict from LLM
        """
        target_model = model or settings.LLM_MODEL_NAME
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens or settings.LLM_MAX_TOKENS

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        url = f"{self._base_url}/chat/completions"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise RuntimeError(f"LLM Server Error ({response.status_code}): {response.text[:200]}")

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return {"content": "", "thought": "", "tool_calls": [], "raw_message": {}}

                msg = choices[0].get("message", {})
                content = msg.get("content") or ""
                reasoning_content = msg.get("reasoning_content") or ""

                raw_tool_calls = msg.get("tool_calls") or []
                parsed_tool_calls = []
                for idx, tc in enumerate(raw_tool_calls):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    parsed_args = {}
                    if isinstance(raw_args, dict):
                        parsed_args = raw_args
                    elif isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except Exception:
                            parsed_args = {"query": raw_args}

                    parsed_tool_calls.append({
                        "id": tc.get("id") or f"call_{idx}_{abs(hash(fn_name)) % 10000}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": parsed_args,
                        }
                    })

                # Fallback: If no structured tool_calls returned, inspect content for open-source tool calling tags or JSON
                if not parsed_tool_calls and content:
                    import re
                    # Pattern 1: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
                    tc_matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
                    for idx, tc_text in enumerate(tc_matches):
                        try:
                            tc_json = json.loads(tc_text.strip())
                            fn_name = tc_json.get("name") or tc_json.get("function", {}).get("name", "")
                            fn_args = tc_json.get("arguments") or tc_json.get("function", {}).get("arguments", {})
                            if fn_name:
                                parsed_tool_calls.append({
                                    "id": f"call_text_{idx}_{abs(hash(fn_name)) % 10000}",
                                    "type": "function",
                                    "function": {
                                        "name": fn_name,
                                        "arguments": fn_args if isinstance(fn_args, dict) else {},
                                    }
                                })
                        except Exception:
                            pass

                thought = reasoning_content
                if not thought and content and "<thought>" in content:
                    import re
                    m = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
                    if m:
                        thought = m.group(1).strip()
                        content = content.replace(m.group(0), "").strip()

                return {
                    "content": content,
                    "thought": thought,
                    "tool_calls": parsed_tool_calls,
                    "raw_message": msg,
                }
            except Exception as e:
                logger.error(f"Error during LLM tool completion: {e}", exc_info=True)
                raise e


# Global Singleton LLM Client
llm_client = LLMClient()
