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


# Global Singleton LLM Client
llm_client = LLMClient()
