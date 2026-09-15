"""
LLM Client integration module for LiteLLM / vLLM (OpenAI-compatible)
"""
from typing import AsyncGenerator, Dict, Any, List, Optional
from openai import AsyncOpenAI
import httpx

from app.core.config import settings
from app.core.logging import logger


class LLMClient:
    def __init__(self):
        # Configure httpx client with generous timeouts for LLM streaming
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=settings.LLM_TIMEOUT_SECONDS,
                write=10.0,
                pool=10.0,
            ),
            follow_redirects=True,
        )

        # Standard OpenAI client pointing to the remote LiteLLM/vLLM domain
        self._client = AsyncOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            http_client=http_client,
            max_retries=2,
        )
        logger.info(
            f"Initialized LLMClient pointing to LiteLLM/vLLM at {settings.LLM_BASE_URL} (Model: {settings.LLM_MODEL_NAME})"
        )

    @property
    def raw_client(self) -> AsyncOpenAI:
        return self._client

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM tokens using standard OpenAI-compatible SSE.
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

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
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
        Generate structured JSON strictly conforming to response_schema using vLLM/LiteLLM Guided Decoding.
        """
        target_model = model or settings.LLM_MODEL_NAME
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE

        try:
            response = await self._client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temp,
                response_format={
                    "type": "json_object",
                    "schema": response_schema,
                },
            )
            content = response.choices[0].message.content or "{}"
            return content
        except Exception as e:
            logger.error(f"Error during guided JSON completion: {e}", exc_info=True)
            raise e


# Global Singleton LLM Client
llm_client = LLMClient()
