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
        Supports both Native tool_calls API and Prompt-Engineered JSON ReAct Fallback.
        """
        target_model = model or settings.LLM_MODEL_NAME
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens or settings.LLM_MAX_TOKENS
        headers = self._get_headers()
        url = f"{self._base_url}/chat/completions"

        # Attempt 1: Native OpenAI tools API
        if tools:
            payload: Dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tok,
                "stream": False,
                "tools": tools,
                "tool_choice": tool_choice,
            }
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    if response.status_code == 200:
                        return self._parse_tool_response(response.json())
                    else:
                        logger.warning(
                            f"Native tools API returned status {response.status_code}. "
                            f"Falling back to Structured Prompt ReAct mode... (Response: {response.text[:120]})"
                        )
            except Exception as ex:
                logger.warning(f"Native tools request error: {ex}. Falling back to Structured Prompt ReAct mode...")

        # Attempt 2: Structured Prompt ReAct Mode (Works universally on all vLLM/OpenAI models)
        return await self._execute_prompt_based_tool_calling(
            messages=messages,
            tools=tools,
            model=target_model,
            temperature=temp,
            max_tokens=max_tok,
        )

    def _parse_tool_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
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

        # Fallback: inspect content for open-source tool calling tags or JSON
        if not parsed_tool_calls and content:
            parsed_tool_calls = self._extract_tool_calls_from_text(content)

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

    async def _execute_prompt_based_tool_calling(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Execute tool selection via structured JSON prompt orchestration.
        Guaranteed to work on all open-source vLLM backends without native tool flags.
        """
        tools_catalog_lines = []
        if tools:
            for t in tools:
                fn = t.get("function", t)
                name = fn.get("name", "")
                desc = fn.get("description", "")
                params = fn.get("parameters", {}).get("properties", {})
                param_list = ", ".join([f"{k} ({v.get('type', 'string')})" for k, v in params.items()])
                tools_catalog_lines.append(f"- 도구명: `{name}` | 설명: {desc} | 매개변수: [{param_list}]")

        catalog_text = "\n".join(tools_catalog_lines)
        prompt_instruction = f"""
[사용 가능한 도구 카탈로그]:
{catalog_text}

[응답 규칙 - 엄격한 JSON 형식]:
사용자의 질문을 해결하기 위해 도구 호출이 필요하면 반드시 아래 JSON 형식으로 응답하세요:
```json
{{
  "thought": "사용자의 질문을 해결하기 위해 이 도구를 선택한 이유와 파라미터 판단 근거",
  "tool_calls": [
    {{
      "name": "도구명",
      "arguments": {{ "매개변수명": "값" }}
    }}
  ]
}}
```

모든 정보가 충분하여 더 이상의 도구 호출이 필요 없는 경우:
```json
{{
  "thought": "필요한 모든 데이터가 수집되었으므로 최종 응답을 작성합니다.",
  "tool_calls": []
}}
```
마크다운 백틱 없이 유효한 JSON 문자열만 출력하세요.
"""

        augmented_messages: List[Dict[str, Any]] = []
        has_system = False
        for msg in messages:
            if msg.get("role") == "system":
                augmented_messages.append({
                    "role": "system",
                    "content": str(msg.get("content", "")) + "\n\n" + prompt_instruction
                })
                has_system = True
            else:
                augmented_messages.append(msg)

        if not has_system:
            augmented_messages.insert(0, {"role": "system", "content": prompt_instruction})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": augmented_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        url = f"{self._base_url}/chat/completions"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Prompt ReAct LLM Server Error ({resp.status_code}): {resp.text[:200]}")

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return {"content": "", "thought": "", "tool_calls": [], "raw_message": {}}

            msg = choices[0].get("message", {})
            raw_content = (msg.get("content") or "").strip()

            thought, tool_calls = self._parse_json_react_content(raw_content)

            return {
                "content": raw_content if not tool_calls else "",
                "thought": thought,
                "tool_calls": tool_calls,
                "raw_message": msg,
            }

    def _parse_json_react_content(self, text: str) -> tuple[str, List[Dict[str, Any]]]:
        import re
        clean_text = text.strip()
        # Remove markdown fences ```json ... ```
        if "```" in clean_text:
            clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.MULTILINE)
            clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(clean_text)
            thought = parsed.get("thought", "")
            raw_tcs = parsed.get("tool_calls", [])
            tcs = []
            if isinstance(raw_tcs, list):
                for idx, tc in enumerate(raw_tcs):
                    if isinstance(tc, dict):
                        fn_name = tc.get("name") or tc.get("tool") or ""
                        fn_args = tc.get("arguments") or tc.get("params") or {}
                        if fn_name:
                            tcs.append({
                                "id": f"call_react_{idx}_{abs(hash(fn_name)) % 10000}",
                                "type": "function",
                                "function": {
                                    "name": fn_name,
                                    "arguments": fn_args if isinstance(fn_args, dict) else {},
                                }
                            })
            return thought, tcs
        except Exception:
            # Fallback: regex search for tool_calls in messy text
            tool_calls = self._extract_tool_calls_from_text(text)
            return "", tool_calls

    def _extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        import re
        tcs = []
        # Pattern: <tool_call>{"name": "...", ...}</tool_call>
        tc_matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        for idx, tc_text in enumerate(tc_matches):
            try:
                tc_json = json.loads(tc_text.strip())
                fn_name = tc_json.get("name") or tc_json.get("function", {}).get("name", "")
                fn_args = tc_json.get("arguments") or tc_json.get("function", {}).get("arguments", {})
                if fn_name:
                    tcs.append({
                        "id": f"call_text_{idx}_{abs(hash(fn_name)) % 10000}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": fn_args if isinstance(fn_args, dict) else {},
                        }
                    })
            except Exception:
                pass
        return tcs


# Global Singleton LLM Client
llm_client = LLMClient()
