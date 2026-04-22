"""
LLM Client Wrapper
Unified OpenAI format API calls
Supports Ollama num_ctx parameter to prevent prompt truncation
Supports OpenAI Responses API for newer models (gpt-5.x etc.)
"""

import json
import os
import re
import time
import logging
from typing import Optional, Dict, Any, List
from openai import OpenAI, AzureOpenAI, BadRequestError

from ..config import Config

logger = logging.getLogger('mirofish.llm_client')

_LLM_LOG_PATH = '/app/logs/llm_requests.jsonl'


def _append_llm_log(record: dict) -> None:
    """Append a record to the centralized LLM request log."""
    try:
        os.makedirs(os.path.dirname(_LLM_LOG_PATH), exist_ok=True)
        with open(_LLM_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception:
        pass


class LLMClient:
    """LLM Client"""

    # Global cache: model_name -> True/False/None (Responses API detection)
    # Shared across all instances so the first BadRequestError fallback is not repeated.
    _responses_api_cache: Dict[str, bool] = {}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 1800.0
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY not configured")

        azure_version = Config.AZURE_API_VERSION
        if azure_version:
            self.client = AzureOpenAI(
                azure_endpoint=self.base_url,
                api_key=self.api_key,
                api_version=azure_version,
                timeout=timeout,
            )
            self._azure = True
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
            )
            self._azure = False

        # Ollama context window size — only used when talking to Ollama.
        self._num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))

        # Detect Responses API requirement:
        # 1. Check global cache first (shared across all LLMClient instances)
        # 2. If unknown, auto-detect from model name (gpt-5.x → Responses API)
        # 3. Otherwise will be detected on first call via BadRequestError fallback
        if self.model in LLMClient._responses_api_cache:
            self._use_responses_api: Optional[bool] = LLMClient._responses_api_cache[self.model]
        elif re.match(r'gpt-5', self.model):
            # gpt-5.x models only support Responses API — skip chat.completions entirely
            self._use_responses_api = True
            LLMClient._responses_api_cache[self.model] = True
            logger.info(f"Model {self.model} detected as Responses API only (gpt-5.x)")
        else:
            self._use_responses_api = None

    def _is_ollama(self) -> bool:
        """Check if we're talking to an Ollama server."""
        return not self._azure and '11434' in (self.base_url or '')

    # ------------------------------------------------------------------
    # Responses API helpers (gpt-5.x and other newer Azure models)
    # ------------------------------------------------------------------

    def _chat_via_responses(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Call the OpenAI Responses API, converting messages format."""
        # Extract system prompt (goes into `instructions`)
        instructions = None
        user_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                instructions = content
            elif role in ("user", "assistant"):
                user_parts.append(content)

        # Use the full conversation as input (last user message)
        input_text = user_parts[-1] if user_parts else ""

        # Responses API requires the word "json" in input when json_object format is used
        if json_mode and "json" not in input_text.lower():
            input_text += "\n\nRespond with valid JSON only."

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_text,
            "max_output_tokens": max_tokens,
        }
        if instructions:
            kwargs["instructions"] = instructions
        # Note: json_object format is intentionally NOT set for Responses API.
        # Reasoning models (gpt-5.x) generate valid JSON naturally from the prompt;
        # forcing json_object mode can cause truncation and inference failures.

        response = self.client.responses.create(**kwargs)
        return response.output_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send chat request.

        Automatically falls back to the Responses API when the model does not
        support Chat Completions (e.g. gpt-5.x on Azure).

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count
            response_format: Response format (e.g., JSON mode) — ignored for Responses API

        Returns:
            Model response text
        """
        t0 = time.time()
        api_used = "unknown"
        error_msg = None
        content = ""

        try:
            if self._use_responses_api:
                api_used = "responses"
                content = self._chat_via_responses(
                    messages, temperature, max_tokens,
                    json_mode=(response_format or {}).get("type") == "json_object",
                )
                return content

            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if response_format:
                kwargs["response_format"] = response_format

            # For Ollama: pass num_ctx via extra_body to prevent prompt truncation
            if self._is_ollama() and self._num_ctx:
                kwargs["extra_body"] = {
                    "options": {"num_ctx": self._num_ctx}
                }

            try:
                api_used = "chat_completions"
                response = self.client.chat.completions.create(**kwargs)
            except BadRequestError:
                # Model doesn't support Chat Completions — switch to Responses API permanently
                self._use_responses_api = True
                LLMClient._responses_api_cache[self.model] = True
                logger.info(f"Model {self.model} switched to Responses API (BadRequestError fallback)")
                api_used = "responses (fallback)"
                content = self._chat_via_responses(
                    messages, temperature, max_tokens,
                    json_mode=(response_format or {}).get("type") == "json_object",
                )
                return content

            self._use_responses_api = False
            LLMClient._responses_api_cache[self.model] = False
            content = response.choices[0].message.content
            # Some models (like MiniMax M2.5) include <think>thinking content in response, need to remove
            content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
            return content

        except Exception as e:
            error_msg = str(e)
            raise

        finally:
            elapsed = round(time.time() - t0, 2)
            # Build a compact prompt snippet for the log
            prompt_snippet = ""
            if messages:
                last_msg = messages[-1].get("content", "")
                prompt_snippet = last_msg[:300]

            _append_llm_log({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": self.model,
                "api": api_used,
                "elapsed_s": elapsed,
                "prompt_snippet": prompt_snippet,
                "response_snippet": content[:500] if content else "",
                "error": error_msg,
            })

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send chat request and return JSON.

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        # Clean markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Fallback: extract the first {...} block from the response
            # (handles cases where the model adds preamble/postamble text)
            match = re.search(r'\{[\s\S]*\}', cleaned_response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Invalid JSON format from LLM: {cleaned_response}")
