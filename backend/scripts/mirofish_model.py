"""
MiroFishAzureModel — CAMEL-compatible model backend using LLMClient (Responses API).

Wraps our LLMClient so OASIS/CAMEL can use gpt-5.4-pro (Responses API only)
by converting messages → LLMClient.chat() → fake ChatCompletion response.
"""

import asyncio
import sys
import os
import time
import logging
from typing import Any, Dict, List, Optional, Union

# Ensure backend package is on path
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

from camel.models.base_model import BaseModelBackend
from camel.models.openai_model import OpenAIModel
from camel.token_counter import OpenAITokenCounter
from camel.types import ModelPlatformType, ModelType
from camel.messages import OpenAIMessage

from app.utils.llm_client import LLMClient

logger = logging.getLogger('mirofish.camel_model')


def _make_chat_completion(content: str, model: str) -> ChatCompletion:
    """Wrap a text response into a ChatCompletion object that CAMEL/OASIS expects."""
    return ChatCompletion(
        id="mirofish-" + str(int(time.time() * 1000)),
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                ),
            )
        ],
        created=int(time.time()),
        model=model,
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=len(content) // 4,
            prompt_tokens=100,
            total_tokens=100 + len(content) // 4,
        ),
    )


class MiroFishAzureModel(BaseModelBackend):
    """
    CAMEL model backend that delegates to MiroFish LLMClient.
    Supports gpt-5.4-pro and other Azure models requiring Responses API.
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter=None,
    ):
        # Store model name before super().__init__ which may validate it
        self._model_name = str(model_type) if not isinstance(model_type, str) else model_type
        if hasattr(model_type, 'value'):
            self._model_name = model_type.value

        # Create our LLMClient wrapper (handles Azure / Responses API automatically)
        self._llm = LLMClient(
            api_key=api_key,
            base_url=url,
            model=self._model_name,
        )

        # Minimal super init — pass required positional args
        super().__init__(
            model_type=model_type,
            model_config_dict=model_config_dict or {},
            api_key=api_key or self._llm.api_key,
            url=url or self._llm.base_url,
            token_counter=token_counter,
        )

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    def check_model_config(self):
        """No extra validation needed."""
        pass

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format=None,
        tools=None,
    ) -> ChatCompletion:
        """Synchronous call: convert messages → LLMClient.chat() → ChatCompletion."""
        msg_list = [{"role": m["role"], "content": m.get("content") or ""} for m in messages]
        try:
            content = self._llm.chat(messages=msg_list, temperature=0.7, max_tokens=4096)
        except Exception as e:
            logger.error(f"MiroFishAzureModel._run error: {e}")
            content = "DO_NOTHING"
        return _make_chat_completion(content, self._model_name)

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format=None,
        tools=None,
    ) -> ChatCompletion:
        """Async call: run sync LLMClient in executor to avoid blocking event loop."""
        msg_list = [{"role": m["role"], "content": m.get("content") or ""} for m in messages]
        loop = asyncio.get_event_loop()
        try:
            content = await loop.run_in_executor(
                None,
                lambda: self._llm.chat(messages=msg_list, temperature=0.7, max_tokens=4096),
            )
        except Exception as e:
            logger.error(f"MiroFishAzureModel._arun error: {e}")
            content = "DO_NOTHING"
        return _make_chat_completion(content, self._model_name)

    @property
    def token_counter(self):
        if not self._token_counter:
            # Use gpt-4o as proxy for token counting (similar tokenizer)
            try:
                self._token_counter = OpenAITokenCounter(ModelType.GPT_4O)
            except Exception:
                self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

    @property
    def token_limit(self) -> int:
        return 128000

    @property
    def stream(self) -> bool:
        return False
