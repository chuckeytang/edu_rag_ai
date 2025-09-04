# services/llm_service.py

from http import HTTPStatus
from typing import Any, AsyncGenerator, Sequence
from llama_index.llms.dashscope import DashScope as LlamaDashScope
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import ChatResponse, ChatResponseGen
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.llms.dashscope.utils import (
    chat_message_to_dashscope_messages,
    dashscope_response_to_chat_response
)

# 确保你可以从这里导入 acall_with_messages 和 astream_call_with_messages
# 你需要从 LlamaIndex 源码的 llama_index/llms/dashscope/base.py 中复制这两个函数
from llama_index.llms.dashscope.base import acall_with_messages, astream_call_with_messages

class DashScopeWithTools(LlamaDashScope):
    """
    一个增强版的 DashScope LLM，支持 Function Calling。
    继承自 LlamaIndex 的 DashScope 类，并重写了 achat 和 astream_chat 方法。
    """

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        parameters = self._get_default_parameters()
        parameters.update(kwargs)
        
        # 核心改造：从 kwargs 中取出 tools 和 tool_choice，并从 parameters 中移除
        tools = parameters.pop("tools", None)
        tool_choice = parameters.pop("tool_choice", "auto")

        # 将 tools 和 tool_choice 添加到底层调用参数中
        if tools:
            parameters["tools"] = tools
            parameters["tool_choice"] = tool_choice

        # 移除流式参数，确保调用的是非流式 API
        parameters.pop("stream", None)
        parameters.pop("incremental_output", None)
        parameters["result_format"] = "message"

        response = await acall_with_messages(
            model=self.model_name,
            messages=chat_message_to_dashscope_messages(messages),
            api_key=self.api_key,
            parameters=parameters,
        )
        return dashscope_response_to_chat_response(response)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        parameters = self._get_default_parameters()
        parameters.update(kwargs)
        
        # 核心改造：从 kwargs 中取出 tools 和 tool_choice
        tools = parameters.pop("tools", None)
        tool_choice = parameters.pop("tool_choice", "auto")
        
        parameters["incremental_output"] = True
        parameters["result_format"] = "message"
        parameters["stream"] = True

        # 将 tools 和 tool_choice 添加到底层调用参数中
        if tools:
            parameters["tools"] = tools
            parameters["tool_choice"] = tool_choice

        dashscope_messages = chat_message_to_dashscope_messages(messages)

        async_responses = astream_call_with_messages(
            model=self.model_name,
            messages=dashscope_messages,
            api_key=self.api_key,
            parameters=parameters,
        )

        async def gen() -> AsyncGenerator[ChatResponse, None]:
            content = ""
            async for response in async_responses:
                if response.status_code == HTTPStatus.OK:
                    top_choice = response.output.choices[0]
                    role = top_choice["message"]["role"]
                    incremental_output = top_choice["message"].get("content", "")
                    content += incremental_output

                    yield ChatResponse(
                        message=ChatMessage(role=role, content=content),
                        delta=incremental_output,
                        raw=response,
                    )
                else:
                    yield ChatResponse(message=ChatMessage(), raw=response)
                    return

        return gen()