# services/llm_service.py

from http import HTTPStatus
import json
import logging
from typing import Any, AsyncGenerator, Sequence
from llama_index.llms.dashscope import DashScope as LlamaDashScope
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import ChatResponse, ChatResponseGen
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.llms.dashscope.utils import (
    chat_message_to_dashscope_messages,
    dashscope_response_to_chat_response
)
logger = logging.getLogger(__name__)

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
        
        # ... (参数提取和准备代码不变)
        tools = parameters.pop("tools", None)
        tool_choice = parameters.pop("tool_choice", "auto")

        if tools:
            parameters["tools"] = tools
            parameters["tool_choice"] = tool_choice

        parameters.pop("stream", None)
        parameters.pop("incremental_output", None)
        parameters["result_format"] = "message"

        dashscope_response = await acall_with_messages(
            model=self.model_name,
            messages=chat_message_to_dashscope_messages(messages),
            api_key=self.api_key,
            parameters=parameters,
        )
        
        if dashscope_response.status_code != HTTPStatus.OK:
            logger.error(f"DashScope API error: {dashscope_response.code} - {dashscope_response.message}")
            return ChatResponse(message=ChatMessage(), raw=dashscope_response)

        # 核心：从原始响应中提取信息
        top_choice = dashscope_response.output.choices[0]
        choice_message = top_choice.get("message", {})
        
        content = choice_message.get("content", "")
        role = choice_message.get("role")
        tool_calls_raw = choice_message.get("tool_calls", [])
        
        additional_kwargs = {}
        
        # 1. 提取并封装 tool_calls
        if tool_calls_raw:
            # 直接构建符合 OpenAI/LlamaIndex 期望的字典列表
            llama_tool_calls = []
            for tc in tool_calls_raw:
                func = tc.get("function", {})
                
                # 尝试安全地解析 arguments 字符串为字典
                args_str = func.get("arguments", "{}")
                # LlamaIndex 期望 additional_kwargs["tool_calls"] 中的 arguments 是一个字符串！
                # 它是要被传递给底层 LLM 客户端的 OpenAI 兼容结构，args 字段应是字符串。
                
                # 构造符合 OpenAI Message ToolCall 规范的字典
                tool_call_dict = {
                    "id": tc.get("id", f"call_{func.get('name')}_{tc.get('index', 0)}"), # 确保有 id
                    "type": "function",
                    "function": {
                        "name": func.get("name"),
                        "arguments": args_str # 保持为 JSON 字符串
                    }
                }
                llama_tool_calls.append(tool_call_dict)

            if llama_tool_calls:
                 additional_kwargs["tool_calls"] = llama_tool_calls
                 logger.info(f"Successfully injected {len(llama_tool_calls)} tool calls into additional_kwargs.")

        # 2. 构建最终的 ChatResponse
        return ChatResponse(
            message=ChatMessage(
                role=role, 
                content=content,
                additional_kwargs=additional_kwargs # 注入工具调用
            ), 
            raw=dashscope_response
        )

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