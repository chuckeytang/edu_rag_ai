# services/mcp_service.py
import logging
import json
from typing import Dict, Any, Union
from llama_index.llms.openai_like import OpenAILike
from core.config import settings
from llama_index.core.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self, llm_for_function_calling: OpenAILike):
        self.llm = llm_for_function_calling
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "query_exam_questions",
                    "description": "这是一个用于查询**外部数据库**中历年考试真题的工具。当用户请求**获取**、**查找**或**查询**指定科目或主题（例如'数据结构'、'计算机网络'）的真题或试卷时，必须使用此工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "要查询的考试科目或主题，例如：数据结构、计算机网络、C语言等。",
                            }
                        },
                        "required": ["topic"]
                    }
                }
            }
            # 这里可以添加更多工具定义，例如：
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "query_syllabus_info",
            #         "description": "查询特定科目的教学大纲信息。",
            #         "parameters": {...}
            #     }
            # }
        ]
        logger.info("MCPService initialized with tool definitions.")

    def get_tool_for_llm(self):
        """返回LLM需要的tools格式"""
        return self.tool_definitions

    async def generate_mcp_command(self, user_question: str) -> Dict[str, Any]:
        """
        根据用户问题，调用LLM生成一个MCP协议命令。
        """
        logger.info(f"Attempting to generate MCP command for question: '{user_question}'")
        
        # 1. 创建 ChatMessage 列表，将工具定义作为系统消息的一部分
        messages = [
            ChatMessage(
                role=MessageRole.USER, 
                content=user_question,
            )
        ]
        
        try:
            # 2. 调用 LLM 并启用 function_calling
            response = await self.llm.achat(messages, tools=self.get_tool_for_llm(), tool_choice="auto")
            tool_calls = response.message.additional_kwargs.get("tool_calls", None)

            # 3. 解析 LLM 的响应
            if tool_calls:
                # LLM 选择了工具调用，提取其内容
                # 这里的 tool_call 是一个 `ChatCompletionMessageToolCall` 对象
                tool_call = tool_calls[0]
                
                # 核心修改：使用 . 访问属性
                function_name = tool_call.function.name
                arguments_str = tool_call.function.arguments
                
                try:
                    parameters = json.loads(arguments_str)
                except json.JSONDecodeError:
                    logger.error("Failed to parse tool call arguments as JSON.")
                    parameters = {}

                logger.info(f"LLM decided to call function: {function_name} with parameters: {parameters}")
                
                # 5. 封装成 MCP 协议格式
                mcp_command = {
                    "protocol": "mcp_v1",
                    "action": "function_call",
                    "function_name": function_name,
                    "parameters": parameters
                }
                return mcp_command
            
            else:
                # LLM 没有选择工具，返回一个特殊的 'general_query' 命令
                logger.info("LLM did not choose any tool. This is a general query.")
                return {
                    "protocol": "mcp_v1",
                    "action": "general_query",
                    "parameters": {"original_question": user_question, "llm_response_text": response.message.content}
                }

        except Exception as e:
            logger.error(f"Error during MCP command generation: {e}", exc_info=True)
            # 发生错误时，回退到常规查询，或者返回一个错误命令
            return {
                "protocol": "mcp_v1",
                "action": "general_query",
                "parameters": {
                    "original_question": user_question,
                    "llm_response_text": "抱歉，无法理解您的请求。"
                }
            }