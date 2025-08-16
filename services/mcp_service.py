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
                    "description": "查询历年考试的真题，返回一个真题列表。在用户提及'真题'、'往年考题'、'试卷'等关键词并指定了具体科目或主题时调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "要查询的考试科目或主题，例如：数据结构、计算机网络、C语言等。"
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
            
            # 3. 解析 LLM 的响应
            if response.tool_calls:
                # LLM 选择了工具调用，提取其内容
                tool_call = response.tool_calls[0] # 假设只返回一个工具调用
                function_name = tool_call.name
                parameters = tool_call.args
                
                logger.info(f"LLM decided to call function: {function_name} with parameters: {parameters}")
                
                # 4. 封装成 MCP 协议格式
                mcp_command = {
                    "protocol": "mcp_v1",
                    "action": "function_call",
                    "function_name": function_name,
                    "parameters": parameters
                }
                return mcp_command
            
            else:
                # LLM 没有选择工具，这是一个常规问题，返回一个特殊命令
                logger.info("LLM did not choose any tool. This is a general query.")
                return {
                    "protocol": "mcp_v1",
                    "action": "general_query",
                    "parameters": {"question": user_question}
                }

        except Exception as e:
            logger.error(f"Error during MCP command generation: {e}", exc_info=True)
            # 发生错误时，回退到常规查询，或者返回一个错误命令
            return {
                "protocol": "mcp_v1",
                "action": "error",
                "parameters": {"error": "Failed to generate command.", "details": str(e)}
            }