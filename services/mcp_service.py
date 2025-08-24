# services/mcp_service.py
import logging
import json
from typing import Dict, Any, List, Union
from llama_index.llms.openai_like import OpenAILike
from core.config import settings
from llama_index.core.llms import ChatMessage, MessageRole

from models.schemas import MCPResponse
from services.retrieval_service import RetrievalService
from llama_index.core.schema import NodeWithScore


logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self, llm_for_function_calling: OpenAILike, retrieval_service: RetrievalService):
        self.llm = llm_for_function_calling
        self.retrieval_service = retrieval_service
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "query_exam_questions",
                    "description": "这是一个用于查询**外部数据库**中历年考试真题的工具。当用户请求**获取**、**查找**或**查询**指定科目的真题或试卷时，必须使用此工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "要查询的考试科目或主题，例如：金融、生物、医学、计算机科学等。",
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

    async def generate_mcp_command(self, user_question: str) -> MCPResponse:
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
                
                function_name = tool_call.function.name
                arguments_str = tool_call.function.arguments
                
                try:
                    parameters = json.loads(arguments_str)
                    topic = parameters.get("topic")
                except json.JSONDecodeError:
                    logger.error("Failed to parse tool call arguments as JSON.")
                    parameters = {}
                    topic = None

                # LLM通常不会在function_call响应中同时返回文本，所以我们手动生成
                # 或者从你的LLM返回中尝试获取。如果无法获取，就用一个模板。
                # 假设你的LLM偶尔会返回llm_response_text
                llm_response_text = response.message.content
                if not llm_response_text and topic:
                    # 如果LLM没有返回文字，我们手动生成一段
                    llm_response_text = f"正在为您查找关于“{topic}”的真题，请稍候。"
                elif not llm_response_text:
                    llm_response_text = "正在为您查询相关信息，请稍候。"

                logger.info(f"LLM decided to call function: {function_name} with parameters: {parameters}")
                
                # --- 命中 Function Calling 后，调用 QueryService 进行召回 ---
                if function_name == "query_exam_questions" and topic:
                    logger.info(f"Function call 'query_exam_questions' detected. Performing RAG recall for topic: {topic}")
                    
                    try:
                        # 直接调用 RetrievalService 的通用召回方法
                        retrieved_nodes: List[NodeWithScore] = await self.retrieval_service.retrieve_documents(
                            query_text=user_question, 
                            collection_name="public_collection", 
                            filters={"type": "PaperCut"}, 
                            top_k=10,
                            use_reranker=False # 此处不需要重排器
                        )
                        
                        # 从返回的节点列表中提取 material_id
                        retrieved_ids = [int(node.node.metadata.get("paper_cut_id")) 
                                         for node in retrieved_nodes 
                                         if node.node.metadata.get("paper_cut_id") is not None]
                        
                        logger.info(f"RAG recall for '{topic}' returned {len(retrieved_ids)} PaperCut IDs: {retrieved_ids}")
                        
                        parameters["llm_response_text"] = llm_response_text
                        parameters["paper_cut_ids"] = retrieved_ids
                        return MCPResponse(
                            action="function_call",
                            function_name=function_name,
                            parameters=parameters
                        )

                    except Exception as e:
                        logger.error(f"Error during RAG recall for function calling: {e}", exc_info=True)
                        # 如果召回失败，回退到通用查询并返回错误信息
                        return MCPResponse(
                            action="general_query",
                            parameters={"llm_response_text": "抱歉，检索真题时发生错误，请稍后再试。"}
                        )

                # 如果是其他 Function Calling 或者 topic 为空，依然返回常规的 function_call 响应
                logger.info(f"LLM decided to call function: {function_name} with parameters: {parameters}")
                parameters["llm_response_text"] = llm_response_text
                return MCPResponse(
                    action="function_call",
                    function_name=function_name,
                    parameters=parameters
                )
            
            else:
                logger.info("LLM did not choose any tool. This is a general query.")
                return MCPResponse(
                    action="general_query",
                    function_name=None,
                    parameters={"original_question": user_question, "llm_response_text": response.message.content}
                )

        except Exception as e:
            logger.error(f"Error during MCP command generation: {e}", exc_info=True)
            # 发生错误时，回退到常规查询
            return MCPResponse(
                action="general_query",
                function_name=None,
                parameters={
                    "original_question": user_question,
                    "llm_response_text": "抱歉，无法理解您的请求。"
                }
            )