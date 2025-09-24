import logging
import json
from typing import Dict, Any, List, Union
from llama_index.llms.openai_like import OpenAILike
from core.config import settings
from llama_index.core.llms import ChatMessage, MessageRole

from models.schemas import MCPResponse
from services.volcano_rag_service import VolcanoEngineRagService

logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self, llm_for_function_calling: OpenAILike, volcano_rag_service: VolcanoEngineRagService):
        self.llm = llm_for_function_calling
        # 依赖注入新的火山引擎服务
        self.volcano_rag_service = volcano_rag_service
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
        
        messages = [
            ChatMessage(
                role=MessageRole.USER, 
                content=user_question,
            )
        ]
        
        try:
            response = await self.llm.achat(messages, tools=self.get_tool_for_llm(), tool_choice="auto")
            tool_calls = response.message.additional_kwargs.get("tool_calls", None)

            if tool_calls:
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

                llm_response_text = response.message.content
                if not llm_response_text and topic:
                    llm_response_text = f"正在为您查找关于“{topic}”的真题，请稍候。"
                elif not llm_response_text:
                    llm_response_text = "正在为您查询相关信息，请稍候。"

                logger.info(f"LLM decided to call function: {function_name} with parameters: {parameters}")
                
                if function_name == "query_exam_questions" and topic:
                    logger.info(f"Function call 'query_exam_questions' detected. Performing RAG recall for topic: {topic}")
                    
                    try:
                        # 调用新的 VolcanoEngineRagService
                        retrieved_chunks: List[Dict[str, Any]] = await self.volcano_rag_service.retrieve_documents(
                            query_text=user_question, 
                            knowledge_base_id="paper_cut_collection", # PaperCut 对应的知识库ID
                            limit=10,
                            rerank_switch=True # 火山引擎的重排通常是开启的
                        )
                        
                        # 从返回的 chunks 中提取 paper_cut_id
                        retrieved_ids = [
                            int(json.loads(chunk.get("doc_info", {}).get("user_data", "{}")).get("paper_cut_id"))
                            for chunk in retrieved_chunks
                            if json.loads(chunk.get("doc_info", {}).get("user_data", "{}")).get("paper_cut_id") is not None
                        ]

                        # 去重
                        retrieved_ids = list(set(retrieved_ids))
                        
                        logger.info(f"Volcano RAG recall for '{topic}' returned {len(retrieved_ids)} PaperCut IDs: {retrieved_ids}")
                        
                        parameters["llm_response_text"] = llm_response_text
                        parameters["paper_cut_ids"] = retrieved_ids
                        return MCPResponse(
                            action="function_call",
                            function_name=function_name,
                            parameters=parameters
                        )

                    except Exception as e:
                        logger.error(f"Error during RAG recall for function calling: {e}", exc_info=True)
                        return MCPResponse(
                            action="general_query",
                            parameters={"llm_response_text": "抱歉，检索真题时发生错误，请稍后再试。"}
                        )

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
            return MCPResponse(
                action="general_query",
                function_name=None,
                parameters={
                    "original_question": user_question,
                    "llm_response_text": "抱歉，无法理解您的请求。"
                }
            )