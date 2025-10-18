import logging
import json
from typing import Dict, Any, List, Union
from llama_index.llms.openai_like import OpenAILike
from core.config import settings
from llama_index.core.llms import ChatMessage, MessageRole

from models.schemas import MCPResponse
from services.abstract_kb_service import AbstractKnowledgeBaseService

logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self, llm_for_function_calling: OpenAILike, kb_service: AbstractKnowledgeBaseService): # 更改：注入抽象接口
        self.llm = llm_for_function_calling
        # 依赖注入抽象知识库服务
        self.kb_service = kb_service 
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
        logger.info("MCPService initialized with tool definitions and Abstract KB Service.")

    def get_tool_for_llm(self):
        """返回LLM需要的tools格式"""
        return self.tool_definitions

    # 辅助函数：安全地将复杂对象转换为可序列化的字典
    def safe_to_dict(self, obj: Any, max_depth: int = 15, current_depth: int = 0) -> Any:
        """递归地将对象的关键属性转换为字典，以避免循环引用和不可序列化错误。"""
        if current_depth >= max_depth:
            return f"<<Max Depth Reached: {type(obj).__name__}>>"
        
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        
        if isinstance(obj, (list, tuple)):
            return [self.safe_to_dict(item, max_depth, current_depth + 1) for item in obj]
            
        if isinstance(obj, dict):
            return {k: self.safe_to_dict(v, max_depth, current_depth + 1) for k, v in obj.items()}

        # 针对 LlamaIndex 或其他 LLM 响应对象
        result = {}
        
        # 尝试使用内置方法 to_dict() 或 dict()
        if hasattr(obj, 'to_dict'):
            try:
                return self.safe_to_dict(obj.to_dict(), max_depth, current_depth + 1)
            except Exception:
                pass # 忽略错误
                
        if hasattr(obj, 'dict'): # Pydantic v1
            try:
                return self.safe_to_dict(obj.dict(), max_depth, current_depth + 1)
            except Exception:
                pass # 忽略错误
                
        # 遍历对象的公共属性
        for attr_name in dir(obj):
            # 忽略私有属性、魔法方法和方法
            if attr_name.startswith('_') or callable(getattr(obj, attr_name)):
                continue
            
            try:
                attr_value = getattr(obj, attr_name)
                # 仅处理常见类型和已知的重要对象
                if isinstance(attr_value, (str, int, float, bool, dict, list, tuple, type(None))) or hasattr(attr_value, 'to_dict') or hasattr(attr_value, 'dict') or hasattr(attr_value, 'additional_kwargs'):
                    result[attr_name] = self.safe_to_dict(attr_value, max_depth, current_depth + 1)
                else:
                    result[attr_name] = f"<{type(attr_value).__name__} object>"
            except Exception:
                # 捕获访问属性时的任何错误
                result[attr_name] = f"<<Error accessing {attr_name}>>"

        if result:
            # 确保类型信息也被保留
            return {"__type__": type(obj).__name__, **result}
        
        return f"<{type(obj).__name__} instance>"


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
            logger.info("--- LLM Achat Response Deep Inspection START ---")
            
            # 将复杂的响应对象安全地转换为字典
            # response_dict = self.safe_to_dict(response, max_depth=15) 
            
            # # 打印完整的结构
            # logger.info(f"Full LLM Response Structure (JSON): \n{json.dumps(response_dict, indent=2, ensure_ascii=False)}")
            # logger.info("--- LLM Achat Response Deep Inspection END ---")

            tool_calls = response.message.additional_kwargs.get("tool_calls", None)

            if tool_calls:
                tool_call = tool_calls[0]
                
                function_name = tool_call.get("function", {}).get("name")
                arguments_str = tool_call.get("function", {}).get("arguments")
                
                if not function_name or not arguments_str:
                    logger.error("Tool call structure is invalid or missing function name/arguments.")
                    raise ValueError("Invalid tool call structure received from LLM.")
                
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
                        # 调用抽象接口进行检索
                        retrieved_chunks: List[Dict[str, Any]] = await self.kb_service.retrieve_documents(
                            query_text=user_question, 
                            knowledge_base_id=settings.BAILIAN_PAPERCUT_INDEX_ID or "qip7hsynj3", # 知识库ID从配置中读取，或者使用默认值
                            limit=10,
                            rerank_switch=True 
                        )
                        
                        # --- 以下是业务侧的后处理逻辑，适配火山引擎的返回值结构 ---
                        
                        retrieved_ids = []
                        for chunk in retrieved_chunks:
                            # 这里的逻辑是针对 Volcano Engine 的特定返回结构（用户数据在 user_data 字段）
                            user_data = chunk.get("user_data", {})
                            paper_cut_id_raw = user_data.get("paper_cut_id")
                            
                            if paper_cut_id_raw is not None:
                                try:
                                    # 尝试转换成 int，兼容原始逻辑
                                    retrieved_ids.append(int(paper_cut_id_raw))
                                except ValueError:
                                    logger.warning(f"Could not convert paper_cut_id '{paper_cut_id_raw}' to int.")

                        # 去重
                        retrieved_ids = list(set(retrieved_ids))
                        
                        logger.info(f"KB RAG recall for '{topic}' returned {len(retrieved_ids)} PaperCut IDs: {retrieved_ids}")
                        
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
                    # parameters={"original_question": user_question, "llm_response_text": response.message.content}
                    parameters={"original_question": user_question, "llm_response_text": ""}
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