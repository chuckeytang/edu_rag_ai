import logging
import os
import shutil
import tiktoken
import asyncio

from llama_cloud import TextNode

from core.metadata_utils import prepare_metadata_for_storage
from typing import TYPE_CHECKING, Optional

from services.indexer_service import IndexerService
if TYPE_CHECKING:
    from services.chat_history_service import ChatHistoryService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from typing import Any, AsyncGenerator, List, Union
from llama_index.core.schema import Document as LlamaDocument

from llama_index.core import VectorStoreIndex, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

# from llama_index.cura import CURARetriever
# print(CURARetriever)
from core.config import Settings, settings
from models.schemas import ChatQueryRequest, StreamChunk
from typing import List, Dict
import chromadb

PERSIST_DIR = settings.INDEX_PATH

try:
    # 尝试加载 Qwen 模型专用的 tokenizer，如果 DashScope SDK 提供
    # 注意：LlamaIndex 的 DashScope LLM 对象可能不直接暴露 tokenizer，
    # 考虑直接使用 dashscope 库或 tiktoken 估算
    _tokenizer = tiktoken.get_encoding("cl100k_base") 
    logger.info("Using tiktoken 'cl100k_base' for token counting estimation.")
except Exception as e:
    logger.warning(f"Could not load tiktoken tokenizer 'cl100k_base', token counting may be inaccurate: {e}")
    _tokenizer = None

class QueryService:
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embedding_model: BaseEmbedding,
                 llm: LLM, 
                 indexer_service: IndexerService,
                 chat_history_service: Optional['ChatHistoryService'] = None):
        """
        构造函数 - 实现“按需加载”策略
        启动时只初始化客户端和空的索引缓存，不加载任何实际的索引数据。
        """
        self.chroma_client = chroma_client       # 使用传入的客户端
        self.embedding_model = embedding_model   # 使用传入的 embedding model
        self.llm = llm                           # 使用传入的 LLM
        
        self.indices: Dict[str, VectorStoreIndex] = {}
        self._indexer_service = indexer_service
        self.qa_prompt = self._create_qa_prompt()

        # 保存 chat_history_service 实例
        self._chat_history_service = chat_history_service
        
    def _get_or_load_index(self, collection_name: str) -> VectorStoreIndex:
        """
        重定向到 IndexerService 的按需加载方法。
        """
        return self._indexer_service._get_or_load_index(collection_name)

    def retrieve_with_filters(self, question: str, collection_name: str, filters: dict, similarity_top_k: int = 5):
        """
        [DEBUG METHOD] 仅执行带过滤的召回，并返回召回的节点列表。
        """
        logger.info(f"[DEBUG] Retrieving from collection '{collection_name}' with filters: {filters}")
        
        index = self._get_or_load_index(collection_name)
        if not index:
            raise ValueError(f"Collection '{collection_name}' does not exist or could not be loaded.")

        chroma_where_clause = self._build_chroma_where_clause(filters)

        # 1. 创建一个Retriever，配置和查询时完全一样
        retriever = index.as_retriever(
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
            similarity_top_k=similarity_top_k
        )
        
        # 2. 执行召回
        retrieved_nodes = retriever.retrieve(question)
        
        logger.info(f"[DEBUG] Retriever found {len(retrieved_nodes)} nodes matching the criteria.")
        
        # 3. 返回召回的原始节点
        return retrieved_nodes

    def _build_chroma_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        为 ChromaDB 构建原生 where 子句。
        此版本根据值的类型，自动使用 $in 或 $eq。
        它能处理复杂过滤器（如 {"field": {"$in": [val1, val2]}}）或简单过滤器（如 {"field": val}）。
        """
        if not filters:
            return {}

        chroma_filters = []

        for key, value in filters.items():
            # 情况 1: 值本身是一个字典，并且包含 ChromaDB 支持的操作符 (如 "$in")
            # 这表示调用方已经提供了复杂的过滤条件，直接使用即可。
            if isinstance(value, dict) and any(op in value for op in ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"]):
                chroma_filters.append({key: value})
            # 情况 2: 值是一个列表，我们应使用 "$in" 操作符
            elif isinstance(value, list):
                # 检查列表是否为空，避免生成 {"key": {"$in": []}} 这种可能导致问题的情况
                if value:
                    chroma_filters.append({key: {"$in": value}})
                else:
                    # 如果列表为空，则该条件不会匹配任何文档，可以忽略或根据业务逻辑处理
                    logger.warning(f"Empty list provided for filter key '{key}'. This condition will be ignored.")
            # 情况 3: 其他所有简单类型的值 (str, int, float)，使用默认的相等比较
            # ChromaDB 对于 {"key": value} 形式，默认就是相等比较，不需要显式写 "$eq"
            else:
                chroma_filters.append({key: {"$eq": value}})

        if not chroma_filters:
            return {}
        
        # 如果有多个过滤条件，用 $and 连接
        if len(chroma_filters) > 1:
            return {"$and": chroma_filters}
        
        # 只有一个条件，直接返回
        return chroma_filters[0]
            
    def _create_qa_prompt(self, prompt: str = None) -> PromptTemplate:
        if prompt:
            return PromptTemplate(prompt)
        else:
            return self.default_qa_prompt

    @property
    def title_generation_prompt(self) -> PromptTemplate:
        """
        用于从用户查询和相关文档中生成简短会话标题的提示词。
        """
        return PromptTemplate(
            "You are a title generator assistant working in an educational AI consultation system. "
            "Based on the user's initial question or message, generate a short, clear, and meaningful title that summarizes the core academic intent of the query.\n\n"
            "Requirements:\n"
            "- Title must be concise (ideally 5–12 words)\n"
            "- Use academic, study-friendly tone (not too casual)\n"
            "- Format in English Title Case (capitalize major words)\n"
            "- Do not include emojis or decorative elements\n"
            "- If the user writes in another language (e.g. Chinese), you may include short bilingual elements or mix languages naturally — but the main title should still be understandable in English\n"
            "- Focus on accuracy: represent the actual intent or topic of the user's question\n"
            "---------------------\n"
            "User query: {query_str}\n"      # LLM Query Engine will automatically fill this
            "Relevant document content: {context_str}\n" # LLM Query Engine will automatically fill this
            "Session Title:" # Guides the LLM to output only the title
        )
    
    @property
    def default_qa_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            # === 将聊天历史上下文放在最前面 ===
            "{chat_history_context}" 
            "You are an advanced academic AI assistant specializing in high school curricula (IB, A-Level, AP, IGCSE, etc.). "
            "Your task is to provide clear, elegant, and academically accurate responses based **only** on the provided context and documents.\n"
            "\n"
            "--- Your Response Guidelines ---\n"
            "1. **Accuracy & Fact-based**: Adhere strictly to the provided 'Document content'. **Never guess or invent information.**\n"
            "   - **If the information is missing, ambiguous, or if you cannot answer based *solely* on the provided documents, state clearly: 'I cannot find enough information in the provided documents to answer this question.' Do not attempt to answer if the context is insufficient.**\n"
            "2. **Concise & Elegant**: Use polished, academic language. Avoid verbosity or repetition.\n"
            "3. **Formatting**: Use section headers, bullet points, or Q&A structures for clarity. Avoid dense paragraphs.\n"
            "4. **Language**: Primarily respond in English. If the user's query includes Chinese, you may add short, clarifying Chinese phrases naturally, but keep English as the main language.\n"
            "5. **Target Audience**: Tailor your answer for motivated high school students, prioritizing clarity and depth.\n"
            "\n"
            "--- Tone ---\n"
            "Academic, supportive, focused, efficient.\n"
            "\n"
            "--- Specific Instructions ---\n"
            "If the query is an exam question (e.g., '6(b)(ii)'): first briefly describe its content, then identify relevant syllabus knowledge from the documents, and finally provide a clear, step-by-step response.\n"
            "\n"
            "Document content: {context_str}\n" # RAG 检索到的文档内容
            "Query: {query_str}" # 用户当前的问题
        )
            
    @property
    def general_chat_prompt(self) -> PromptTemplate:
        # 新增一个通用聊天提示词，不依赖任何文档内容
        return PromptTemplate(
            "{chat_history_context}"
            "You are a friendly and helpful AI assistant. Respond to the user's query naturally. If the query is a general greeting or does not require specific knowledge, respond in a conversational manner. If the user asks for academic information, but no relevant documents are found, you may state that you are an academic assistant but cannot find specific information on that topic. Always maintain a polite and supportive tone.\n"
            "\n"
            "Query: {query_str}"
        )
    
    async def rag_query_with_context(
        self,
        request: ChatQueryRequest 
    ) -> AsyncGenerator[bytes, None]:
        
        logger.info(f"Starting RAG query for session {request.session_id}, user {request.account_id} with query: '{request.question}'")
        
        # --- 语义检索历史聊天上下文 ---
        chat_history_context_string = ""
        try:
            if self._chat_history_service: 
                chat_history_context_nodes = self._chat_history_service.retrieve_chat_history_context( 
                    session_id=request.session_id,
                    account_id=request.account_id,
                    query_text=request.context_retrieval_query,
                    top_k=request.similarity_top_k or 5
                )
                if chat_history_context_nodes:
                    chat_history_context_string = "以下是与您当前问题相关的历史对话片段：\n" + \
                                                  "\n".join([f"[{node.metadata.get('role', '未知')}]: {node.text}" for node in chat_history_context_nodes]) + \
                                                  "\n---\n"
                logger.info(f"Chat history context: \n{chat_history_context_string}")
            else:
                logger.warning("ChatHistoryService instance not available in QueryService. Skipping chat history retrieval.")

        except Exception as e:
            logger.error(f"Failed to retrieve chat history context: {e}", exc_info=True)
            chat_history_context_string = ""

        rag_index = self._get_or_load_index(request.collection_name)
        if not rag_index:
            logger.error(f"RAG collection '{request.collection_name}' does not exist or could not be loaded.")
            error_chunk_json = StreamChunk(content="抱歉，知识库未准备好，请稍后再试。", is_last=True).json()
            yield f"data: {error_chunk_json}\n\n".encode("utf-8") # <--- 直接返回完整 SSE 格式的 bytes
            return

        combined_rag_filters = request.filters if request.filters else {}
        # Initialize the list to store the material_ids that were actually requested by the user
        final_referenced_material_ids = [] 

        if request.target_file_ids and len(request.target_file_ids) > 0:
            try:
                material_ids_int = [int(mid) for mid in request.target_file_ids]
                # Store the original string IDs for the metadata
                final_referenced_material_ids = request.target_file_ids # <--- Collect the string IDs here
                
                if "material_id" in combined_rag_filters and isinstance(combined_rag_filters["material_id"], dict) and "$in" in combined_rag_filters["material_id"]:
                    current_material_ids = combined_rag_filters["material_id"]["$in"]
                    # 确保不重复添加
                    combined_rag_filters["material_id"]["$in"] = list(set(current_material_ids + material_ids_int))
                else:
                    combined_rag_filters["material_id"] = {"$in": material_ids_int}
            except ValueError:
                logger.warning("Invalid material_id in target_file_ids. Ignoring file filter.")
        
        chroma_where_clause = self._build_chroma_where_clause(combined_rag_filters)
        logger.info(f"Main RAG ChromaDB `where` clause: {chroma_where_clause}")

        generated_title = ""
        if request.is_first_query: 
            try:
                # ... (标题生成逻辑保持不变)
                title_query_engine = rag_index.as_query_engine(
                    llm=self.llm,
                    vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
                    similarity_top_k=2,
                    text_qa_template=self.title_generation_prompt
                )
                title_response = await title_query_engine.aquery(request.question)
                generated_title = title_response.response.strip()
                logger.info(f"Generated session title: '{generated_title}'")
            except Exception as e:
                logger.error(f"Failed to generate session title: {e}", exc_info=True)
                generated_title = ""
        else:
            logger.info("Skipping session title generation as not requested by Java side (is_first_query is False).")

        final_qa_prompt_template = self._create_qa_prompt(request.prompt) 
        # --- 主 RAG 查询 ---
        max_retries = 3
        
        # --- 召回文档 ---
        retriever = rag_index.as_retriever(
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
            similarity_top_k=request.similarity_top_k or 5
        )
        retrieved_nodes = await retriever.aretrieve(request.question)
        num_retrieved_chunks = len(retrieved_nodes)
        logger.info(f"Retrieved {num_retrieved_chunks} chunks for query: '{request.question[:50]}...'")

        llm_response_obj = None 
        # 用于存储RAG召回的源节点信息，通用聊天模式下应为空
        final_rag_sources_info = []
        
        # 根据召回结果选择不同的 LLM 行为
        if num_retrieved_chunks == 0:
            logger.info(f"No documents retrieved for query '{request.question[:50]}...'. Switching to general chat mode.")
            
            # 直接使用 LLM 进行通用对话，不带上下文
            # 使用一个更通用的 prompt，不强制要求基于文档
            final_llm_prompt = self.general_chat_prompt.format(
                chat_history_context=chat_history_context_string,
                query_str=request.question
            )

            if _tokenizer:
                token_count = len(_tokenizer.encode(final_llm_prompt))
                logger.info(f"Total input tokens for general query (no RAG): {token_count}. Query: '{request.question[:50]}...'")

            # 直接调用 LLM，不通过 QueryEngine
            try:
                llm_response_obj = await self.llm.acomplete(final_llm_prompt) # 保存到 llm_response_obj
                full_response_content = llm_response_obj.text 
                
                if not full_response_content or full_response_content.strip() == "":
                    logger.warning(f"General chat LLM returned an empty response for query: '{request.question}'.")
                    error_chunk_json = StreamChunk(content="抱歉，我暂时无法回答您的问题，请稍后再试。", is_last=True).json()
                    yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                    return

            except Exception as e:
                logger.error(f"Error during general LLM query for '{request.question}': {e}", exc_info=True)
                error_chunk_json = StreamChunk(content="抱歉，通用对话服务出现问题，请稍后再试。", is_last=True).json()
                yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                return

        else: # 有召回文档时，走 RAG 流程
            context_str = "\n\n".join([n.get_content() for n in retrieved_nodes]) # 获得召回的文档内容
            
            # 手动渲染最终的提示词，以便计算令牌数
            final_prompt_str = final_qa_prompt_template.format(
                chat_history_context=chat_history_context_string,
                context_str=context_str,
                query_str=request.question
            )
            
            if _tokenizer:
                token_count = len(_tokenizer.encode(final_prompt_str))
                logger.info(f"Total input tokens for RAG query: {token_count}. Query: '{request.question[:50]}...'")
            else:
                logger.warning("Tokenizer not available, skipping token count for RAG query.")

            for attempt in range(max_retries):
                try:
                    query_engine = rag_index.as_query_engine(
                        llm=self.llm,
                        vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
                        similarity_top_k=request.similarity_top_k or 5,
                        streaming=True,
                        text_qa_template=final_qa_prompt_template # 注意这里是模板
                    )

                    # 执行 LLM 查询
                    response = await query_engine.aquery(request.question) # LlamaIndex 会在内部处理 prompt 填充
                    
                    # 检查响应是否为空。如果 response.response_gen 没有内容，或者 response.response 是空字符串
                    if not hasattr(response, 'response_gen') and (not response.response or response.response.strip() == ""):
                        logger.warning(f"Attempt {attempt + 1}: LLM returned an empty or nearly empty response for query: '{request.question}'. Retrying...")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt) # 指数退避
                            continue # 进入下一次重试
                        else:
                            logger.error(f"All {max_retries} attempts failed for query: '{request.question}', returning error to client.")
                            logger.error(f"Failed query input details: \n"
                                         f"Chat History: {chat_history_context_string}\n"
                                         f"Document Content: {context_str}\n"
                                         f"User Query: {request.question}\n"
                                         f"Full Prompt (first 500 chars): {final_prompt_str[:500]}...")
                            error_chunk_json = StreamChunk(content="抱歉，AI未能生成有效回复，请尝试换种方式提问。", is_last=True).json()
                            yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                            return

                    # 如果成功，则退出重试循环
                    break 

                except Exception as e:
                    logger.error(f"Attempt {attempt + 1}: Error during RAG query for '{request.question}': {e}", exc_info=True)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt) # 指数退避
                        continue
                    else:
                        logger.error(f"All {max_retries} attempts failed for query: '{request.question}', returning error to client.")
                        logger.error(f"Failed query input details: \n"
                                     f"Chat History: {chat_history_context_string}\n"
                                     f"Document Content: {context_str}\n"
                                     f"User Query: {request.question}\n"
                                     f"Full Prompt (first 500 chars): {final_prompt_str[:500]}...")
                        error_chunk_json = StreamChunk(content="抱歉，查询过程中发生错误，请稍后再试。", is_last=True).json()
                        yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                        return

        if num_retrieved_chunks > 0 and llm_response_obj and hasattr(llm_response_obj, 'source_nodes') and llm_response_obj.source_nodes:
            for node in llm_response_obj.source_nodes: 
                final_rag_sources_info.append({
                    "file_name": node.metadata.get("file_name", "未知文件"),
                    "page_number": node.metadata.get("page_label", "未知页"),
                    "material_id": node.metadata.get("material_id")
                })
        
        full_response_content = ""
        if llm_response_obj and hasattr(llm_response_obj, 'response_gen') and llm_response_obj.response_gen is not None:
            async for chunk_text in llm_response_obj.response_gen: 
                full_response_content += chunk_text
                sse_event_json = StreamChunk(content=chunk_text, is_last=False).json()
                yield f"data: {sse_event_json}\n\n".encode("utf-8") 
        elif llm_response_obj and hasattr(llm_response_obj, 'text') and llm_response_obj.text: # 使用 .text 属性来获取非流式内容
            full_response_content = llm_response_obj.text
            sse_event_json = StreamChunk(content=full_response_content, is_last=False).json()
            yield f"data: {sse_event_json}\n\n".encode("utf-8")
        else:
            logger.warning(f"No stream or direct response content from LLM for query: '{request.question}'. This should have been handled earlier.")
            error_chunk_json = StreamChunk(content="AI未能生成有效回复，请尝试换种方式提问。", is_last=True).json()
            yield f"data: {error_chunk_json}\n\n".encode("utf-8")
            return 
        
        # --- 构建最终的 metadata ---
        final_metadata = {}
        
        # 添加 rag_sources_info (详细的源节点信息)
        final_metadata["rag_sources"] = final_rag_sources_info

        # --- Add the new 'referenced_docs' list ---
        # Use the collected final_referenced_material_ids
        if final_referenced_material_ids: # 只有当有实际引用的文档时才添加
            final_metadata["referenced_docs"] = final_referenced_material_ids

        # Add title to metadata if generated
        if generated_title:
            final_metadata["session_title"] = generated_title
        
        logger.debug(f"DEBUG: Final metadata before StreamChunk creation: {final_metadata}")

        final_sse_event_json = StreamChunk(
            content="",
            is_last=True,
            metadata=final_metadata # 使用包含标题的元数据
        ).json()
        temp_stream_chunk = StreamChunk(content="", is_last=True, metadata=final_metadata)
        logger.debug(f"DEBUG: Final StreamChunk object's dict (pre-json): {temp_stream_chunk.__dict__}")

        logger.debug(f"Yielding final raw JSON chunk: {final_sse_event_json}")
        yield f"data: {final_sse_event_json}\n\n".encode("utf-8")
