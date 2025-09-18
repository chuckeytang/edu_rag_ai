import json
import logging
import asyncio

from typing import Optional, Any, AsyncGenerator, List, Dict
from memory_profiler import profile

# 新增的导入，用于替代 LlamaIndex 的概念
from services.volcano_rag_service import VolcanoEngineRagService
from services.indexer_service import IndexerService
from services.chat_history_service import ChatHistoryService

# 引入一个可以替代 LlamaIndex 节点的简单数据结构，或者直接使用字典
from dataclasses import dataclass
@dataclass
class SimpleNode:
    text: str
    metadata: Dict[str, Any]

from core.rag_config import RagConfig
from core.config import settings
from models.schemas import ChatQueryRequest, StreamChunk
from tools.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self, 
                 # 移除 chroma_client, embedding_model 和 retrieval_service
                 llm: Any,  # LLM实例仍然需要
                 indexer_service: IndexerService,
                 volcano_rag_service: VolcanoEngineRagService, # 新增：注入火山引擎服务
                 rag_config: RagConfig, 
                 chat_history_service: Optional[ChatHistoryService] = None):
        """
        构造函数 - 切换到火山引擎 RAG 服务
        """
        self.llm = llm                           
        self._indexer_service = indexer_service
        self.volcano_rag_service = volcano_rag_service # 新增
        self._chat_history_service = chat_history_service
        self.rag_config = rag_config

        # 使用配置中的提示词
        # 这里可以使用字符串格式化代替 PromptTemplate，以减少对 LlamaIndex 的依赖
        self.qa_prompt_template = rag_config.qa_prompt_template
        self.title_generation_prompt_template = rag_config.title_prompt_template
        self.general_chat_prompt_template = rag_config.general_chat_prompt_template
        
        # 移除本地重排器，因为火山引擎已处理
        self.llm_reranker = None
        self.local_reranker = None
        logger.info("QueryService initialized. Using Volcano Engine RAG for retrieval and reranking.")
        
    @profile
    async def rag_query_with_context(
        self,
        request: ChatQueryRequest,
        rag_config: RagConfig,
    ) -> AsyncGenerator[bytes, None]:
        
        logger.info(f"Starting RAG query for session {request.session_id}, user {request.account_id} with query: '{request.question}'")
        self.rag_config = rag_config
        
        # --- 语义检索历史聊天上下文 ---
        chat_history_context_string = ""
        chat_history_chunk_texts = []
        chat_history_chunk_ids = []
        try:
            if self._chat_history_service: 
                chat_history_context_nodes = await self._chat_history_service.retrieve_chat_history_context( 
                    session_id=request.session_id,
                    account_id=request.account_id,
                    query_text=request.context_retrieval_query,
                    top_k=self.rag_config.history_retrieval_top_k or 5
                )
                if chat_history_context_nodes:
                    chat_history_context_string = "The following are excerpts from a historical conversation relevant to your current problem:\n" + \
                                                  "\n".join([f"[{node.metadata.get('role', '未知')}]: {node.text}" for node in chat_history_context_nodes]) + \
                                                  "\n---\n"
                    chat_history_chunk_texts = [node.text for node in chat_history_context_nodes]
                    chat_history_chunk_ids = [node.id_ for node in chat_history_context_nodes]
                     
                logger.info(f"Chat history context: \n{chat_history_context_string}")
            else:
                logger.warning("ChatHistoryService instance not available in QueryService. Skipping chat history retrieval.")

        except Exception as e:
            logger.error(f"Failed to retrieve chat history context: {e}", exc_info=True)
            chat_history_context_string = ""
        
        combined_rag_filters = request.filters if request.filters else {}
        query_type = combined_rag_filters.pop("type", None)
        final_referenced_material_ids = []
        generated_title = "" 

        # --- 流程1: 统一调用火山引擎的检索服务 ---
        knowledge_base_ids_to_query = []
        if query_type == "PaperCut":
            # 这里你需要为 "PaperCut" 定义一个对应的火山引擎知识库ID
            knowledge_base_ids_to_query = [settings.VOLCANO_ENGINE_KNOWLEDGE_BASE_ID] # 示例
            logger.info("Query type is 'PaperCut', retrieving from specific knowledge base.")
        elif query_type:
            # 你可能需要将你的 `collection_name` 映射到火山引擎的知识库ID
            # 这里简化为直接使用配置中的ID
            knowledge_base_ids_to_query = [settings.VOLCANO_ENGINE_KNOWLEDGE_BASE_ID] 
            logger.info(f"Query type is '{query_type}', retrieving from knowledge base.")
        else:
            knowledge_base_ids_to_query = [settings.VOLCANO_ENGINE_KNOWLEDGE_BASE_ID] 
            logger.info("Query type not specified, retrieving from default knowledge base.")
        
        # 使用 request.similarity_top_k 作为检索数量，火山引擎内部会处理重排
        final_top_k = request.similarity_top_k if request.similarity_top_k is not None else rag_config.retrieval_top_k
        
        # --- 流程2: 异步并行调用火山引擎检索API ---
        retrieve_tasks = []
        for kb_id in knowledge_base_ids_to_query:
            task = self.volcano_rag_service.retrieve_documents(
                query_text=request.question,
                knowledge_base_id=kb_id,
                limit=final_top_k,
                rerank_switch=True, # 始终开启重排
            )
            retrieve_tasks.append(task)
        
        # 使用 asyncio.gather 并行执行所有召回任务
        results_from_collections = await asyncio.gather(*retrieve_tasks)

        # --- 流程3: 合并结果集 ---
        combined_retrieved_chunks = []
        for result_list in results_from_collections:
            combined_retrieved_chunks.extend(result_list)
        
        logger.info(f"Retrieved {len(combined_retrieved_chunks)} chunks from all knowledge bases combined.")

        # 检查是否召回了文档
        if not combined_retrieved_chunks:
            logger.info("No documents retrieved for the main query. Directly calling LLM.")
            async for chunk in self._stream_llm_without_rag_context(
                request.question,
                chat_history_context_string,
                generated_title
            ):
                yield chunk
            return
        
        # --- 流程4: 准备传递给 LLM 的上下文 ---
        # 火山引擎返回的已经是最终重排过的结果，不再需要本地过滤和重排
        final_retrieved_chunks_for_llm = combined_retrieved_chunks
        
        # 计算并限制发送给 LLM 的总 token 数量
        tokenizer = get_tokenizer()
        if not tokenizer:
            logger.warning("Tokenizer not available, cannot perform token-based content trimming.")
        
        llm_max_context_tokens = rag_config.llm_max_context_tokens
        
        static_prompt_tokens = 0
        if tokenizer:
            try:
                template_tokens = len(tokenizer.encode(self.qa_prompt_template))
                history_tokens = len(tokenizer.encode(chat_history_context_string))
                query_tokens = len(tokenizer.encode(request.question))
                placeholder_tokens = len(tokenizer.encode('{chat_history_context}{context_str}{query_str}'))
                
                static_prompt_tokens = template_tokens + history_tokens + query_tokens - placeholder_tokens
                
            except Exception as e:
                logger.warning(f"Failed to calculate static prompt tokens: {e}. Falling back to approximation.")
                static_prompt_tokens = 500 # 近似值
        
        reserved_tokens = 500
        remaining_context_tokens = llm_max_context_tokens - static_prompt_tokens - reserved_tokens
        
        logger.info(f"LLM max context: {llm_max_context_tokens}, static prompt tokens: {static_prompt_tokens}, remaining for RAG content: {remaining_context_tokens}")
        
        context_parts = []
        current_context_tokens = 0
        rag_sources_info = []
        
        # 使用火山引擎返回的 chunk 数据构建上下文
        for chunk in final_retrieved_chunks_for_llm:
            chunk_content = chunk.get('content', '')
            chunk_source = chunk.get('source', '未知文件')
            chunk_doc_id = chunk.get('docId', 'N/A')
            chunk_material_id = chunk.get('material_id', 'N/A')
            
            if tokenizer:
                chunk_tokens = len(tokenizer.encode(chunk_content))
                if current_context_tokens + chunk_tokens > remaining_context_tokens:
                    logger.warning("Exceeding LLM context window. Truncating RAG content.")
                    break
                current_context_tokens += chunk_tokens

            context_parts.append(
                f"--- Document Content (Source: {chunk_source}, DocID: {chunk_doc_id}) ---\n"
                f"{chunk_content}\n"
                f"--------------------------------------------------------------------------------"
            )

            rag_sources_info.append({
                "file_name": chunk_source,
                "page_number": "N/A", # 火山引擎API通常不直接返回页码，可能需要从元数据解析
                "material_id": chunk_material_id
            })

        context_str_for_llm = "\n\n".join(context_parts)
        
        max_retries = 3
        # --- 主 RAG 查询 LLM ---
        llm_response_obj = None 
        for attempt in range(max_retries):
            try:
                final_prompt_for_llm = self.qa_prompt_template.format(
                    chat_history_context=chat_history_context_string,
                    context_str=context_str_for_llm, 
                    query_str=request.question
                )
                logger.info(f"final_prompt_for_llm {final_prompt_for_llm}.")

                tokenizer = get_tokenizer()
                if tokenizer:
                    token_count = len(tokenizer.encode(final_prompt_for_llm))
                    logger.info(f"Attempt {attempt + 1}: Total input tokens for LLM query: {token_count}. Query: '{request.question[:50]}...'")
                else:
                    logger.warning("Tokenizer not available, skipping token count for LLM query.")

                # 使用 LLM 的直接调用，而不是 LlamaIndex 的 QueryEngine
                llm_response_gen = await self.llm.astream_chat(messages=[ 
                    {"role": "system", "content": final_prompt_for_llm},
                    {"role": "user", "content": request.question}
                ])

                llm_response_obj = llm_response_gen
                logger.info("Finish query.")

                is_response_empty = False
                if llm_response_obj is None:
                    is_response_empty = True
                else:
                    # 检查LLM响应是否为空，这需要根据你使用的LLM客户端进行调整
                    pass # 这里的逻辑可能需要根据具体的LLM客户端进行调整

                if is_response_empty:
                    logger.warning(f"Attempt {attempt + 1}: LLM returned an empty or nearly empty response for query: '{request.question}'. Retrying...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt) 
                        continue 
                    else:
                        logger.error(f"All {max_retries} attempts failed for query: '{request.question}', returning error to client.")
                        error_chunk_json = StreamChunk(content="Sorry, AI did not generate a valid response, please try a different way to ask questions.", is_last=True).json()
                        yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                        return

                break 

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error during LLM query for '{request.question}': {e}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt) 
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed for query: '{request.question}', returning error to client.")
                    error_chunk_json = StreamChunk(content="Sorry, an error occurred during the query. Please try again later.", is_last=True).json()
                    yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                    return
        
        # === 核心响应流式处理和引用生成逻辑 ===
        full_response_content = ""
        # 用于引用生成的 chunk 列表
        referenced_chunks = [chunk for chunk in final_retrieved_chunks_for_llm if chunk.get('content')]
        referenced_chunk_texts = [c['content'] for c in referenced_chunks]
        
        # 因为火山引擎的 chunk 没有唯一的ID，我们使用其 docId 和 chunkId 组合来作为唯一ID
        referenced_chunk_ids = [f"{c.get('docId')}-{c.get('chunkId')}" for c in referenced_chunks]
        
        # 将聊天历史的文本和ID也加入引用池
        combined_referenced_texts = referenced_chunk_texts + chat_history_chunk_texts
        combined_referenced_ids = referenced_chunk_ids + chat_history_chunk_ids

        # 接收并流式返回 LLM 响应
        async for chunk_resp in llm_response_obj:
            chunk_text = chunk_resp.delta.content
            if chunk_text:
                full_response_content += chunk_text
                sse_event_json = StreamChunk(content=chunk_text, is_last=False).json()
                yield f"data: {sse_event_json}\n\n".encode("utf-8") 
            
        logger.debug("Successfully streamed response chunks.")

        if not full_response_content or full_response_content.strip() == "NO_RELEVANT_DOCUMENTS_FOUND":
             # 处理LLM找不到答案的情况
             logger.info("LLM did not find a relevant response based on context.")
             final_sse_event_json = StreamChunk(content="抱歉，我未能从现有资料中找到相关信息。", is_last=True).json()
             yield f"data: {final_sse_event_json}\n\n".encode("utf-8")
             return
    
        # --- 句子级引用生成 ---
        response_sentences = []
        sentence_citations = []
        if full_response_content:
            import re
            response_sentences = re.split(r'(?<=[.!?。！？])\s+', full_response_content)

        # ⚠️ 这里需要一个嵌入模型来生成句子嵌入，但你的 __init__ 中移除了它
        # 你需要在 QueryService 中注入一个单独的嵌入模型，或者用一个外部服务
        # 否则，这部分逻辑无法运行
        if False: # 暂时禁用这部分代码
            # ... (如果重新添加嵌入模型，可以恢复这部分逻辑)
            # 这是一个占位符，提示需要嵌入模型
            pass
        else:
            logger.warning("Embedding model is not available for sentence-level citation. Skipping this step.")

        # --- 构建最终的 metadata ---
        final_metadata = {}
        
        # 构建来源信息
        unique_rag_sources = {}
        for chunk in final_retrieved_chunks_for_llm:
            material_id = chunk.get("material_id")
            if material_id not in unique_rag_sources:
                unique_rag_sources[material_id] = {
                    "file_name": chunk.get("source", "未知文件"),
                    "page_number": chunk.get("page_label", "N/A"),
                    "material_id": material_id
                }
        
        final_metadata["rag_sources"] = list(unique_rag_sources.values())

        if final_referenced_material_ids: 
            final_metadata["referenced_docs"] = final_referenced_material_ids

        if generated_title:
            final_metadata["session_title"] = generated_title
        
        if sentence_citations:
            final_metadata["sentence_citations"] = sentence_citations
        
        logger.debug(f"DEBUG: Final metadata before StreamChunk creation: {final_metadata}")

        final_sse_event_json = StreamChunk(
            content="",
            is_last=True,
            metadata=final_metadata 
        ).json()
        temp_stream_chunk = StreamChunk(content="", is_last=True, metadata=final_metadata)
        logger.debug(f"DEBUG: Final StreamChunk object's dict (pre-json): {temp_stream_chunk.__dict__}")

        logger.debug(f"Yielding final raw JSON chunk: {final_sse_event_json}")
        yield f"data: {final_sse_event_json}\n\n".encode("utf-8")
        
    async def _stream_llm_without_rag_context(self, 
                                              question: str, 
                                              chat_history_context_string: str, 
                                              generated_title: Optional[str] = None,
                                              final_referenced_material_ids: Optional[List[str]] = None) -> AsyncGenerator[bytes, None]:
        """
        在召回文档为空时，直接调用LLM生成回复。
        """
        logger.info(f"RAG context is empty. Calling LLM directly for question: '{question}'")
        
        final_prompt_for_llm = self.general_chat_prompt_template.format(
            chat_history_context=chat_history_context_string,
            query_str=question
        )
        
        llm_response_gen = None 
        max_retries = self.rag_config.llm_max_retries
        for attempt in range(max_retries):
            try:
                llm_response_gen = await self.llm.astream_chat(messages=[ 
                     {"role": "system", "content": final_prompt_for_llm}, 
                     {"role": "user", "content": question} 
                 ])
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error calling LLM without RAG context for '{question}': {e}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.rag_config.retry_base_delay * (2 ** attempt))
                    continue
                else:
                    error_chunk_json = StreamChunk(content="抱歉，AI未能生成有效回复，请稍后再试。", is_last=True).json()
                    yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                    return
        if not llm_response_gen:
            error_chunk_json = StreamChunk(content="AI未能生成有效回复，请尝试换种方式提问。", is_last=True).json()
            yield f"data: {error_chunk_json}\n\n".encode("utf-8")
            return

        full_response_content = ""
        async for chunk_resp in llm_response_gen:
            chunk_text = chunk_resp.delta.content
            full_response_content += chunk_text
            sse_event_json = StreamChunk(content=chunk_text, is_last=False).json()
            yield f"data: {sse_event_json}\n\n".encode("utf-8")
        
        final_metadata = {
            "rag_sources": [],
            "referenced_docs": final_referenced_material_ids,
            "session_title": generated_title
        }
        final_sse_event_json = StreamChunk(content="", is_last=True, metadata=final_metadata).json()
        yield f"data: {final_sse_event_json}\n\n".encode("utf-8")

        return
    