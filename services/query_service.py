import json
import logging
import asyncio

from typing import Optional, Any, AsyncGenerator, List, Dict
from memory_profiler import profile
import re
import numpy as np

from services.abstract_kb_service import AbstractKnowledgeBaseService
from llama_index.core import PromptTemplate
from services.indexer_service import IndexerService
from services.chat_history_service import ChatHistoryService
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.embeddings import BaseEmbedding

# 定义截断长度
TRUNCATE_LENGTH = 60

# 引入一个可以替代 LlamaIndex 节点的简单数据结构，或者直接使用字典
from dataclasses import dataclass
@dataclass
class SimpleNode:
    text: str
    metadata: Dict[str, Any]
    id_: Optional[str] = None 

from core.rag_config import RagConfig
from core.config import settings
from models.schemas import ChatQueryRequest, StreamChunk
from tools.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self, 
                 llm: Any,
                 embedding_model: BaseEmbedding,
                 indexer_service: IndexerService,
                 kb_service: AbstractKnowledgeBaseService, 
                 rag_config: RagConfig, 
                 chat_history_service: Optional[ChatHistoryService] = None):
        """构造函数 - 切换到抽象 RAG 服务"""
        self.llm = llm             
        self.embedding_model = embedding_model              
        self._indexer_service = indexer_service
        # 使用通用名称 self.kb_service
        self.kb_service = kb_service 
        self._chat_history_service = chat_history_service
        self.rag_config = rag_config

        # 使用配置中的提示词
        # 这里可以使用字符串格式化代替 PromptTemplate，以减少对 LlamaIndex 的依赖
        self.qa_prompt_template = rag_config.qa_prompt_template
        self.title_generation_prompt_template = rag_config.title_prompt_template
        self.general_chat_prompt_template = rag_config.general_chat_prompt_template
        
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
        
        request_params = request.dict()
        logger.info(f"ChatQueryRequest parameters: {request_params}")
        
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

        # --- 流程1: 统一调用抽象服务的检索服务 ---
        knowledge_base_ids_to_query = []
        default_index_id = settings.BAILIAN_INDEX_ID if hasattr(settings, 'BAILIAN_INDEX_ID') else settings.VOLCANO_ENGINE_KNOWLEDGE_BASE_ID
        
        if query_type == "PaperCut":
            knowledge_base_ids_to_query = [default_index_id]
            logger.info("Query type is 'PaperCut', retrieving from specific knowledge base.")
        elif query_type:
            knowledge_base_ids_to_query = [default_index_id] 
            logger.info(f"Query type is '{query_type}', retrieving from knowledge base.")
        else:
            knowledge_base_ids_to_query = [default_index_id] 
            logger.info("Query type not specified, retrieving from default knowledge base.")
        
        final_top_k = request.similarity_top_k if request.similarity_top_k is not None else rag_config.retrieval_top_k
        
        # --- 流程2: 异步并行调用知识库检索API ---
        retrieve_tasks = []
        # 构建文档过滤器
        filters = {}
        if request.target_file_ids:
            logger.info(f"Applying doc_id filter: {request.target_file_ids}. Note: Filter support depends on underlying KB implementation.")
            # 简化 filters 示例: 
            filters = {"doc_id_list": request.target_file_ids} # 传递简化后的过滤器，由底层服务决定如何处理

        for kb_id in knowledge_base_ids_to_query:
            # 调用抽象接口
            task = self.kb_service.retrieve_documents(
                query_text=request.question,
                knowledge_base_id=kb_id,
                limit=final_top_k,
                rerank_switch=True,
                filters=filters
            )
            retrieve_tasks.append(task)
        
        # 使用 asyncio.gather 并行执行所有召回任务
        results_from_collections = await asyncio.gather(*retrieve_tasks)

        # --- 流程3: 合并结果集 ---
        combined_retrieved_chunks = []
        for result_list in results_from_collections:
            combined_retrieved_chunks.extend(result_list)
        
        logger.info(f"Retrieved {len(combined_retrieved_chunks)} chunks from all knowledge bases combined.")

        # 1. 切片级别的过滤
        score_threshold = self.rag_config.retrieval_score_threshold

        # 过滤出所有分数高于阈值的切片
        filtered_chunks = [
            chunk for chunk in combined_retrieved_chunks
            if chunk.get('rerank_score', chunk.get('score', 0.0)) >= score_threshold
        ]

        # 2. 重新计算最高分数（基于已过滤的切片）
        highest_score = 0.0
        if filtered_chunks:
            highest_score = max([chunk.get('rerank_score', chunk.get('score', 0.0)) for chunk in filtered_chunks])
        
        logger.info(f"Filtered down to {len(filtered_chunks)} chunks (Score >= {score_threshold}). Highest score: {highest_score}")

        # 3. 如果有效切片为空，则直接回退
        if not filtered_chunks: # 仅检查 filtered_chunks 是否为空
            logger.info("No relevant documents found (all chunks scored below threshold or empty). Falling back to LLM general chat.")
            async for chunk in self._stream_llm_without_rag_context(
                request.question,
                chat_history_context_string,
                generated_title
            ):
                yield chunk
            return
        
        # --- 流程4: 准备传递给 LLM 的上下文 ---
        final_retrieved_chunks_for_llm = filtered_chunks
        
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
        retrieved_chunk_map = {}
        
        # 使用知识库返回的 chunk 数据构建上下文
        for chunk in final_retrieved_chunks_for_llm:
            chunk_content = chunk.get('content', '')
            chunk_source = chunk.get('source', '未知文件')
            chunk_doc_id = chunk.get('docId', 'N/A')
            chunk_material_id = chunk.get('material_id', 'N/A')
            chunk_id = f"{chunk_doc_id}-{chunk.get('chunkId', 'N/A')}" 
            
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
                "page_number": "N/A", 
                "material_id": chunk_material_id
            })

            retrieved_chunk_map[chunk_id] = SimpleNode(
                text=chunk_content,
                metadata={
                    'document_id': chunk_doc_id,
                    'material_id': chunk_material_id,
                    'file_name': chunk_source,
                    'page_label': "N/A"
                },
                id_=chunk_id
            )

        context_str_for_llm = "\n\n".join(context_parts)
        
        max_retries = 3
        # --- 主 RAG 查询 LLM ---
        llm_response_obj = None 
        for attempt in range(max_retries):
            try:
                self.qa_prompt_template = PromptTemplate(rag_config.qa_prompt_template)
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
                messages_for_llm = [
                    ChatMessage(role=MessageRole.SYSTEM, content=final_prompt_for_llm),
                    ChatMessage(role=MessageRole.USER, content=request.question)
                ]

                llm_response_gen = await self.llm.astream_chat(messages=messages_for_llm)

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
        async for chunk_resp in llm_response_gen:
            # 1. 首先检查 chunk_resp 是否有效
            if not chunk_resp:
                continue

            # 2. 根据 astream_chat 的源码，chunk_resp 是一个 ChatResponse 对象
            # 它包含一个 `delta` 属性，即本次新增的文本块
            # 我们需要确保它是一个有效的 ChatResponse 对象，并且它的 delta 不为空
            if hasattr(chunk_resp, 'delta') and chunk_resp.delta:
                chunk_text = chunk_resp.delta
                
                # 累积完整的回复内容，用于后续的引用生成
                full_response_content += chunk_text
                
                # 将每次的增量文本块包装成 StreamChunk 并发送
                sse_event_json = StreamChunk(content=chunk_text, is_last=False).json()
                logger.info(f"SSE TX: data: {sse_event_json[:70]}...") 
                yield f"data: {sse_event_json}\n\n".encode("utf-8")
            else:
                # 记录警告，以防出现预料之外的 chunk 类型
                logger.warning(f"Received unexpected chunk from LLM: type={type(chunk_resp)}. Skipping.")
                continue
            
        logger.debug("Successfully streamed response chunks.")

        if not full_response_content or full_response_content.strip() == "NO_RELEVANT_DOCUMENTS_FOUND":
             # 处理LLM找不到答案的情况
             logger.info("LLM did not find a relevant response based on context.")
             final_sse_event_json = StreamChunk(content="抱歉，我未能从现有资料中找到相关信息。", is_last=True).json()
             logger.info(f"SSE TX (Final - No Content): data: {final_sse_event_json}")
             yield f"data: {final_sse_event_json}\n\n".encode("utf-8")
             return
        
        # --- 句子级引用后处理逻辑 ---
        response_sentences = []
        sentence_citations = []
        if full_response_content:
            response_sentences = re.split(r'(?<=[.!?。！？])\s+', full_response_content)

        if self.embedding_model:
            referenced_chunk_texts = [node.text for node in retrieved_chunk_map.values()]
            referenced_chunk_ids = list(retrieved_chunk_map.keys())

            combined_referenced_texts = referenced_chunk_texts + chat_history_chunk_texts
            combined_referenced_ids = referenced_chunk_ids + chat_history_chunk_ids

            if not combined_referenced_texts:
                logger.warning("No texts to embed for citation. Skipping citation generation.")
                sentence_citations = []
            else:
                all_text_to_embed = response_sentences + combined_referenced_texts
                all_embeddings = await self.embedding_model.aget_text_embedding_batch(all_text_to_embed, show_progress=False)

                response_sentence_embeddings = all_embeddings[:len(response_sentences)]
                referenced_chunk_embeddings = all_embeddings[len(response_sentences):]

                def cosine_similarity(v1, v2):
                    v1 = np.array(v1)
                    v2 = np.array(v2)
                    dot_product = np.dot(v1, v2)
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    if norm_v1 == 0 or norm_v2 == 0:
                        return 0
                    return dot_product / (norm_v1 * norm_v2)

                for i, sentence in enumerate(response_sentences):
                    sentence_embedding = response_sentence_embeddings[i]
                    best_match_id = None
                    max_similarity = -1.0
                    best_match_type = ""
                    citation_info = {}

                    for j, chunk_embedding in enumerate(referenced_chunk_embeddings):
                        similarity = cosine_similarity(sentence_embedding, chunk_embedding)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match_id = combined_referenced_ids[j]
                            if j < len(referenced_chunk_ids):
                                best_match_type = "document"
                            else:
                                best_match_type = "chat_history"

                    if max_similarity > self.rag_config.citation_similarity_threshold:
                        # 获取完整的引用文本
                        full_referenced_text = combined_referenced_texts[combined_referenced_ids.index(best_match_id)]
                        
                        # 核心修改点：对引用文本进行截断
                        truncated_referenced_text = (full_referenced_text[:TRUNCATE_LENGTH] + '...') if len(full_referenced_text) > TRUNCATE_LENGTH else full_referenced_text

                        citation_info = {
                            "sentence": sentence,
                            "referenced_chunk_id": best_match_id, 
                            "source_type": best_match_type, 
                            "referenced_chunk_text": truncated_referenced_text,
                            "document_id": None,
                            "material_id": None,
                            "file_name": None,
                            "page_label": None
                        }

                        if best_match_type == "document":
                            source_node = retrieved_chunk_map.get(best_match_id)
                            if source_node:
                                citation_info.update({
                                    "document_id": source_node.metadata.get('document_id'),
                                    "material_id": source_node.metadata.get('material_id'),
                                    "file_name": source_node.metadata.get('file_name'),
                                    "page_label": source_node.metadata.get('page_label')
                                })
                        else: 
                            source_node = next((node for node in chat_history_context_nodes if node.id_ == best_match_id), None)
                            if source_node:
                                citation_info.update({
                                    "document_id": None,
                                    "material_id": None,
                                    "file_name": "聊天历史",
                                    "page_label": source_node.metadata.get('role', '未知')
                                })
                        sentence_citations.append(citation_info)
        else:
            logger.warning("Embedding model is not available for sentence-level citation. Skipping this step.")

        # --- 构建最终的 metadata ---
        final_metadata = {}
        final_metadata["rag_sources"] = rag_sources_info 

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

        logger.info(f"SSE TX (Final Metadata): {final_sse_event_json}")
        yield f"data: {final_sse_event_json}\n\n".encode("utf-8")
        
    async def _stream_llm_without_rag_context(self, 
                                              question: str, 
                                              chat_history_context_string: str, 
                                              generated_title: Optional[str] = None,
                                              final_referenced_material_ids: Optional[List[str]] = None) -> AsyncGenerator[bytes, None]:
        
        logger.info(f"RAG context is empty. Calling LLM directly for question: '{question}'")
        
        final_prompt_for_llm = self.general_chat_prompt_template.format(
            chat_history_context=chat_history_context_string,
            query_str=question
        )
        
        llm_response_gen = None 
        max_retries = self.rag_config.llm_max_retries
        for attempt in range(max_retries):
            try:
                # 使用 LlamaIndex 的 ChatMessage 对象
                messages_for_llm = [
                    ChatMessage(role=MessageRole.SYSTEM, content=final_prompt_for_llm),
                    ChatMessage(role=MessageRole.USER, content=question)
                ]

                llm_response_gen = await self.llm.astream_chat(messages=messages_for_llm)
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error calling LLM without RAG context for '{question}': {e}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.rag_config.retry_base_delay * (2 ** attempt))
                    continue
                else:
                    error_chunk_json = StreamChunk(content="抱歉，AI未能生成有效回复，请稍后再试。", is_last=True).json()
                    logger.info(f"SSE TX: {error_chunk_json}")
                    yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                    return
        if not llm_response_gen:
            error_chunk_json = StreamChunk(content="AI未能生成有效回复，请尝试换种方式提问。", is_last=True).json()
            logger.info(f"SSE TX: {error_chunk_json}")
            yield f"data: {error_chunk_json}\n\n".encode("utf-8")
            return

        full_response_content = ""
        # 接收并流式返回 LLM 响应
        async for chunk_resp in llm_response_gen:
            # 1. 首先检查 chunk_resp 是否有效
            if not chunk_resp:
                continue

            # 2. 根据 astream_chat 的源码，chunk_resp 是一个 ChatResponse 对象
            # 它包含一个 `delta` 属性，即本次新增的文本块
            # 我们需要确保它是一个有效的 ChatResponse 对象，并且它的 delta 不为空
            if hasattr(chunk_resp, 'delta') and chunk_resp.delta:
                chunk_text = chunk_resp.delta
                
                # 累积完整的回复内容，用于后续的引用生成
                full_response_content += chunk_text
                
                # 将每次的增量文本块包装成 StreamChunk 并发送
                sse_event_json = StreamChunk(content=chunk_text, is_last=False).json()
                logger.info(f"SSE TX (Fallback): data: {sse_event_json}")
                yield f"data: {sse_event_json}\n\n".encode("utf-8")
            else:
                # 记录警告，以防出现预料之外的 chunk 类型
                logger.warning(f"Received unexpected chunk from LLM: type={type(chunk_resp)}. Skipping.")
                continue
        
        final_metadata = {
            "rag_sources": "[]", 
            "referenced_docs": final_referenced_material_ids,
            "session_title": generated_title
        }
        final_sse_event_json = StreamChunk(content="", is_last=True, metadata=final_metadata).json()
        logger.info(f"SSE TX (Fallback Final Metadata): {final_sse_event_json}")
        yield f"data: {final_sse_event_json}\n\n".encode("utf-8")

        return