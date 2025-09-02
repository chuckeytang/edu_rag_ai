import json
import logging
from fastapi import Depends
import asyncio

from typing import Optional
from memory_profiler import profile

from services.indexer_service import IndexerService
from services.retrieval_service import RetrievalService
from tools.tokenizer_utils import get_tokenizer
from services.chat_history_service import ChatHistoryService

logger = logging.getLogger(__name__)

from typing import Any, AsyncGenerator, List
from llama_index.core.schema import Document as NodeWithScore, TextNode as LlamaTextNode, QueryBundle

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine 
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank 
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.llms import ChatMessage
from core.rag_config import RagConfig
import torch 

from core.config import Settings, settings
from models.schemas import ChatQueryRequest, StreamChunk
from typing import List, Dict
import chromadb

PERSIST_DIR = settings.INDEX_PATH

class CustomRetriever(BaseRetriever):
    """
    一个自定义的 Retriever，它不执行实际的向量搜索，
    而是直接返回初始化时传入的节点列表。
    用于在 RAG 召回为 0 时，将虚拟节点传递给 QueryEngine。
    """
    def __init__(self, nodes: List[NodeWithScore]):
        self._nodes = nodes
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        logger.debug(f"CustomRetriever: _aretrieve called for query '{query_bundle.query_str[:50]}...'. Returning {len(self._nodes)} pre-set nodes.")
        return self._nodes
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        logger.debug(f"CustomRetriever: _retrieve called for query '{query_bundle.query_str[:50]}...'. Returning {len(self._nodes)} pre-set nodes.")
        return self._nodes
    
class QueryService:
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embedding_model: BaseEmbedding,
                 llm: LLM, 
                 indexer_service: IndexerService,
                 rag_config: RagConfig, 
                 retrieval_service: RetrievalService,
                 chat_history_service: Optional[ChatHistoryService] = None,
                 deepseek_llm_for_reranker: Optional[LLM] = None):
        """
        构造函数 - 实现“按需加载”策略
        启动时只初始化客户端和空的索引缓存，不加载任何实际的索引数据。
        """
        self.chroma_client = chroma_client       # 使用传入的客户端
        self.embedding_model = embedding_model   # 使用传入的 embedding model
        self.llm = llm                           # 使用传入的 LLM
        
        self.retrieval_service = retrieval_service
        self.indices: Dict[str, VectorStoreIndex] = {}
        self._indexer_service = indexer_service

        # 保存 chat_history_service 实例
        self._chat_history_service = chat_history_service
        self.rag_config = rag_config

        # 使用配置中的提示词
        self.qa_prompt = PromptTemplate(rag_config.qa_prompt_template)
        self.title_generation_prompt = PromptTemplate(rag_config.title_prompt_template)
        self.general_chat_prompt = PromptTemplate(rag_config.general_chat_prompt_template)
        
        # 1. 本地模型 SentenceTransformer Reranker
        # self.local_reranker = SentenceTransformerRerank(
        #     model="BAAI/bge-reranker-base", 
        #     top_n=rag_config.retrieval_top_k*rag_config.initial_retrieval_multiplier, 
        #     device="cuda" if torch.cuda.is_available() else "cpu" 
        # )
        # logger.info(f"Initialized Local Reranker: SentenceTransformerRerank with model '{self.local_reranker.model}' on device '{self.local_reranker.device}' and fixed top_n={self.local_reranker.top_n}.")

        # 2. 基于 LLM 的 Reranker (使用 DeepSeek)
        self.llm_reranker = None
        if deepseek_llm_for_reranker and rag_config.reranker_type == "llm":
            self.llm_reranker = LLMRerank(
                llm=deepseek_llm_for_reranker, # 传入 DeepSeek LLM 实例
                top_n=rag_config.retrieval_top_k*rag_config.initial_retrieval_multiplier
            )
            logger.info(f"Initialized LLM Reranker: LLMRerank with model '{deepseek_llm_for_reranker.model}' and fixed top_n={self.llm_reranker.top_n}.")
        else:
            logger.warning("DeepSeek LLM for Reranker not provided. LLM Reranker will not be initialized.")
        
    def _get_or_load_index(self, collection_name: str) -> VectorStoreIndex:
        """
        重定向到 IndexerService 的按需加载方法。
        """
        return self._indexer_service.get_rag_index(collection_name)

    def retrieve_with_filters(self, question: str, collection_name: str, filters: dict, similarity_top_k: int = 5):
        """
        [DEBUG METHOD] 仅执行带过滤的召回，并返回召回的节点列表。
        此方法现在使用通用的 RetrievalService。
        """
        logger.info(f"[DEBUG] Retrieving from collection '{collection_name}' with filters: {filters}")
        
        # 直接调用 RetrievalService 的通用召回方法
        
        try:
            # 遵照你的原始代码格式，我们暂时假设存在一个同步版本或者在调用时进行异步处理
            final_retrieved_nodes = self.retrieval_service.retrieve_documents_sync(
                query_text=question,
                collection_name=collection_name,
                filters=filters,
                top_k=similarity_top_k,
                use_reranker=False # 此处不需要重排器
            )
            
            logger.info(f"[DEBUG] RetrievalService returned {len(final_retrieved_nodes)} nodes.")
            return final_retrieved_nodes
        except Exception as e:
            logger.error(f"[DEBUG] Error during retrieval: {e}", exc_info=True)
            raise ValueError(f"Retrieval failed: {e}")
    
    async def retrieve_only_for_function_calling(self,
                                                 query_text: str,
                                                 collection_name: str,
                                                 filters: Optional[Dict[str, Any]] = None,
                                                 top_k: int = 15) -> List[int]:
        """
        专为 Function Calling 提供的纯文档召回方法，只返回 material_id 列表。
        此方法**不应用重排器**。
        """
        retrieved_nodes = await self.retrieval_service.retrieve_documents(
            query_text=query_text,
            collection_name=collection_name,
            filters=filters,
            top_k=top_k,
            use_reranker=False # 核心：在这里禁用重排器
        )
        
        material_ids = []
        for node_with_score in retrieved_nodes:
            node = node_with_score.node
            material_id = node.metadata.get("material_id")
            if material_id:
                material_ids.append(int(material_id))
        
        return list(set(material_ids))

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
                    # 准备聊天历史chunk的文本和ID
                    chat_history_chunk_texts = [node.text for node in chat_history_context_nodes]
                    # 使用node.id_作为唯一标识符
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

        # --- 流程1: 根据 type 分流召回 ---
        collection_names_to_query = []
        if query_type == "PaperCut":
            collection_names_to_query = ["paper_cut_collection"]
            logger.info("Query type is 'PaperCut', retrieving from 'paper_cut_collection'.")
        elif query_type and query_type != "PaperCut":
            # 假设其他所有类型都在 public_collection
            collection_names_to_query = [request.collection_name]
            logger.info(f"Query type is '{query_type}', retrieving from '{request.collection_name}'.")
        else: # type 为空或未指定
            collection_names_to_query = [request.collection_name, "paper_cut_collection"]
            logger.info("Query type not specified, retrieving from 'public_collection' and 'paper_cut_collection'.")
        
        # 在使用 request.similarity_top_k 之前，为其提供一个默认值
        # 例如，如果 request.similarity_top_k 为 None，则使用 rag_config.retrieval_top_k 作为默认值
        final_top_k = request.similarity_top_k if request.similarity_top_k is not None else rag_config.retrieval_top_k
        
        # --- 流程2: 异步并行召回 ---
        retrieve_tasks = []
        for collection_name in collection_names_to_query:
            # 这里的 filters 已经移除了 'type'
            task = self.retrieval_service.retrieve_documents(
                query_text=request.question,
                collection_name=collection_name,
                # 在这里，我们将 filters 设置为 None，因为所有过滤都将后置处理
                filters=None,
                top_k=final_top_k * rag_config.initial_retrieval_multiplier,
                use_reranker=False, # 禁用重排器，重排将在合并后进行
            )
            retrieve_tasks.append(task)

        # 使用 asyncio.gather 并行执行所有召回任务
        results_from_collections = await asyncio.gather(*retrieve_tasks)

        # --- 流程3: 合并结果集 ---
        combined_retrieved_nodes = []
        for result_list in results_from_collections:
            combined_retrieved_nodes.extend(result_list)
        
        logger.info(f"Retrieved {len(combined_retrieved_nodes)} chunks from all collections combined.")

        # 检查是否召回了文档
        if not combined_retrieved_nodes:
            logger.info("No documents retrieved for the main query. Directly calling LLM.")
            async for chunk in self._stream_llm_without_rag_context(
                request.question,
                chat_history_context_string,
                generated_title
            ):
                yield chunk
            return
        
        # --- 流程4: 后置元数据过滤 ---
        final_filtered_nodes = []
        # 使用 _build_chroma_where_clause 方法来处理复杂的过滤器
        filter_fn = self.retrieval_service._build_chroma_where_clause(combined_rag_filters)
        if filter_fn:
            logger.info(f"Applying post-retrieval filters: {combined_rag_filters}")
            for node_with_score in combined_retrieved_nodes:
                if self._check_node_with_filter(node_with_score.node, combined_rag_filters):
                    final_filtered_nodes.append(node_with_score)
        else:
            final_filtered_nodes = combined_retrieved_nodes
            logger.info("No metadata filters to apply, skipping post-retrieval filtering.")

        # --- 流程5: 后置重排 ---
        if self.rag_config.use_reranker and len(final_filtered_nodes) > 0:
            logger.info("Applying reranker on the filtered nodes.")
            query_bundle_for_rerank = QueryBundle(query_str=request.question)
            if self.rag_config.reranker_type == "llm" and self.llm_reranker:
                final_retrieved_nodes = self.llm_reranker._postprocess_nodes(
                    nodes=final_filtered_nodes, query_bundle=query_bundle_for_rerank
                )
            elif self.rag_config.reranker_type == "local" and self.local_reranker:
                final_retrieved_nodes = self.local_reranker.postprocess_nodes(
                    final_filtered_nodes, query_bundle=query_bundle_for_rerank
                )
            else:
                logger.warning("Reranker type is unknown. Skipping reranking.")
                final_retrieved_nodes = final_filtered_nodes
        else:
            final_retrieved_nodes = final_filtered_nodes
            logger.info("Reranking disabled or no nodes to rerank.")
        
        # 确保最终返回的节点数量不超过 top_k
        final_retrieved_nodes = final_retrieved_nodes[:request.similarity_top_k]

        # --- 计算并限制发送给 LLM 的总 token 数量 ---
        tokenizer = get_tokenizer()
        if not tokenizer:
            logger.warning("Tokenizer not available, cannot perform token-based content trimming.")
        
        llm_max_context_tokens = rag_config.llm_max_context_tokens
        
        # 计算固定部分的 token 数量 (Prompt 模板 + 聊天历史 + 用户问题)
        # 这里的 `qa_prompt.template` 是一个更准确的计算方法，它包含了整个模板文本
        static_prompt_tokens = 0
        if tokenizer:
            try:
                # 这是一个更精确的计算，因为它是整个模板的 token
                template_tokens = len(tokenizer.encode(self.qa_prompt.template))
                history_tokens = len(tokenizer.encode(chat_history_context_string))
                query_tokens = len(tokenizer.encode(request.question))
                
                # 减去占位符的 tokens，因为它们会被实际内容替换
                placeholder_tokens = len(tokenizer.encode('{chat_history_context}{context_str}{query_str}'))
                
                static_prompt_tokens = template_tokens + history_tokens + query_tokens - placeholder_tokens
                
            except Exception as e:
                logger.warning(f"Failed to calculate static prompt tokens: {e}. Falling back to approximation.")
                static_prompt_tokens = len(tokenizer.encode(self.qa_prompt.format(
                    chat_history_context="", context_str="", query_str=""
                )))
                static_prompt_tokens += len(tokenizer.encode(chat_history_context_string))
                static_prompt_tokens += len(tokenizer.encode(request.question))
        
        # 预留一些空间给 LLM 的回答，以防万一
        reserved_tokens = 500
        remaining_context_tokens = llm_max_context_tokens - static_prompt_tokens - reserved_tokens
        
        logger.info(f"LLM max context: {llm_max_context_tokens}, static prompt tokens: {static_prompt_tokens}, remaining for RAG content: {remaining_context_tokens}")
        
        
        final_retrieved_nodes_for_llm = []
        context_parts = []
        current_context_tokens = 0
        rag_sources_info = []
        retrieved_node_map = {}

        for n in final_retrieved_nodes: # 使用重排后的节点列表
            # 检查节点类型，确保代码健壮性
            if isinstance(n, NodeWithScore):
                node = n.node
            else:
                # 假设它是一个直接的文档或文本节点
                node = n
            
            # 过滤掉虚拟节点
            if node.metadata.get("source") == "virtual_fallback" or \
               node.metadata.get("document_type") == "no_document_found":
                continue

            node_content = n.get_content()
            if tokenizer:
                node_tokens = len(tokenizer.encode(node_content))
                if current_context_tokens + node_tokens > remaining_context_tokens:
                    logger.warning(f"Exceeding LLM context window. Truncating RAG content. Remaining tokens: {remaining_context_tokens}, trying to add {node_tokens}.")
                    break # 超过限制，停止添加文档
                current_context_tokens += node_tokens

            # 添加到最终的节点列表中
            final_retrieved_nodes_for_llm.append(n)
            
            doc_type = n.node.metadata.get("document_type", "Unknown")
            page_label = n.node.metadata.get("page_label", "N/A")
            file_name = n.node.metadata.get("file_name", "N/A")
            context_parts.append(
                f"--- Document Content (Type: {doc_type}, Page: {page_label}, File: {file_name}) ---\n"
                f"{n.get_content()}\n"
                f"--------------------------------------------------------------------------------"
            )

            # 构建来源信息
            material_id = node.metadata.get("material_id")
            if '_node_content' in node.metadata and isinstance(node.metadata['_node_content'], str):
                try:
                    node_content_data = json.loads(node.metadata['_node_content'])
                    node_content_meta = node_content_data.get('metadata', {})
                    file_name = node_content_meta.get('file_name', file_name)
                    page_label = node_content_meta.get('page_label', page_label)
                except json.JSONDecodeError:
                    pass

            rag_sources_info.append({
                "file_name": file_name,
                "page_number": page_label,
                "material_id": material_id
            })

        context_str_for_llm = "\n\n".join(context_parts)

        logger.debug(f"Created map of {len(retrieved_node_map)} retrieved nodes for lookup.")

        max_retries = 3
        # --- 主 RAG 查询 LLM ---
        llm_response_obj = None 
        for attempt in range(max_retries):
            try:
                final_prompt_for_llm = self.qa_prompt.format(
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

                query_engine = RetrieverQueryEngine.from_args(
                    retriever=CustomRetriever(final_retrieved_nodes_for_llm),
                    llm=self.llm,
                    streaming=True,
                    text_qa_template=PromptTemplate(final_prompt_for_llm) 
                )
                
                llm_response_obj = await query_engine.aquery(request.question) 

                is_response_empty = False
                if llm_response_obj and hasattr(llm_response_obj, 'response_gen') and llm_response_obj.response_gen is not None:
                    pass 
                elif llm_response_obj and hasattr(llm_response_obj, 'response') and (not llm_response_obj.response or llm_response_obj.response.strip() == ""):
                    is_response_empty = True
                elif not llm_response_obj: 
                     is_response_empty = True

                if is_response_empty:
                    logger.warning(f"Attempt {attempt + 1}: LLM returned an empty or nearly empty response for query: '{request.question}'. Retrying...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt) 
                        continue 
                    else:
                        logger.error(f"All {max_retries} attempts failed for query: '{request.question}', returning error to client.")
                        logger.error(f"Failed query input details: \n"
                                     f"Chat History: {chat_history_context_string}\n"
                                     f"Document Content: {context_str_for_llm}\n" 
                                     f"User Query: {request.question}\n"
                                     f"Full Prompt (first 500 chars): {final_prompt_for_llm[:500]}...")
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
                    logger.error(f"Failed query input details: \n"
                                 f"Chat History: {chat_history_context_string}\n"
                                 f"Document Content: {context_str_for_llm}\n" 
                                 f"User Query: {request.question}\n"
                                 f"Full Prompt (first 500 chars): {final_prompt_for_llm[:500]}...")
                    error_chunk_json = StreamChunk(content="Sorry, an error occurred during the query. Please try again later.", is_last=True).json()
                    yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                    return

        rag_sources_info = []
        # === 基于 final_retrieved_nodes 来构建来源信息 ===
        # 确保只包含实际的文档节点，排除虚拟节点
        for node_with_score in final_retrieved_nodes:
            node = node_with_score.node # 获取 LlamaIndex 节点对象
            
            # 过滤掉虚拟节点
            if node.metadata.get("source") == "virtual_fallback" or \
               node.metadata.get("document_type") == "no_document_found": # 确保捕获所有虚拟节点标识
                continue
                
            # 确保 file_name 和 page_label 存在于 metadata 中，并处理可能的嵌套
            file_name = node.metadata.get("file_name", "未知文件")
            page_label = node.metadata.get("page_label", "未知页")
            material_id = node.metadata.get("material_id")

            # 如果 file_name 或 page_label 可能在 _node_content 的 metadata 中，需要解析
            # 这与之前在 debug_index 中的逻辑类似
            if '_node_content' in node.metadata and isinstance(node.metadata['_node_content'], str):
                try:
                    node_content_data = json.loads(node.metadata['_node_content'])
                    node_content_meta = node_content_data.get('metadata', {})
                    file_name = node_content_meta.get('file_name', file_name)
                    page_label = node_content_meta.get('page_label', page_label)
                except json.JSONDecodeError:
                    pass

            rag_sources_info.append({
                "file_name": file_name,
                "page_number": page_label,
                "material_id": material_id # 确保 material_id 也被正确传递
            })
        
        full_response_content = ""
        if llm_response_obj and hasattr(llm_response_obj, 'response_gen'):
            async for chunk_text in llm_response_obj.response_gen:
                full_response_content += chunk_text
                sse_event_json = StreamChunk(content=chunk_text, is_last=False).json()
                yield f"data: {sse_event_json}\n\n".encode("utf-8") 
            logger.debug("Successfully streamed response chunks.")
        elif llm_response_obj and hasattr(llm_response_obj, 'response') and llm_response_obj.response: 
            full_response_content = llm_response_obj.response
            sse_event_json = StreamChunk(content=full_response_content, is_last=False).json()
            yield f"data: {sse_event_json}\n\n".encode("utf-8")
            logger.warning(f"Non-streaming QueryEngine response yielded once: '{full_response_content[:50]}...'. Expected streaming.")
        else:
            logger.warning(f"No valid LLM response content (stream or direct response) for query: '{request.question}'. This should have been caught by retry logic.")
            error_chunk_json = StreamChunk(content="AI未能生成有效回复，请尝试换种方式提问。", is_last=True).json()
            yield f"data: {error_chunk_json}\n\n".encode("utf-8")
            return 
        

        if not full_response_content or full_response_content.strip() == "NO_RELEVANT_DOCUMENTS_FOUND":
             # 处理LLM找不到答案的情况
             # ... ( fallback 逻辑) ...
             return
    
        # 获取所有被引用的 chunk 的文本列表和 ID 列表
        referenced_chunk_texts = [node.text for node in retrieved_node_map.values()]
        referenced_chunk_ids = list(retrieved_node_map.keys())

        # 新增逻辑：合并文档 chunk 和聊天历史 chunk
        combined_referenced_texts = referenced_chunk_texts + chat_history_chunk_texts
        combined_referenced_ids = referenced_chunk_ids + chat_history_chunk_ids

        # 步骤 3.1: 将回答分割成句子
        import re
        # 使用正则表达式进行句子分割
        # 注意：这可能不完美，但对于大多数情况足够
        response_sentences = re.split(r'(?<=[.!?。！？])\s+', full_response_content)
        
        sentence_citations = []
        
        # 步骤 3.2: 遍历每个句子，找到它最相似的引用 chunk
        # LlamaIndex 的 `embedding_model` 实例可以用来生成句子向量
        # 你需要在 QueryService 中注入 embedding_model
        
        if self.embedding_model:
            # 批量获取句子和 chunk 的 embeddings，效率更高
            all_text_to_embed = response_sentences + combined_referenced_texts
            all_embeddings = await self.embedding_model.aget_text_embedding_batch(all_text_to_embed, show_progress=False)
            
            response_sentence_embeddings = all_embeddings[:len(response_sentences)]
            referenced_chunk_embeddings = all_embeddings[len(response_sentences):]

            import numpy as np
            # 计算余弦相似度
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
                best_match_type = "" # 标记引用来源类型
                citation_info = {}
                
                for j, chunk_embedding in enumerate(referenced_chunk_embeddings):
                    similarity = cosine_similarity(sentence_embedding, chunk_embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_id = combined_referenced_ids[j]
                        # 根据索引判断来源类型
                        if j < len(referenced_chunk_ids):
                            best_match_type = "document"
                        else:
                            best_match_type = "chat_history"


                # 阈值判断：如果最高相似度低于某个阈值，可能就不认为是引用
                if max_similarity > self.rag_config.citation_similarity_threshold: 
                    citation_info = { 
                        "sentence": sentence,
                        "referenced_chunk_id": best_match_id,
                        "referenced_chunk_text": combined_referenced_texts[combined_referenced_ids.index(best_match_id)],
                        "source_type": best_match_type, # 新增字段
                        "document_id": None,
                        "material_id": None,
                        "file_name": None,
                        "page_label": None
                    }
                    
                    if best_match_type == "document":
                        source_node = retrieved_node_map.get(best_match_id)
                        if source_node:
                            source_meta = source_node.metadata
                            # 使用 .update() 方法安全地更新字典
                            citation_info.update({
                                "document_id": source_meta.get('document_id'),
                                "material_id": source_meta.get('material_id'),
                                "file_name": source_meta.get('file_name'),
                                "page_label": source_meta.get('page_label')
                            })
                    else: # 聊天历史
                        source_node = next((node for node in chat_history_context_nodes if node.id_ == best_match_id), None)
                        if source_node:
                            citation_info.update({
                                "document_id": None,
                                "material_id": None,
                                "file_name": "聊天历史",
                                "page_label": source_node.metadata.get('role', '未知')
                            })
                    
                    # 只有在满足阈值时才添加引用
                    sentence_citations.append(citation_info)
                        
        else:
            logger.warning("Embedding model not available for sentence-level citation.")
            
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

        logger.debug(f"Yielding final raw JSON chunk: {final_sse_event_json}")
        yield f"data: {final_sse_event_json}\n\n".encode("utf-8")

    # 在没有召回文档时，直接调用LLM
    async def _stream_llm_without_rag_context(self, 
                                              question: str, 
                                              chat_history_context_string: str, 
                                              generated_title: Optional[str] = None,
                                              final_referenced_material_ids: Optional[List[str]] = None) -> AsyncGenerator[bytes, None]:
        """
        在召回文档为空时，直接调用LLM生成回复。
        """
        logger.info(f"RAG context is empty. Calling LLM directly for question: '{question}'")
        
        # 使用通用的 LLM 聊天模板，或者一个简单的无上下文模板
        final_prompt_for_llm = self.general_chat_prompt.format(
            chat_history_context=chat_history_context_string,
            query_str=question
        )
        
        llm_response_gen = None 
        max_retries = self.rag_config.llm_max_retries
        for attempt in range(max_retries):
            try:
                response_gen = await self.llm.astream_chat(messages=[ 
                     ChatMessage(role="system", content=final_prompt_for_llm), 
                     ChatMessage(role="user", content=question) 
                 ])

                llm_response_gen = response_gen
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
            chunk_text = chunk_resp.delta
            full_response_content += chunk_text
            sse_event_json = StreamChunk(content=chunk_text, is_last=False).json()
            yield f"data: {sse_event_json}\n\n".encode("utf-8")
        
        # 封装最终的元数据，不包含任何来源信息
        final_metadata = {
            "rag_sources": [],
            "referenced_docs": final_referenced_material_ids,
            "session_title": generated_title
        }
        final_sse_event_json = StreamChunk(content="", is_last=True, metadata=final_metadata).json()
        yield f"data: {final_sse_event_json}\n\n".encode("utf-8")

        return
    

    def _check_node_with_filter(self, node, filters: Dict[str, Any]) -> bool:
        """
        根据复杂的过滤器字典，检查单个节点的元数据是否匹配。
        此方法需要你自己实现，以处理 '$and' 和 '$in' 等逻辑。
        """
        # 这是一个简化的示例，你需要根据你的 _build_chroma_where_clause 逻辑来实现
        # 更健壮的过滤检查。
        if not filters:
            return True
        
        # 如果是简单的 {"key": "value"} 格式
        if all(isinstance(v, (str, int, float)) for v in filters.values()):
            return all(node.metadata.get(k) == v for k, v in filters.items())

        # 如果是复杂格式，例如 {"$and": [...]}
        if "$and" in filters:
            return all(self._check_node_with_filter(node, sub_filter) for sub_filter in filters["$and"])
        if "$in" in filters and len(filters) == 1:
            key, values = next(iter(filters.items()))
            return node.metadata.get(key) in values
            
        # 简化示例，需要更详细的实现
        return True