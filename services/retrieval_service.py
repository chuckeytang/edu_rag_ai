# services/retrieval_service.py

import logging
from typing import Dict, Any, List, Optional
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.llms.openai_like import OpenAILike
from services.indexer_service import IndexerService
from core.rag_config import RagConfig
import torch

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, indexer_service: IndexerService, rag_config: RagConfig, deepseek_llm_for_reranker: OpenAILike):
        self._indexer_service = indexer_service
        self.rag_config = rag_config
        
        self.llm_reranker = None
        self.local_reranker = None

        if self.rag_config.use_reranker:
            reranker_final_top_n = self.rag_config.retrieval_top_k
            
            if self.rag_config.reranker_type == "llm":
                self.llm_reranker = LLMRerank(
                    llm=deepseek_llm_for_reranker,
                    top_n=reranker_final_top_n
                )
                logger.info("LLM Reranker initialized in RetrievalService.")
            elif self.rag_config.reranker_type == "local":
                self.local_reranker = SentenceTransformerRerank(
                    model="BAAI/bge-reranker-base",
                    top_n=reranker_final_top_n,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("Local (SentenceTransformer) Reranker initialized in RetrievalService.")
            else:
                logger.warning("Reranker type is set to an unknown value. Reranking disabled.")


    # 将更健壮的 _build_chroma_where_clause 方法移到这里
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
                    logger.warning(f"Empty list provided for filter key '{key}'. This condition will be ignored.")
            # 情况 3: 其他所有简单类型的值 (str, int, float)，使用默认的相等比较
            else:
                chroma_filters.append({key: {"$eq": value}})

        if not chroma_filters:
            return {}
        
        # 如果有多个过滤条件，用 $and 连接
        if len(chroma_filters) > 1:
            return {"$and": chroma_filters}
        
        # 只有一个条件，直接返回
        return chroma_filters[0]

    async def retrieve_documents(self,
                                  query_text: str,
                                  collection_name: str,
                                  filters: Optional[Dict[str, Any]] = None,
                                  top_k: Optional[int] = None,
                                  use_reranker: bool = True) -> List[NodeWithScore]:
        """
        通用的文档召回方法，可根据参数决定是否应用重排器。
        """
        rag_index = self._indexer_service.get_rag_index(collection_name)
        if not rag_index:
            logger.error(f"RAG collection '{collection_name}' does not exist or could not be loaded.")
            return []

        # 获取顶层过滤条件，如果存在的话
        chroma_where_clause = self._build_chroma_where_clause(filters)
        logger.info(f"Retrieving documents with ChromaDB `where` clause: {chroma_where_clause}")

        # 初始召回数量应为 top_k * multiplier
        initial_retrieval_top_k = top_k * self.rag_config.initial_retrieval_multiplier if top_k else self.rag_config.retrieval_top_k * self.rag_config.initial_retrieval_multiplier
        
        retriever = rag_index.as_retriever(
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
            similarity_top_k=int(initial_retrieval_top_k)
        )
        retrieved_nodes = await retriever.aretrieve(query_text)
        logger.info(f"Retrieved {len(retrieved_nodes)} chunks for query: '{query_text[:50]}...'")

        final_retrieved_nodes = retrieved_nodes
        
        # 如果启用重排器且use_reranker为True，则应用重排器
        if self.rag_config.use_reranker and use_reranker and (self.llm_reranker or self.local_reranker):
            if len(retrieved_nodes) == 0:
                 logger.info("No nodes retrieved, skipping reranking.")
            else:
                query_bundle_for_rerank = QueryBundle(query_str=query_text)
                
                if self.rag_config.reranker_type == "llm" and self.llm_reranker:
                    logger.info(f"Applying LLM Reranker to {len(retrieved_nodes)} nodes...")
                    final_retrieved_nodes = self.llm_reranker._postprocess_nodes(nodes=retrieved_nodes, query_bundle=query_bundle_for_rerank)
                elif self.rag_config.reranker_type == "local" and self.local_reranker:
                    logger.info(f"Applying Local Reranker to {len(retrieved_nodes)} nodes...")
                    final_retrieved_nodes = self.local_reranker.postprocess_nodes(retrieved_nodes, query_bundle=query_bundle_for_rerank)
                logger.info(f"Reranker returned {len(final_retrieved_nodes)} nodes.")
        
        # 无论是否重排，最终都确保返回 top_k 数量的节点
        final_top_k = top_k if top_k else self.rag_config.retrieval_top_k
        if len(final_retrieved_nodes) > final_top_k:
             final_retrieved_nodes = final_retrieved_nodes[:final_top_k]

        return final_retrieved_nodes
    
    def retrieve_documents_sync(self,
                                query_text: str,
                                collection_name: str,
                                filters: Optional[Dict[str, Any]] = None,
                                top_k: Optional[int] = None,
                                use_reranker: bool = True) -> List[NodeWithScore]:
        """
        通用的文档召回方法的同步版本。
        此方法用于在非异步上下文（如调试或同步API）中调用。
        """
        rag_index = self._indexer_service.get_rag_index(collection_name)
        if not rag_index:
            logger.error(f"RAG collection '{collection_name}' does not exist or could not be loaded.")
            return []

        chroma_where_clause = self._build_chroma_where_clause(filters)
        logger.info(f"Retrieving documents with ChromaDB `where` clause: {chroma_where_clause}")
        
        initial_retrieval_top_k = top_k * self.rag_config.initial_retrieval_multiplier if top_k else self.rag_config.retrieval_top_k * self.rag_config.initial_retrieval_multiplier

        retriever = rag_index.as_retriever(
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
            similarity_top_k=int(initial_retrieval_top_k)
        )
        # 调用同步版本的 retrieve()
        retrieved_nodes = retriever.retrieve(query_text)
        logger.info(f"Retrieved {len(retrieved_nodes)} chunks for query: '{query_text[:50]}...'")

        final_retrieved_nodes = retrieved_nodes
        
        if self.rag_config.use_reranker and use_reranker and (self.llm_reranker or self.local_reranker):
            if len(retrieved_nodes) == 0:
                 logger.info("No nodes retrieved, skipping reranking.")
            else:
                query_bundle_for_rerank = QueryBundle(query_str=query_text)
                
                if self.rag_config.reranker_type == "llm" and self.llm_reranker:
                    logger.info(f"Applying LLM Reranker to {len(retrieved_nodes)} nodes...")
                    # 调用 LLM Reranker 的同步方法
                    final_retrieved_nodes = self.llm_reranker._postprocess_nodes(nodes=retrieved_nodes, query_bundle=query_bundle_for_rerank)
                elif self.rag_config.reranker_type == "local" and self.local_reranker:
                    logger.info(f"Applying Local Reranker to {len(retrieved_nodes)} nodes...")
                    # 调用本地 Reranker 的同步方法
                    final_retrieved_nodes = self.local_reranker.postprocess_nodes(retrieved_nodes, query_bundle=query_bundle_for_rerank)
                logger.info(f"Reranker returned {len(final_retrieved_nodes)} nodes.")

        final_top_k = top_k if top_k else self.rag_config.retrieval_top_k
        if len(final_retrieved_nodes) > final_top_k:
             final_retrieved_nodes = final_retrieved_nodes[:final_top_k]

        return final_retrieved_nodes