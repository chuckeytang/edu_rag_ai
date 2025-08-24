# api/services/chat_history_service.py
import logging
from typing import Dict, Any, List, Optional
from llama_index.core.schema import Document as LlamaDocument, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding

import chromadb

from core.config import settings
from core.metadata_utils import prepare_metadata_for_storage
from core.rag_config import RagConfig
from services.indexer_service import IndexerService
from services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

class ChatHistoryService:
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embedding_model: Optional[BaseEmbedding],
                 indexer_service: IndexerService, 
                 retrieval_service: RetrievalService):
        self.chroma_path = settings.CHROMA_PATH
        self.chroma_client = chroma_client
        self._embedding_model = embedding_model 
        self.chat_history_collection_name = "chat_history_collection"
        self._indexer_service = indexer_service
        self.retrieval_service = retrieval_service
        self._collection = self.chroma_client.get_or_create_collection(name=self.chat_history_collection_name)
        self._initialize_chat_history_collection()

    def _initialize_chat_history_collection(self):
        """确保聊天历史的 ChromaDB Collection 存在"""
        try:
            self.chroma_client.get_or_create_collection(name=self.chat_history_collection_name)
        except Exception as e:
            logger.error(f"Failed to initialize chat history collection: {e}")
            raise

    def add_chat_message_to_chroma(self, message_data: Dict[str, Any], rag_config: Optional[RagConfig] = None):
        """
        将聊天消息作为文档添加到 ChromaDB 的聊天历史Collection。
        """
        doc_id = str(message_data.get("id"))
        if not doc_id:
            # Fallback for ID, though Spring should provide a unique one
            doc_id = f"chat_msg_{message_data['session_id']}_{message_data['timestamp']}_{message_data['role']}"

        metadata = message_data.get("metadata", {})
        metadata["session_id"] = message_data.get("session_id")
        metadata["account_id"] = message_data.get("account_id")
        metadata["role"] = message_data.get("role")
        metadata["timestamp"] = message_data.get("timestamp")

        # 使用 LlamaIndex Document 方便管理
        doc = LlamaDocument(
            text=message_data["content"],
            id_=doc_id,
            metadata=prepare_metadata_for_storage(metadata)
        )

        # 1. 检查传入的 rag_config 是否为 None 或字典类型
        if rag_config and isinstance(rag_config, dict):
            try:
                # 2. 从字典反序列化为 RagConfig 实例
                # RagConfig 的构造函数会检查并赋值，从而让它变成一个真正的对象
                rag_config_obj = RagConfig(**rag_config)
                logger.info("Successfully converted rag_config dict to RagConfig object.")
            except Exception as e:
                # 如果反序列化失败，记录错误并回退到 None
                logger.error(f"Failed to convert rag_config dict to RagConfig object: {e}", exc_info=True)
                rag_config_obj = None
        else:
            # 3. 如果 rag_config 已经是正确的对象实例（或为 None），则直接使用
            rag_config_obj = rag_config
            
        try:
            self._indexer_service.add_documents_to_index(
                documents=[doc],
                collection_name=self.chat_history_collection_name,
                rag_config=rag_config_obj
            )
        except Exception as e:
            logger.error(f"Failed to add chat message '{doc.id_}' to ChromaDB via IndexerService: {e}", exc_info=True)

    def delete_chat_messages(self, session_id: str, account_id: int) -> Dict[str, Any]:
        """
        根据 session_id 和 account_id 删除 ChromaDB 中的所有聊天消息。
        """
        filters = {
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"account_id": {"$eq": account_id}}
            ]
        }
        logger.info(f"Attempting to delete chat messages from ChromaDB with filters: {filters}")
        try:
            # Re-using indexer_service for deletion by metadata
            result = self._indexer_service.delete_nodes_by_metadata(
                collection_name=self.chat_history_collection_name,
                filters=filters
            )
            logger.info(f"ChromaDB deletion result for session {session_id}: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete chat messages from ChromaDB for session {session_id}: {e}", exc_info=True)
            raise

    async def retrieve_chat_history_context(self, session_id: str, account_id: int, query_text: str, top_k: int = 5) -> List[TextNode]:
        """
        从聊天历史Collection中语义检索相关上下文。
        """
        try:
            chat_context_filters = {
                "$and": [
                    {"session_id": {"$eq": session_id}},
                    {"account_id": {"$eq": account_id}}
                ]
            }
            logger.info(f"Retrieving chat history context with filters: {chat_context_filters}")
            
            # 调用 QueryService 的通用召回方法，并禁用重排器（聊天历史通常不需要）
            retrieved_nodes = await self.retrieval_service.retrieve_documents(
                query_text=query_text,
                collection_name=self.chat_history_collection_name,
                filters=chat_context_filters,
                top_k=top_k,
                use_reranker=False # 聊天历史召回不需要重排，因为它本身就是时序的
            )
            
            # 将 LlamaIndex 的 NodeWithScore 对象转换为 TextNode
            text_nodes = [node_with_score.node for node_with_score in retrieved_nodes]
            logger.info(f"Retrieved {len(text_nodes)} relevant chat history nodes.")
            return text_nodes

        except Exception as e:
            logger.error(f"Failed to retrieve chat history context: {e}", exc_info=True)
            return []

    def get_session_message_count(self, session_id: str, account_id: int) -> int:
        """
        获取指定会话ID和用户ID下的聊天消息总数。
        这会直接查询ChromaDB，不进行语义检索。
        """
        try:
            # 使用 where 过滤 session_id 和 account_id
            # include=[] 表示我们只关心数量，不需要返回实际的数据
            results = self._collection.get(
                where={
                    "$and": [
                        {"session_id": {"$eq": session_id}},
                        {"account_id": {"$eq": account_id}}
                    ]
                },
                include=[] # 只获取ID，用于计数
            )
            count = len(results.get('ids', []))
            logger.debug(f"Retrieved {count} chat messages for session {session_id}, account {account_id}.")
            return count
        except Exception as e:
            logger.error(f"Failed to get chat message count for session {session_id}: {e}", exc_info=True)
            return 0 # 发生错误时返回0，确保不阻塞主流程
        
