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
from services.indexer_service import IndexerService # 假设你的这个工具函数存在

logger = logging.getLogger(__name__)

class ChatHistoryService:
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embedding_model: Optional[BaseEmbedding],
                 indexer_service: IndexerService):
        self.chroma_path = settings.CHROMA_PATH
        self.chroma_client = chroma_client
        self._embedding_model = embedding_model 
        self.chat_history_collection_name = "chat_history_collection"
        self._indexer_service = indexer_service
        self._initialize_chat_history_collection()

        logger.info("ChatHistoryService initialized with provided embedding model.")

    def _initialize_chat_history_collection(self):
        """确保聊天历史的 ChromaDB Collection 存在"""
        try:
            self.chroma_client.get_or_create_collection(name=self.chat_history_collection_name)
            logger.info(f"ChromaDB chat history collection '{self.chat_history_collection_name}' ensured.")
        except Exception as e:
            logger.error(f"Failed to initialize chat history collection: {e}")
            raise

    def add_chat_message_to_chroma(self, message_data: Dict[str, Any]):
        """
        将聊天消息作为文档添加到 ChromaDB 的聊天历史Collection。
        """
        collection = self.chroma_client.get_collection(name=self.chat_history_collection_name)
        
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
        
        try:
            self._indexer_service.add_documents_to_index(
                documents=[doc],
                collection_name=self.chat_history_collection_name
            )
            logger.info(f"Successfully added chat message '{doc.id_}' to ChromaDB chat history collection via IndexerService.")
        except Exception as e:
            logger.error(f"Failed to add chat message '{doc.id_}' to ChromaDB via IndexerService: {e}", exc_info=True)

    def retrieve_chat_history_context(self, session_id: str, account_id: int, query_text: str, top_k: int = 5) -> List[TextNode]:
        """
        从聊天历史Collection中语义检索相关上下文。
        """
        try:
            chat_history_index = self._indexer_service._get_or_load_index(self.chat_history_collection_name)
            if not chat_history_index:
                logger.error(f"Chat history index for collection '{self.chat_history_collection_name}' could not be loaded.")
                return [] # 无法加载索引，返回空列表

            chat_context_filters = {
                "$and": [
                    {"session_id": {"$eq": session_id}},
                    {"account_id": {"$eq": account_id}}
                ]
            }
            logger.info(f"Retrieving chat history context with filters: {chat_context_filters}")
            
            chat_retriever = chat_history_index.as_retriever(
                vector_store_kwargs={"where": chat_context_filters},
                similarity_top_k=top_k
            )
            chat_history_context_nodes = chat_retriever.retrieve(query_text)
            logger.info(f"Retrieved {len(chat_history_context_nodes)} relevant chat history nodes.")
            return chat_history_context_nodes

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