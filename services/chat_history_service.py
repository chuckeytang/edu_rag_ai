# api/services/chat_history_service.py
import logging
from typing import Dict, Any, List, Optional
from llama_index.core.schema import Document as LlamaDocument, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding

import chromadb

from core.config import settings
from core.metadata_utils import prepare_metadata_for_storage # 假设你的这个工具函数存在

logger = logging.getLogger(__name__)

class ChatHistoryService:
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embedding_model: Optional[BaseEmbedding] = None):
        self.chroma_path = settings.CHROMA_PATH
        self.chroma_client = chroma_client
        self._embedding_model = embedding_model 
        self.chat_history_collection_name = "chat_history_collection"
        self._initialize_chat_history_collection()

        logger.info("ChatHistoryService initialized with provided embedding model.")

        # 如果传入了 embedding_model，则使用它；否则，暂时设为 None
        # 它会在需要时（如在 FastAPI 依赖注入时）被设置
        if self._embedding_model is None:
             logger.error("Embedding model not initialized in QueryService. Please check.")
             # Fallback or raise error
             raise Exception("Embedding model not available in QueryService.")

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
            # 使用 ChromaDB 原生方法更直接，LlamaIndex 的 insert 在此处可能不必要
            # ChromaDB 的 add 会自动处理重复 id (覆盖)
            collection.add(
                documents=[doc.text],
                metadatas=[doc.metadata],
                ids=[doc.id_]
            )
            logger.info(f"Successfully added chat message '{doc.id_}' to ChromaDB chat history collection.")
        except Exception as e:
            logger.error(f"Failed to add chat message '{doc.id_}' to ChromaDB: {e}", exc_info=True)
            # 根据需求决定是否重新抛出异常

    def retrieve_chat_history_context(self, session_id: str, account_id: int, query_text: str, top_k: int = 5) -> List[TextNode]:
        """
        从聊天历史Collection中语义检索相关上下文。
        """
        try:
            chat_history_collection = self.chroma_client.get_collection(name=self.chat_history_collection_name)
            chat_history_vector_store = ChromaVectorStore(chroma_collection=chat_history_collection)
            chat_history_index = VectorStoreIndex.from_vector_store(
                vector_store=chat_history_vector_store,
                embed_model=self._embedding_model
            )

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

# chat_history_service = ChatHistoryService()