import json
import logging
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings
from llama_cloud import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument, TextNode, NodeWithScore, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.core import StorageContext

from core.config import settings
from core.metadata_utils import prepare_metadata_for_storage
from core.rag_config import RagConfig

logger = logging.getLogger(__name__)

class ChatHistoryService:
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embedding_model: Optional[BaseEmbedding]):
        # 移除对 indexer_service 和 retrieval_service 的依赖
        self.chroma_path = settings.CHROMA_PATH
        self.chroma_client = chroma_client
        self._embedding_model = embedding_model 
        self.chat_history_collection_name = "chat_history_collection"
        CHROMA_CHAT_HISTORY_CHUNK_SIZE = 4096

        self.node_parser = SentenceSplitter(
            chunk_size=CHROMA_CHAT_HISTORY_CHUNK_SIZE,
            chunk_overlap=0 
        )
        
        # 直接在 __init__ 中准备索引，而不是依赖 IndexerService
        self._vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_client.get_or_create_collection(name=self.chat_history_collection_name)
        )
        self._index = self._get_chat_history_index()
        
        logger.info("ChatHistoryService initialized and ready.")

    def _get_chat_history_index(self) -> VectorStoreIndex:
        """
        创建一个专门用于聊天历史的 VectorStoreIndex 实例。
        这个索引是轻量级的，只在内存中操作，不进行持久化。
        """
        try:
            # 手动创建空的存储组件，因为聊天历史索引不需要持久化
            storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store,
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore(),
            )
            
            # 从现有的 VectorStore 重建索引对象
            index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                embed_model=self._embedding_model,
                storage_context=storage_context,
                transformations=[self.node_parser]
            )
            logger.info(f"Chat history index object loaded/reconstructed from ChromaDB collection '{self.chat_history_collection_name}'.")
            return index
        except Exception as e:
            logger.error(f"Failed to create VectorStoreIndex for chat history: {e}", exc_info=True)
            raise

    def add_chat_message_to_chroma(self, message_data: Dict[str, Any], rag_config: Optional[RagConfig] = None):
        """
        将聊天消息作为文档添加到 ChromaDB 的聊天历史Collection。
        """
        doc_id = str(message_data.get("id"))
        if not doc_id:
            doc_id = f"chat_msg_{message_data['session_id']}_{message_data['timestamp']}_{message_data['role']}"

        metadata = message_data.get("metadata")
        if metadata is None:
            metadata = {}
            
        metadata["session_id"] = message_data.get("session_id")
        metadata["account_id"] = message_data.get("account_id")
        metadata["role"] = message_data.get("role")
        metadata["timestamp"] = message_data.get("timestamp")

        doc = LlamaDocument(
            text=message_data["content"],
            id_=doc_id,
            metadata=prepare_metadata_for_storage(metadata)
        )
            
        try:
            # 直接使用内部的 index 对象来插入节点，不再通过 IndexerService
            self._index.insert(doc)
            logger.info(f"Successfully added chat message '{doc.id_}' to ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to add chat message '{doc.id_}' to ChromaDB: {e}", exc_info=True)
            # 抛出异常以确保调用方知道失败
            raise

    def delete_chat_messages(self, session_id: str, account_id: int) -> Dict[str, Any]:
        """
        根据 session_id 和 account_id 删除 ChromaDB 中的所有聊天消息。
        直接调用 ChromaDB 客户端进行删除。
        """
        filters = {
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"account_id": {"$eq": account_id}}
            ]
        }
        logger.info(f"Attempting to delete chat messages from ChromaDB with filters: {filters}")
        try:
            # 直接使用 ChromaDB 客户端的 delete 方法
            self._vector_store.delete(where_metadata=filters)
            message = f"Successfully deleted chat messages for session {session_id}."
            logger.info(message)
            return {"status": "success", "message": message}
        except Exception as e:
            logger.error(f"Failed to delete chat messages from ChromaDB for session {session_id}: {e}", exc_info=True)
            raise

    async def retrieve_chat_history_context(self, session_id: str, account_id: int, query_text: str, top_k: int = 5) -> List[TextNode]:
        """
        从聊天历史Collection中语义检索相关上下文。
        直接使用内部的索引对象进行检索。
        """
        try:
            chat_context_filters = {
                "$and": [
                    {"session_id": {"$eq": session_id}},
                    {"account_id": {"$eq": account_id}}
                ]
            }
            logger.info(f"Retrieving chat history context with filters: {chat_context_filters}")
            
            # 创建一个检索器，并使用 ChromaDB 的 where 参数来应用过滤器
            retriever = self._index.as_retriever(
                similarity_top_k=top_k,
                vector_store_kwargs={"where": chat_context_filters}
            )
            
            # 执行异步检索
            retrieved_nodes_with_score: List[NodeWithScore] = await retriever.aretrieve(query_text)
            
            # 对检索到的节点的元数据进行反序列化
            deserialized_nodes = []
            for node_with_score in retrieved_nodes_with_score:
                node = node_with_score.node
                metadata = node.metadata
                
                # 检查并反序列化可能包含 JSON 字符串的字段
                for key, value in metadata.items():
                    if isinstance(value, str):
                        try:
                            # 尝试解析 JSON 字符串
                            deserialized_value = json.loads(value)
                            # 仅当解析成功且不是简单的字符串时才替换
                            if isinstance(deserialized_value, (list, dict)):
                                metadata[key] = deserialized_value
                        except (json.JSONDecodeError, TypeError):
                            # 如果不是有效的 JSON，则不做处理
                            pass
                
                deserialized_nodes.append(node)

            logger.info(f"Retrieved and deserialized {len(deserialized_nodes)} relevant chat history nodes.")
            return deserialized_nodes

        except Exception as e:
            logger.error(f"Failed to retrieve chat history context: {e}", exc_info=True)
            return []

    def get_session_message_count(self, session_id: str, account_id: int) -> int:
        """
        获取指定会话ID和用户ID下的聊天消息总数。
        这会直接查询ChromaDB。
        """
        try:
            collection = self.chroma_client.get_collection(name=self.chat_history_collection_name)
            results = collection.get(
                where={
                    "$and": [
                        {"session_id": {"$eq": session_id}},
                        {"account_id": {"$eq": account_id}}
                    ]
                },
                include=[]
            )
            count = len(results.get('ids', []))
            logger.debug(f"Retrieved {count} chat messages for session {session_id}, account {account_id}.")
            return count
        except Exception as e:
            logger.error(f"Failed to get chat message count for session {session_id}: {e}", exc_info=True)
            return 0