# api/dependencies.py
from services.query_service import QueryService
from services.chat_history_service import ChatHistoryService
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from core.config import settings
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)

# 全局存储服务实例，确保它们是单例
_query_service: QueryService = None
_chat_history_service: ChatHistoryService = None

def get_chroma_client():
    """获取 ChromaDB 客户端的依赖"""
    return chromadb.PersistentClient(
        path=settings.CHROMA_PATH,
        settings=Settings(is_persistent=True)
    )

def get_embedding_model():
    """获取 Embedding 模型的依赖"""
    return DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        text_type="document"
    )

def get_llm():
    """获取 LLM 的依赖"""
    return DashScope(
        model_name=DashScopeGenerationModels.QWEN_PLUS,
        api_key=settings.DASHSCOPE_API_KEY,
        max_tokens=4096,
        temperature=0.5,
        similarity_cutoff=0.5
    )

def get_chat_history_service() -> ChatHistoryService:
    """提供 ChatHistoryService 的单例实例"""
    global _chat_history_service
    if _chat_history_service is None:
        logger.info("Initializing ChatHistoryService...")
        # ChatHistoryService 现在会通过构造函数接收 embedding_model
        _chat_history_service = ChatHistoryService(
            chroma_client=get_chroma_client(), # 传入 Chroma 客户端
            embedding_model=get_embedding_model() # 传入 embedding model
        )
    return _chat_history_service

def get_query_service() -> QueryService:
    """提供 QueryService 的单例实例"""
    global _query_service
    if _query_service is None:
        logger.info("Initializing QueryService...")
        _query_service = QueryService(
            chroma_client=get_chroma_client(), # 传入 Chroma 客户端
            embedding_model=get_embedding_model(), # 传入 embedding model
            llm=get_llm(), # 传入 LLM
            chat_history_service=get_chat_history_service() # 传入 chat_history_service 实例
        )
    return _query_service
