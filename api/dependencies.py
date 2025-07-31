# api/dependencies.py
from services.ai_extraction_service import AIExtractionService
from services.document_oss_service import DocumentOssService
from services.document_service import DocumentService
from services.indexer_service import IndexerService
from services.oss_service import OssService
from services.query_service import QueryService
from services.chat_history_service import ChatHistoryService
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from core.config import settings
import chromadb
from chromadb.config import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import LLM as LlamaLLM
import logging
from functools import lru_cache

from services.task_manager_service import TaskManagerService

logger = logging.getLogger(__name__)
_global_chroma_client_instance = None 

def get_chroma_client():
    """
    获取 ChromaDB 客户端的依赖。
    现在它通过 _global_chroma_client_instance 管理单例，而不是 lru_cache。
    初始化应该在 FastAPI 应用的 startup 事件中调用 `initialize_global_chroma_client()`。
    """
    if _global_chroma_client_instance is None:
        # 如果在没有通过 initialize_global_chroma_client 预先初始化的情况下被调用
        # 这可能意味着依赖链有问题或在非预期场景下被触发
        logger.warning("ChromaDB client requested but not yet globally initialized. Attempting lazy initialization. "
                       "Consider using FastAPI's startup event for eager initialization.")
        # fallback to lazy initialization, but eager is preferred
        initialize_global_chroma_client()
    return _global_chroma_client_instance

def initialize_global_chroma_client():
    """
    负责初始化全局的 ChromaDB PersistentClient 实例。
    此函数应在应用启动时被调用且仅调用一次。
    """
    global _global_chroma_client_instance
    if _global_chroma_client_instance is None:
        logger.info("Starting global ChromaDB PersistentClient initialization...")
        try:
            _global_chroma_client_instance = chromadb.PersistentClient(
                path=settings.CHROMA_PATH,
                settings=Settings(is_persistent=True)
            )
            logger.info("Global ChromaDB PersistentClient initialized successfully.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to initialize global ChromaDB PersistentClient: {e}", exc_info=True)
            # 在启动时失败，应该阻止应用继续运行
            raise
    else:
        logger.info("Global ChromaDB PersistentClient already initialized.")


@lru_cache(maxsize=1)
def get_embedding_model():
    """获取 Embedding 模型的依赖"""
    logger.info("Initializing DashScopeEmbedding model...")
    return DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        text_type="document"
    )

@lru_cache(maxsize=1)
def get_dashscope_rag_llm() -> LlamaLLM: # 类型提示为 LlamaIndex 的 LLM
    """获取用于 RAG 的 DashScope LLM 依赖"""
    logger.info("Initializing DashScope RAG LLM (QWEN_PLUS)...")
    return DashScope(
        model_name=DashScopeGenerationModels.QWEN_PLUS,
        api_key=settings.DASHSCOPE_API_KEY,
        max_tokens=4096,
        temperature=0.1,
        similarity_cutoff=0.4
    )

@lru_cache(maxsize=1)
def get_deepseek_llm_metadata() -> OpenAILike:
    """获取用于元数据提取的 DeepSeek LLM 依赖"""
    logger.info("Initializing DeepSeek LLM for metadata extraction (deepseek-chat)...")
    return OpenAILike(
        model="deepseek-chat",
        api_base=settings.DEEPSEEK_API_BASE,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.0,
        kwargs={'response_format': {'type': 'json_object'}}
    )

@lru_cache(maxsize=1)
def get_deepseek_llm_flashcard() -> OpenAILike:
    """获取用于闪卡提取的 DeepSeek LLM 依赖"""
    logger.info("Initializing DeepSeek LLM for flashcard extraction (deepseek-chat)...")
    return OpenAILike(
        model="deepseek-chat",
        api_base=settings.DEEPSEEK_API_BASE,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.3,
        kwargs={'response_format': {'type': 'json_object'}}
    )

@lru_cache(maxsize=1)
def get_indexer_service() -> IndexerService:
    """提供 IndexerService 的单例实例"""
    logger.info("Initializing IndexerService...")
    # IndexerService 依赖于 get_chroma_client()，它会获取全局单例
    return IndexerService(
        chroma_client=get_chroma_client(), 
        embedding_model=get_embedding_model()
    )

@lru_cache(maxsize=1)
def get_chat_history_service() -> ChatHistoryService:
    """提供 ChatHistoryService 的单例实例"""
    logger.info("Initializing ChatHistoryService...")
    return ChatHistoryService(
        chroma_client=get_chroma_client(),
        embedding_model=get_embedding_model(),
        indexer_service=get_indexer_service()
    )

@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    """提供 QueryService 的单例实例"""
    logger.info("Initializing QueryService...")
    return QueryService(
        chroma_client=get_chroma_client(),
        embedding_model=get_embedding_model(),
        llm=get_dashscope_rag_llm(),
        indexer_service=get_indexer_service(),
        chat_history_service=get_chat_history_service(),
        deepseek_llm_for_reranker=get_deepseek_llm_metadata()
    )

@lru_cache(maxsize=1)
def get_ai_extraction_service() -> AIExtractionService:
    """提供 AIExtractionService 的单例实例"""
    logger.info("Initializing AIExtractionService...")
    return AIExtractionService(
        llm_metadata_model=get_deepseek_llm_metadata(),
        llm_flashcard_model=get_deepseek_llm_flashcard(),
        oss_service_instance=get_oss_service()
    )

@lru_cache(maxsize=1)
def get_oss_service() -> OssService:
    """提供 OSSService 的单例实例"""
    logger.info("Initializing OssService...")
    return OssService()

@lru_cache(maxsize=1)
def get_task_manager_service() -> TaskManagerService:
    """提供 TaskManagerService 的单例实例"""
    logger.info("Initializing TaskManagerService...")
    return TaskManagerService()

@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    """提供 DocumentService 的单例实例"""
    logger.info("Initializing DocumentService...")
    return DocumentService()

@lru_cache(maxsize=1)
def get_document_oss_service() -> DocumentOssService:
    """提供 DocumentOssService 的单例实例"""
    logger.info("Initializing DocumentOssService...")
    return DocumentOssService(
        indexer_service=get_indexer_service(),
        oss_service_instance=get_oss_service(),
        task_manager_service=get_task_manager_service()
    )