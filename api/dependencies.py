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

from services.task_manager_service import TaskManagerService

logger = logging.getLogger(__name__)

# 全局存储服务实例，确保它们是单例
_indexer_service: IndexerService = None
_query_service: QueryService = None
_chat_history_service: ChatHistoryService = None
_ai_extraction_service: AIExtractionService = None
_document_oss_service: DocumentOssService = None 
_document_service: DocumentService = None
_oss_service: OssService = None 
_task_manager_service: TaskManagerService = None 

# 全局存储 LLM 实例，防止重复创建
_deepseek_llm_metadata: OpenAILike = None
_deepseek_llm_flashcard: OpenAILike = None

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

def get_dashscope_rag_llm() -> LlamaLLM: # 类型提示为 LlamaIndex 的 LLM
    """获取用于 RAG 的 DashScope LLM 依赖"""
    return DashScope(
        model_name=DashScopeGenerationModels.QWEN_PLUS,
        api_key=settings.DASHSCOPE_API_KEY,
        max_tokens=4096,
        temperature=0.5,
        similarity_cutoff=0.5
    )

def get_deepseek_llm_metadata() -> OpenAILike:
    """获取用于元数据提取的 DeepSeek LLM 依赖"""
    global _deepseek_llm_metadata
    if _deepseek_llm_metadata is None:
        _deepseek_llm_metadata = OpenAILike(
            model="deepseek-chat",
            api_base=settings.DEEPSEEK_API_BASE,
            api_key=settings.DEEPSEEK_API_KEY,
            temperature=0.0,
            kwargs={'response_format': {'type': 'json_object'}}
        )
    return _deepseek_llm_metadata

def get_deepseek_llm_flashcard() -> OpenAILike:
    """获取用于闪卡提取的 DeepSeek LLM 依赖"""
    global _deepseek_llm_flashcard
    if _deepseek_llm_flashcard is None:
        _deepseek_llm_flashcard = OpenAILike(
            model="deepseek-chat",
            api_base=settings.DEEPSEEK_API_BASE,
            api_key=settings.DEEPSEEK_API_KEY,
            temperature=0.3,
            kwargs={'response_format': {'type': 'json_object'}}
        )
    return _deepseek_llm_flashcard

def get_indexer_service() -> IndexerService:
    """提供 IndexerService 的单例实例"""
    global _indexer_service
    if _indexer_service is None:
        logger.info("Initializing IndexerService...")
        _indexer_service = IndexerService(
            chroma_client=get_chroma_client(),
            embedding_model=get_embedding_model()
        )
    return _indexer_service

def get_chat_history_service() -> ChatHistoryService:
    """提供 ChatHistoryService 的单例实例"""
    global _chat_history_service
    if _chat_history_service is None:
        logger.info("Initializing ChatHistoryService...")
        # ChatHistoryService 现在会通过构造函数接收 embedding_model
        _chat_history_service = ChatHistoryService(
            chroma_client=get_chroma_client(),
            embedding_model=get_embedding_model(),
            indexer_service=get_indexer_service()
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
            llm=get_dashscope_rag_llm(), # 传入 LLM
            indexer_service=get_indexer_service(), # 传入 indexer_service
            chat_history_service=get_chat_history_service() # 传入 chat_history_service 实例
        )
    return _query_service

def get_ai_extraction_service() -> AIExtractionService:
    """提供 AIExtractionService 的单例实例"""
    global _ai_extraction_service
    if _ai_extraction_service is None:
        logger.info("Initializing AIExtractionService...")
        _ai_extraction_service = AIExtractionService(
            llm_metadata_model=get_deepseek_llm_metadata(),
            llm_flashcard_model=get_deepseek_llm_flashcard(),
            oss_service_instance=get_oss_service() 
        )
    return _ai_extraction_service


def get_oss_service() -> OssService:
    """提供 OSSService 的单例实例"""
    global _oss_service
    if _oss_service is None:
        logger.info("Initializing OSSService...")
        _oss_service = OssService()
    return _oss_service

def get_task_manager_service() -> TaskManagerService:
    """提供 TaskManagerService 的单例实例"""
    global _task_manager_service
    if _task_manager_service is None:
        logger.info("Initializing TaskManagerService...")
        _task_manager_service = TaskManagerService()
    return _task_manager_service

def get_document_service() -> DocumentService: # <-- 新增 DocumentService 的依赖提供者
    """提供 DocumentService 的单例实例"""
    global _document_service
    if _document_service is None:
        logger.info("Initializing DocumentService...")
        _document_service = DocumentService() # <-- DocumentService 不依赖其他服务
    return _document_service

# ... (get_indexer_service, get_chat_history_service, get_query_service, get_ai_extraction_service 保持不变) ...

def get_document_oss_service() -> DocumentOssService:
    """提供 DocumentOssService 的单例实例"""
    global _document_oss_service
    if _document_oss_service is None:
        logger.info("Initializing DocumentOssService...")
        _document_oss_service = DocumentOssService(
            indexer_service=get_indexer_service(),
            oss_service_instance=get_oss_service(),
            task_manager_service=get_task_manager_service()
        )
    return _document_oss_service