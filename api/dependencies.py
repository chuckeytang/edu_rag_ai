# api/dependencies.py

import json
import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence
from http import HTTPStatus

from fastapi import Depends

# 核心配置和服务
from core.config import settings
from core.rag_config import RagConfig
from services.ai_extraction_service import AIExtractionService
from services.document_oss_service import DocumentOssService
from services.document_service import DocumentService
from services.indexer_service import IndexerService
from services.chat_history_service import ChatHistoryService
from services.query_service import QueryService
from services.mcp_service import MCPService
from services.oss_service import OssService
from services.task_manager_service import TaskManagerService

# 引入抽象接口和具体实现
from services.abstract_kb_service import AbstractKnowledgeBaseService
from services.volcano_rag_service import VolcanoEngineRagService
from services.bailian_rag_service import BailianRagService

# 保留 LLM 和 Embedding 依赖，但重新组织
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope as LlamaDashScope, DashScopeGenerationModels
from llama_index.core.llms import LLM as LlamaLLM, ChatMessage
from llama_index.llms.openai_like import OpenAILike
from services.llm.llm_service import DashScopeWithTools

# 保留 ChromaDB 依赖，仅用于聊天历史
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# 全局变量
_global_chroma_client_instance = None 
_current_rag_config = RagConfig.get_default_config()

def get_rag_config():
    """返回当前的RAG配置"""
    return _current_rag_config

def update_rag_config(new_config_json_str: str):
    """用于从后端更新 RAG 配置"""
    global _current_rag_config
    try:
        config_dict = json.loads(new_config_json_str)
        _current_rag_config = RagConfig(**config_dict)
        logger.info("RAG configuration updated successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to update RAG config: {e}")
        return False

# --- ChromaDB 和 Embedding 依赖 (仅用于聊天历史) ---
def get_chroma_client():
    """获取 ChromaDB 客户端的依赖"""
    global _global_chroma_client_instance
    if _global_chroma_client_instance is None:
        logger.warning("ChromaDB client requested but not yet globally initialized. Attempting lazy initialization.")
        initialize_global_chroma_client()
    return _global_chroma_client_instance

def initialize_global_chroma_client():
    """初始化全局 ChromaDB PersistentClient 实例"""
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
            raise

@lru_cache(maxsize=1)
def get_oss_service() -> OssService:
    """提供 OssService 的单例实例"""
    logger.info("Initializing OssService...")
    return OssService()

@lru_cache(maxsize=1)
def get_embedding_model():
    """获取 Embedding 模型的依赖，用于聊天历史"""
    logger.info("Initializing DashScopeEmbedding model...")
    return DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        text_type="document"
    )

# --- LLM 依赖 ---
@lru_cache(maxsize=1)
def get_dashscope_rag_llm() -> LlamaLLM:
    """获取用于 RAG 的 DashScope LLM 依赖"""
    logger.info("Initializing DashScope RAG LLM (QWEN_PLUS)...")
    return LlamaDashScope(
        model_name=DashScopeGenerationModels.QWEN_PLUS,
        api_key=settings.DASHSCOPE_API_KEY,
        max_tokens=4096,
        temperature=0.1,
        similarity_cutoff=0.4,
        streaming=True
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
def get_deepseek_llm_function_calling() -> OpenAILike:
    """获取用于 Function Calling 的 DeepSeek LLM 依赖"""
    logger.info("Initializing DeepSeek LLM for Function Calling (deepseek-chat)...")
    return OpenAILike(
        model="deepseek-chat",
        api_base=settings.DEEPSEEK_API_BASE,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.0,
        is_chat_model=True,
        is_function_calling_model=True,
    )

@lru_cache(maxsize=1)
def get_dashscope_llm_for_function_calling():
    """为MCP服务提供轻量级的通义千问LLM（QWEN_TURBO），并支持 Function Calling"""
    logger.info("Initializing DashScope LLM for Function Calling (QWEN_TURBO)...")
    return DashScopeWithTools( 
        model_name=DashScopeGenerationModels.QWEN_TURBO,
        api_key=settings.DASHSCOPE_API_KEY,
        temperature=0.1,
        streaming=False
    )

@lru_cache(maxsize=1)
def get_ai_extraction_service(
    oss_service: OssService = Depends(get_oss_service) 
) -> AIExtractionService:
    """提供 AIExtractionService 的单例实例"""
    logger.info("Initializing AIExtractionService...")
    return AIExtractionService(
        llm_metadata_model=get_deepseek_llm_metadata(),
        llm_flashcard_model=get_deepseek_llm_flashcard(),
        oss_service_instance=oss_service 
    )

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
# --- 知识库服务抽象工厂 (核心改造) ---

@lru_cache(maxsize=1)
def get_abstract_kb_service() -> AbstractKnowledgeBaseService:
    """
    根据配置实例化并返回具体的知识库服务（Volcano 或 Bailian），
    所有业务层都将依赖这个抽象接口。
    """
    # 假设 settings 中有一个 KB_SERVICE_VENDOR 字段，例如 'VOLCANO' 或 'BAILIAN'
    vendor = getattr(settings, 'KB_SERVICE_VENDOR', 'BAILIAN').upper()
    logger.info(f"Initializing Knowledge Base Service for vendor: {vendor}")

    if vendor == 'BAILIAN':
        return BailianRagService()
    elif vendor == 'VOLCANO':
        # VolcanoEngineRagService 实例保持原样，但作为抽象接口返回
        return VolcanoEngineRagService()
    else:
        raise ValueError(f"Unknown KB_SERVICE_VENDOR: {vendor}. Must be 'VOLCANO' or 'BAILIAN'.")

@lru_cache(maxsize=1)
def get_indexer_service() -> IndexerService:
    """提供 IndexerService 的单例实例，注入抽象 KB Service"""
    logger.info("Initializing IndexerService...")
    rag_config = get_rag_config()
    # 注入抽象知识库服务
    return IndexerService(
        rag_config=rag_config,
        kb_service=get_abstract_kb_service() # 更改：使用抽象接口
    )

@lru_cache(maxsize=1)
def get_chat_history_service() -> ChatHistoryService:
    """提供 ChatHistoryService 的单例实例"""
    logger.info("Initializing ChatHistoryService...")
    # 注入 chroma_client 和 embedding_model
    return ChatHistoryService(
        chroma_client=get_chroma_client(),
        embedding_model=get_embedding_model(),
    )

@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    """提供 QueryService 的单例实例"""
    logger.info("Initializing QueryService...")
    rag_config = get_rag_config()
    return QueryService(
        llm=get_dashscope_rag_llm(),
        embedding_model=get_embedding_model(),
        indexer_service=get_indexer_service(),
        chat_history_service=get_chat_history_service(),
        kb_service=get_abstract_kb_service(),
        rag_config=rag_config, 
    )

@lru_cache(maxsize=1)
def get_ai_extraction_service(
    oss_service: OssService = Depends(get_oss_service) 
) -> AIExtractionService:
    """提供 AIExtractionService 的单例实例"""
    logger.info("Initializing AIExtractionService...")
    return AIExtractionService(
        llm_metadata_model=get_deepseek_llm_metadata(),
        llm_flashcard_model=get_deepseek_llm_flashcard(),
        # 正确传递 oss_service 实例
        oss_service_instance=oss_service 
    )

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

@lru_cache(maxsize=1)
def get_mcp_service() -> MCPService:
    """提供 MCPService 的单例实例，用于处理功能调用"""
    logger.info("Initializing MCPService...")
    return MCPService(
        llm_for_function_calling=get_dashscope_llm_for_function_calling(),
        kb_service=get_abstract_kb_service() # 更改：使用抽象接口
    )