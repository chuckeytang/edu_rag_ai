import asyncio
import time
import json
import logging
import os
from typing import Any, Dict, List, Optional

from core.config import settings
from core.rag_config import RagConfig
from services.abstract_kb_service import AbstractKnowledgeBaseService

logger = logging.getLogger(__name__)

# 不再需要本地索引路径，但为了保留兼容性，可以暂时保留
PERSIST_DIR = settings.INDEX_PATH

class IndexerService:
    def __init__(self, 
                 kb_service: AbstractKnowledgeBaseService): 
        
        self.indices: Dict[str, Any] = {} 
        # 使用通用名称 self.kb_service
        self.kb_service = kb_service 

        logger.info("IndexerService initialized with Volcano Engine RAG service.")

    async def add_documents_to_index(self, 
                               documents: List[Dict[str, Any]], 
                               knowledge_base_id: str) -> Dict[str, Any]:
        """向知识库中添加文档。"""
        if not documents:
            logger.warning("No documents provided for indexing. Skipping.")
            return {"status": "success", "message": "No documents to add."}

        document_to_add = documents[0]
        url = document_to_add.get("url")
        text_content = document_to_add.get("text_content")

        doc_name = document_to_add.get("doc_name")
        doc_id = document_to_add.get("doc_id")
        doc_type = document_to_add.get("doc_type")
        meta = document_to_add.get("meta")
        
        if not url and not text_content:
            logger.error("Document dictionary is missing 'url'. Cannot index.")
            return {"status": "error", "message": "Document dictionary is missing 'url'"}
        
        indexing_metadata = {
            "url_present": bool(url), "text_content_present": bool(text_content), 
            "doc_id": doc_id, "doc_name": doc_name, "doc_type": doc_type, 
            "knowledge_base_id": knowledge_base_id, "meta": meta
        }
        logger.info(f"Preparing to index document with the following metadata: {indexing_metadata}")
        
        try:
            result = None
            if url:
                # --- 路径 1: URL 导入 (PDF/DOCX 等) ---
                logger.info(f"Using URL import path for document: {doc_name}")
                result = await self.kb_service.import_document_url(
                    url=url, 
                    doc_id=doc_id, 
                    doc_name=doc_name, 
                    doc_type=doc_type,
                    knowledge_base_id=knowledge_base_id, 
                    meta=meta
                )
            elif text_content:
                # --- 路径 2: TEXT 导入 (PaperCut) ---
                logger.info(f"Using TEXT content import path for document: {doc_name}")
                # 📢 调用抽象接口的新方法
                result = await self.kb_service.import_document_text( 
                    text_content=text_content, # 传入文本内容
                    doc_id=doc_id, 
                    doc_name=doc_name, 
                    doc_type=doc_type,
                    knowledge_base_id=knowledge_base_id, 
                    meta=meta
                )
            else:
                raise ValueError("Document dictionary is missing both 'url' and 'text_content'. Cannot index.")

            logger.info(f"Document '{doc_name}' uploaded successfully. Response: {result}")
            # 注意：百炼返回的是 file_id，我们统一使用 kb_doc_id
            return {"status": "success", "result": result, "kb_doc_id": result.get('kb_doc_id')} 
            
        except Exception as e:
            logger.error(f"Failed to upload document '{doc_name}' from URL: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "doc_name": doc_name}

    def delete_nodes_by_metadata(self, knowledge_base_id: str, filters: dict) -> dict:
        """根据元数据过滤器删除文档。"""
        
        logger.info(f"Attempting to delete documents from '{knowledge_base_id}' with filters: {filters}")
        
        doc_id_to_delete = filters.get("kb_doc_id") 
        if not doc_id_to_delete:
            message = "Deletion requires a 'kb_doc_id' filter (should be FileId)."
            logger.error(message)
            return {"status": "error", "message": message}
            
        try:
            # 调用抽象接口
            response = self.kb_service.delete_document(
                knowledge_base_id=knowledge_base_id,
                doc_id=doc_id_to_delete
            )
            message = f"Successfully submitted deletion for file '{doc_id_to_delete}'. Response: {response}"
            logger.info(message)
            return {"status": "success", "message": message, "response": response}
            
        except Exception as e:
            message = f"An error occurred during deletion: {e}"
            logger.error(message, exc_info=True)
            return {"status": "error", "message": message}
             
    def update_existing_nodes_metadata(self, knowledge_base_id: str, doc_id: str, metadata_update_payload: dict) -> dict:
        """更新现有文档的元数据。此方法仅作为警告/转发。"""
        # 注意：这里移除了硬编码的 knowledge_base_id
        
        logger.info(f"Performing metadata update for doc_id: {doc_id} in KB '{knowledge_base_id}' with payload: {metadata_update_payload}")
        
        if not doc_id or not metadata_update_payload:
            message = "Update requires a 'doc_id' and metadata payload."
            logger.warning(message)
            return {"status": "error", "message": message}
            
        try:
            # 此处调用 update_document_meta，由底层服务决定是否支持
            response = asyncio.run(self.kb_service.update_document_meta(
                knowledge_base_id=knowledge_base_id,
                doc_id=doc_id,
                meta_updates=[metadata_update_payload] 
            ))
            
            # 由于此方法是同步的，需要使用 asyncio.run 包装异步调用，但在实际应用中，此方法也应该改为异步
            return response
            
        except Exception as e:
            message = f"An error occurred during metadata update: {e}"
            logger.error(message, exc_info=True)
            return {"status": "error", "message": message}
           
    async def update_document_meta(self, 
                             doc_id: str, 
                             file_key: str,
                             knowledge_base_id: str, 
                             meta_updates: List[Dict[str, Any]]) -> dict:
        """通用元数据更新方法，直接调用 AbstractKnowledgeBaseService。"""
        try:
            logger.info(f"Preparing to update meta for doc {doc_id} in KB {knowledge_base_id}. Meta Updates: {meta_updates}")
            
            # 调用抽象接口
            response = await self.kb_service.update_document_meta(
                knowledge_base_id=knowledge_base_id,
                file_key=file_key,
                doc_id=doc_id,
                meta_updates=meta_updates
            )
            
            if response.get("status") == "not_implemented":
                return response
            
            return {
                "message": f"Successfully updated meta for doc {doc_id}.",
                "status": "success",
                "doc_id": doc_id
            }
                
        except Exception as e:
            logger.error(f"Failed to update document meta for doc {doc_id}: {e}", exc_info=True)
            return {
                "message": f"An API error occurred: {str(e)}",
                "status": "error",
            }
