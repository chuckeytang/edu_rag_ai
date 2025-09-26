import time
import json
import logging
import os
from typing import Any, Dict, List, Optional

from core.config import settings
from core.rag_config import RagConfig
from services.volcano_rag_service import VolcanoEngineRagService # 新增：导入火山引擎服务

logger = logging.getLogger(__name__)

# 不再需要本地索引路径，但为了保留兼容性，可以暂时保留
PERSIST_DIR = settings.INDEX_PATH

class IndexerService:
    def __init__(self, 
                 rag_config: RagConfig,
                 volcano_rag_service: VolcanoEngineRagService): # 新增：注入火山引擎服务
        
        self._current_rag_config = rag_config
        self.indices: Dict[str, Any] = {} # 内存缓存不再存储 VectorStoreIndex 对象
        self.volcano_rag_service = volcano_rag_service # 存储火山引擎服务实例

        logger.info("IndexerService initialized with Volcano Engine RAG service.")

    async def add_documents_to_index(self, 
                               documents: List[Dict[str, Any]], 
                               knowledge_base_id: str) -> Dict[str, Any]:
        """
        向火山引擎知识库中添加文档。
        
        Args:
            documents: 包含文档元数据的字典列表。
                       我们只处理列表中的第一个文档，因为它代表了原始文件。
            knowledge_base_id: 火山引擎知识库 ID。
            
        Returns:
            火山引擎 API 的响应。
        """
        if not documents:
            logger.warning("No documents provided for indexing. Skipping.")
            return {"status": "success", "message": "No documents to add."}

        # 移除 first_doc.metadata 的调用，因为 documents 是字典列表
        document_to_add = documents[0]
        
        url = document_to_add.get("url")
        doc_name = document_to_add.get("doc_name")
        doc_id = document_to_add.get("doc_id")
        doc_type = document_to_add.get("doc_type")
        meta = document_to_add.get("meta")
        
        if not url:
            logger.error("Document dictionary is missing 'url'. Cannot index.")
            return {"status": "error", "message": "Document dictionary is missing 'url'"}
        
        # 你的逻辑已经将所有数据准备在了一个字典中，所以这里只需要直接使用它们
        
        indexing_metadata = {
            "url": url,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_type": doc_type,
            "knowledge_base_id": knowledge_base_id,
            "meta": meta
        }
        logger.info(f"Preparing to index document with the following metadata: {indexing_metadata}")
        
        try:
            # 调用火山引擎服务进行上传
            result = await self.volcano_rag_service.import_document_url(
                url=url,
                doc_id=doc_id,
                doc_name=doc_name,
                doc_type=doc_type,
                knowledge_base_id=knowledge_base_id,
                meta=meta
            )
            logger.info(f"Document '{doc_name}' from URL uploaded successfully. Response: {result}")
            return {"status": "success", "result": result}
            
        except Exception as e:
            logger.error(f"Failed to upload document '{doc_name}' from URL: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "doc_name": doc_name}

    def delete_nodes_by_metadata(self, knowledge_base_id: str, filters: dict) -> dict:
        """
        根据元数据过滤器删除文档。
        """
        # 在这里，我们将使用你提供的具体知识库ID
        knowledge_base_id = "kb-a026f6f25e1b2a25"
        
        logger.info(f"Attempting to delete documents from '{knowledge_base_id}' with filters: {filters}")
        
        # 你的业务逻辑应该在调用前获取 doc_id，这里只支持 doc_id 的精确删除
        doc_id_to_delete = filters.get("doc_id")
        if not doc_id_to_delete:
            message = "Deletion requires a 'doc_id' filter."
            logger.error(message)
            return {"status": "error", "message": message}
            
        try:
            response = self.volcano_rag_service.delete_document(
                knowledge_base_id=knowledge_base_id,
                doc_id=doc_id_to_delete
            )
            message = f"Successfully deleted document '{doc_id_to_delete}'. Response: {response}"
            logger.info(message)
            return {"status": "success", "message": message}
            
        except Exception as e:
            message = f"An error occurred during deletion: {e}"
            logger.error(message, exc_info=True)
            return {"status": "error", "message": message}
            
    def update_existing_nodes_metadata(self, knowledge_base_id: str, doc_id: str, metadata_update_payload: dict) -> dict:
        """
        更新现有文档的元数据。
        """
        # 在这里，我们将使用你提供的具体知识库ID
        knowledge_base_id = "kb-a026f6f25e1b2a25"
        
        logger.info(f"Performing metadata update for doc_id: {doc_id} in KB '{knowledge_base_id}' with payload: {metadata_update_payload}")
        
        if not doc_id or not metadata_update_payload:
            message = "Update requires a 'doc_id' and metadata payload."
            logger.warning(message)
            return {"status": "error", "message": message}
            
        try:
            # 火山引擎的API没有直接的元数据更新方法。
            # 这部分逻辑需要你在业务侧重新设计。
            message = f"Update is not directly supported by Volcano Engine API for doc_id {doc_id}. Please re-upload with new metadata."
            logger.warning(message)
            return {"status": "not_implemented", "message": message}
            
        except Exception as e:
            message = f"An error occurred during metadata update: {e}"
            logger.error(message, exc_info=True)
            return {"status": "error", "message": message}
            
    async def update_document_meta(self, 
                             doc_id: str, 
                             knowledge_base_id: str, 
                             meta_updates: List[Dict[str, Any]]) -> dict:
        """
        通用元数据更新方法，直接调用 VolcanoEngineRagService。
        """
        try:
            logger.info(f"Preparing to update meta for doc {doc_id} in KB {knowledge_base_id}. Meta Updates: {meta_updates}")
            # 2. 调用火山引擎更新 API
            response = await self.volcano_rag_service.update_document_meta(
                knowledge_base_id=knowledge_base_id,
                doc_id=doc_id,
                meta_updates=meta_updates
            )
            
            if response.get("code") == 0:
                return {
                    "message": f"Successfully updated meta for doc {doc_id}.",
                    "status": "success",
                    "doc_id": doc_id
                }
            else:
                return {
                    "message": f"Volcano API failed to update meta: {response.get('message')}",
                    "status": "error",
                }
                
        except Exception as e:
            logger.error(f"Failed to update document meta for doc {doc_id}: {e}", exc_info=True)
            return {
                "message": f"An API error occurred: {str(e)}",
                "status": "error",
            }