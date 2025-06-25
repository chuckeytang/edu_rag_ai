# In services/document_oss_service.py

import os
import shutil
import logging
from typing import List, Optional, Tuple

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument
from models.schemas import UploadResponse, UploadFromOssRequest, UpdateMetadataRequest
from core.config import settings
from models.schemas import RAGMetadata
from services.oss_service import oss_service
from services.query_service import query_service  

logger = logging.getLogger(__name__)

class DocumentOssService:
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.data_config_dir = settings.DATA_CONFIG_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.data_config_dir, exist_ok=True)
        # This dictionary tracks processed files by their unique OSS file_key
        self.processed_files = self._load_processed_keys()

    @property
    def key_file_path(self):
        """Path to the file that stores the mapping of processed OSS keys."""
        return os.path.join(self.data_config_dir, ".oss_file_keys")

    def _load_processed_keys(self) -> dict:
        """Loads the map of processed oss_file_key -> original_filename."""
        keys = {}
        if os.path.exists(self.key_file_path):
            with open(self.key_file_path, "r") as f:
                for line in f:
                    if ":" in line:
                        key, filename = line.strip().split(":", 1)
                        keys[key] = filename
        logger.info(f"DocumentOssService: Loaded {len(keys)} processed OSS file keys from '{self.key_file_path}'.")
        return keys

    def _save_processed_keys(self):
        """Saves the current map of oss_file_key -> original_filename."""
        with open(self.key_file_path, "w") as f:
            for key, filename in self.processed_files.items():
                f.write(f"{key}:{filename}\n")

    def process_new_oss_file(self, request: UploadFromOssRequest) -> dict:
        """
        一个完整的、自包含的OSS文件处理流程。
        返回一个包含最终状态的字典。
        """
        local_file_path = None
        task_status = "error"
        message = "An unexpected error occurred."
        pages_loaded = 0
        total_pages = 0
        
        file_key = request.file_key

        # 从请求中获取原始元数据字典
        collection_name = request.collection_name or "public_collection"
         
        if file_key in self.processed_files:
            return {
                "message": f"This OSS file has already been processed (Original Filename: {self.processed_files[file_key]}).",
                "status": "duplicate"
            }
        
        try:
            # --- 1. 使用 Pydantic 的 model_dump() 自动处理别名和命名转换 ---
            # model_dump() 不带参数，会生成一个以字段名(snake_case)为键的字典
            base_metadata = request.metadata.model_dump()
            
            # 将顶层的 file_key 也添加到元数据中
            base_metadata['file_key'] = file_key
            base_metadata.pop('accessible_to', None)
            base_metadata.pop('level_list', None) 
            base_metadata = {k: v for k, v in base_metadata.items() if v is not None}
            logger.info(f"Constructed base metadata for duplication: {base_metadata}")

            # --- 2. 决定存储桶并下载文件 (逻辑不变) ---
            display_file_name = base_metadata.get("file_name", "unknown_file")
            accessible_to_list = request.metadata.accessible_to or []
            if "public" in accessible_to_list:
                target_bucket = settings.OSS_PUBLIC_BUCKET_NAME
            else:
                target_bucket = settings.OSS_PRIVATE_BUCKET_NAME
            local_file_path = oss_service.download_file_to_temp(
                object_key=file_key, 
                bucket_name=target_bucket
            )

            # --- 3. 加载原始文档区块 (逻辑不变) ---
            logger.info(f"Loading documents from temporary file: '{local_file_path}'")
            all_docs = SimpleDirectoryReader(input_files=[local_file_path]).load_data()
            total_pages = len(all_docs)
            
            # --- 4. 核心修改：根据权限列表，创建节点副本 ---
            logger.info("Applying node duplication strategy based on ACLs...")
            final_nodes_to_index = []

            # 如果权限列表为空，为了防止文档丢失，默认将其归属给作者
            if not accessible_to_list and 'author_id' in base_metadata:
                acl_tag = str(base_metadata['author_id'])
                accessible_to_list.append(acl_tag)
                logger.warning(f"ACL list was empty. Defaulting to author_id: {acl_tag}")

            # 为每一个权限标签，创建一套完整的文档节点副本
            for acl_tag in accessible_to_list:
                for doc in all_docs:
                    # 复制基础元数据
                    metadata_copy = base_metadata.copy()
                    # 为这个副本设置单一的、字符串类型的访问标签
                    metadata_copy['accessible_to'] = str(acl_tag)
                    # 保留原始的页码信息
                    metadata_copy['page_label'] = doc.metadata.get("page_label", "1")
                    
                    # 创建一个新的 LlamaDocument 节点对象
                    new_node = LlamaDocument(
                        text=doc.text,
                        metadata=metadata_copy
                    )
                    final_nodes_to_index.append(new_node)
            
            pages_loaded = len(final_nodes_to_index)
            logger.info(f"Generated a total of {pages_loaded} nodes for indexing.")
            
            # 5. 索引所有生成的节点
            if not final_nodes_to_index:
                message = "No valid content found or no nodes generated for indexing."
                task_status = "error"
            else:
                logger.info(f"Indexing {pages_loaded} document chunks into collection '{collection_name}'...")
                query_service.update_index(final_nodes_to_index, collection_name=collection_name)
                
                self.processed_files[file_key] = display_file_name
                self._save_processed_keys()
                
                message = "File from OSS has been processed and indexed successfully."
                task_status = "success"

        except Exception as e:
            logger.error(f"Error during processing of OSS key '{file_key}': {e}", exc_info=True)
            message = f"An error occurred: {str(e)}"
            task_status = "error"
        finally:
            # 清理临时文件
            if local_file_path:
                temp_dir = os.path.dirname(local_file_path)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: '{temp_dir}'")
        
        return {
            "message": message,
            "status": task_status,
            "pages_loaded": pages_loaded,
            "total_pages": total_pages
        }

    def _filter_documents(self, documents: List[LlamaDocument]) -> Tuple[List[LlamaDocument], dict]:
        """Filters out blank or empty pages from a list of documents."""
        filtered_docs = []
        page_info = {}
        for doc in documents:
            text = doc.text.strip()
            if text and "BLANK PAGE" not in text.upper():
                filtered_docs.append(doc)
                fname = doc.metadata.get("file_name", "unknown")
                if fname not in page_info:
                    page_info[fname] = set()
                page_info[fname].add(doc.metadata.get("page_label", "unknown"))
        return filtered_docs, page_info

        
# Create a singleton instance for the application to use
document_oss_service = DocumentOssService()