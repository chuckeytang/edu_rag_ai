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
from services.task_manager_service import task_manager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

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

    def process_new_oss_file(self, request: UploadFromOssRequest, task_id: str) -> dict:
        """
        一个完整的、自包含的OSS文件处理流程。
        它不返回任何东西，而是通过task_manager更新任务进度。
        """
        local_file_path = None
        file_key = request.file_key
        display_file_name = request.metadata.file_name or "unknown_file"
        collection_name = request.collection_name or "public_collection"
         
        # 1. 去重检查
        if file_key in self.processed_files:
            logger.warning(f"[TASK_ID: {task_id}] Duplicate file key detected: {file_key}")
            # --- [最终状态] 发现重复文件 ---
            result_payload = {
                "message": f"This OSS file has already been processed (Original Filename: {self.processed_files[file_key]}).",
                "file_key": file_key
            }
            task_manager.finish_task(task_id, "duplicate", result=result_payload)
            return
        
        try:
            # --- [进度汇报] 开始下载 ---
            task_manager.update_progress(task_id, 10, "Downloading file from OSS...")
            
            # 2. 决定存储桶并下载文件
            accessible_to_list = request.metadata.accessible_to or []
            # 将顶层的 file_key 也添加到元数据中
            base_metadata = request.metadata.model_dump()
            base_metadata['file_key'] = file_key
            base_metadata.pop('accessible_to', None)
            base_metadata.pop('level_list', None) 
            base_metadata = {k: v for k, v in base_metadata.items() if v is not None}
            logger.info(f"Constructed base metadata: {base_metadata}")
            display_file_name = base_metadata.get("file_name", "unknown_file")
            if "public" in accessible_to_list:
                target_bucket = settings.OSS_PUBLIC_BUCKET_NAME
            else:
                target_bucket = settings.OSS_PRIVATE_BUCKET_NAME
            local_file_path = oss_service.download_file_to_temp(
                object_key=file_key, 
                bucket_name=target_bucket
            )
            # --- [进度汇报] 下载完成 ---
            task_manager.update_progress(task_id, 30, "File download complete. Processing document...")
            
            # --- 3. 加载原始文档区块 ---
            logger.info(f"Loading documents from temporary file: '{local_file_path}'")
            all_docs = SimpleDirectoryReader(input_files=[local_file_path]).load_data()
            total_pages = len(all_docs)
            
            logger.info("Applying node duplication strategy based on ACLs...")
            # 如果权限列表为空，为了防止文档丢失，默认将其归属给作者
            if not accessible_to_list and 'author_id' in base_metadata:
                acl_tag = str(base_metadata['author_id'])
                accessible_to_list.append(acl_tag)
                logger.warning(f"ACL list was empty. Defaulting to author_id: {acl_tag}")

            # 为每一个权限标签，创建一套完整的文档节点副本
            final_nodes_to_index = []
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
            # --- [进度汇报] 文档处理完成 ---
            task_manager.update_progress(task_id, 75, f"Processing complete. Found {pages_loaded} nodes to index.")

            # 4. 索引所有生成的节点
            if not final_nodes_to_index:
                task_manager.finish_task(task_id, "error", result={"message": "No valid content or nodes generated for indexing."})
            else:
                logger.info(f"Indexing {pages_loaded} document chunks into collection '{collection_name}'...")
                query_service.update_index(final_nodes_to_index, collection_name=collection_name)
                
                self.processed_files[file_key] = display_file_name
                self._save_processed_keys()
                
                success_result = {
                    "message": "File from OSS has been processed and indexed successfully.",
                    "pages_loaded": pages_loaded,
                    "total_pages": total_pages,
                    "file_key": file_key
                }
                task_manager.finish_task(task_id, "success", result=success_result)

        except Exception as e:
            logger.error(f"Error during processing of OSS key '{file_key}': {e}", exc_info=True)
            task_manager.finish_task(task_id, "error", result={"message": str(e)})
        finally:
            # 清理临时文件
            if local_file_path:
                temp_dir = os.path.dirname(local_file_path)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: '{temp_dir}'")
        
        return 

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