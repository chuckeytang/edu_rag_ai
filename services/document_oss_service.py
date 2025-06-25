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
        accessible_to = request.metadata.accessible_to or []
        metadata = request.metadata.model_dump(by_alias=True, exclude_unset=True)
        file_name = metadata.get("fileName", "unknown_file")
        collection_name = request.collection_name or "public_collection"

        # 1. 去重检查 (Deduplication Check)
        if file_key in self.processed_files:
            original_filename = self.processed_files[file_key]
            return {
                "message": f"This OSS file has already been processed (Original Filename: {original_filename}).",
                "status": "duplicate"
            }
            
        try:
            # 2. 下载文件到临时位置 (Download)
            # 根据 'accessible_to' 字段决定使用哪个 bucket
            if "public" in accessible_to:
                target_bucket = settings.OSS_PUBLIC_BUCKET_NAME
                logger.info(f"'{file_key}' identified as a public document. Using public bucket: '{target_bucket}'.")
            else:
                target_bucket = settings.OSS_PRIVATE_BUCKET_NAME
                logger.info(f"'{file_key}' identified as a private document. Using private bucket: '{target_bucket}'.")
            
            # 2. 下载文件到临时位置 (Download)
            # 将决定好的 target_bucket 传递给 oss_service
            local_file_path = oss_service.download_file_to_temp(
                object_key=file_key, 
                bucket_name=target_bucket
            )

            # 3. 从临时文件加载和处理文档 (Process)
            logger.info(f"Processing temporary file: '{local_file_path}'")
            all_docs, filtered_docs = self._prepare_documents_from_temp_file(local_file_path, metadata)
            
            if not filtered_docs:
                message = "No valid content found in the file after filtering."
                task_status = "error"
            else:
                total_pages = len(all_docs)
                pages_loaded = len(filtered_docs)
                
                logger.debug(f"Type of filtered_docs: {type(filtered_docs)}")
                for i, item in enumerate(filtered_docs):
                    logger.debug(f"Item {i} in filtered_docs has type: {type(item)}")
                    
                # 4. 索引文档 (Index)
                logger.info(f"Indexing {pages_loaded} document chunks into collection '{collection_name}'...")
                query_service.update_index(filtered_docs, collection_name=collection_name)
                
                # 5. 成功后，更新处理记录
                self.processed_files[file_key] = file_name
                self._save_processed_keys()
                
                message = "File from OSS has been processed and indexed successfully."
                task_status = "success"
                logger.info(f"Successfully processed and indexed file for key: '{file_key}'")

        except Exception as e:
            logger.error(f"Error during processing of OSS key '{file_key}': {e}", exc_info=True)
            message = f"An error occurred: {str(e)}"
            task_status = "error"
        finally:
            # 6. 清理临时文件 (Cleanup)
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

    # 这个方法现在是私有的，只在内部被 `process_new_oss_file` 调用
    def _prepare_documents_from_temp_file(self, local_file_path: str, metadata: dict) -> Tuple[List[LlamaDocument], List[LlamaDocument]]:
        logger.info(f"Loading documents from temporary path: '{local_file_path}'")
        all_docs = SimpleDirectoryReader(input_files=[local_file_path]).load_data()
        
        for key, value in metadata.items():
            # 如果发现任何一个值是列表类型
            if isinstance(value, list):
                # 将该列表转换为我们统一的“分隔符字符串”格式
                # 例如: ["public"] -> ",public,"
                # 例如: [] -> ",," (这仍然是一个合法的字符串)
                transformed_value = f",{','.join(map(str, value))},"
                metadata[key] = transformed_value
                logger.info(f"Transformed list in metadata key '{key}' to string: '{transformed_value}'")

        for doc in all_docs:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata.update(metadata)
            if "page_label" not in doc.metadata:
                doc.metadata["page_label"] = doc.metadata.get('page_label', '1')

        filtered_docs, _ = self._filter_documents(all_docs)
        return all_docs, filtered_docs

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