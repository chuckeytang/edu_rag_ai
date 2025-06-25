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
            clean_metadata = request.metadata.model_dump()
            
            # 将顶层的 file_key 也添加到元数据中
            clean_metadata['file_key'] = file_key
            
            # --- 2. 对列表字段进行特殊转换 ---
            # 这里的键已经是正确的蛇形命名 (snake_case)
            for key, value in clean_metadata.items():
                if isinstance(value, list):
                    # 将列表转换为 ",item1,item2," 格式的字符串
                    transformed_value = f",{','.join(map(str, value))},"
                    clean_metadata[key] = transformed_value
                    logger.info(f"Transformed list in metadata key '{key}' to string: '{transformed_value}'")

            # --- 3. 移除所有值为 None 的键，保持元数据干净 ---
            clean_metadata = {k: v for k, v in clean_metadata.items() if v is not None}
            logger.info(f"Constructed clean metadata for indexing: {clean_metadata}")

            # --- 去重检查 ---
            display_file_name = clean_metadata.get("file_name", "unknown_file")
            
            # 下载文件到临时位置 (Download)
            # 使用请求中的原始列表进行判断
            accessible_to_list = request.metadata.accessible_to or []
            if "public" in accessible_to_list:
                target_bucket = settings.OSS_PUBLIC_BUCKET_NAME
            else:
                target_bucket = settings.OSS_PRIVATE_BUCKET_NAME
            
            local_file_path = oss_service.download_file_to_temp(
                object_key=file_key, 
                bucket_name=target_bucket
            )

            # --- 从临时文件加载文档并应用干净的元数据 ---
            logger.info(f"Loading documents from temporary file: '{local_file_path}'")
            # 我们只用 SimpleDirectoryReader 来解析文本和页码
            all_docs = SimpleDirectoryReader(input_files=[local_file_path]).load_data()
            
            # 关键：我们忽略 LlamaIndex 自动生成的元数据，强制使用我们自己构建的 clean_metadata
            for doc in all_docs:
                page_label = doc.metadata.get("page_label", "1") # 只保留页码
                doc.metadata = clean_metadata.copy() # 使用干净元数据的副本
                doc.metadata["page_label"] = page_label # 再把页码加上

            filtered_docs = all_docs

            if not filtered_docs:
                message = "No valid content found in the file after filtering."
                task_status = "error"
            else:
                total_pages = len(all_docs)
                pages_loaded = len(filtered_docs)
                
                logger.debug(f"Type of filtered_docs: {type(filtered_docs)}")
                for i, item in enumerate(filtered_docs):
                    logger.debug(f"Item {i} in filtered_docs has type: {type(item)}")
                    
                # 索引文档 (Index)
                logger.info(f"Indexing {pages_loaded} document chunks into collection '{collection_name}'...")
                query_service.update_index(filtered_docs, collection_name=collection_name)
                
                # 成功后，更新处理记录
                self.processed_files[file_key] = display_file_name
                self._save_processed_keys()
                
                message = "File from OSS has been processed and indexed successfully."
                task_status = "success"
                logger.info(f"Successfully processed and indexed file for key: '{file_key}'")

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