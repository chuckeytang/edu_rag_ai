# In services/document_oss_service.py

import os
import shutil
import logging
from typing import List, Optional, Tuple

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument
from models.schemas import UploadResponse, UploadFromOssRequest
from core.config import settings
from models.schemas import RAGMetadata

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

    # def process_temp_file(self, local_file_path: str, file_key: str, metadata: dict) -> Tuple[List[LlamaDocument], List[LlamaDocument]]:
    #     """
    #     Processes a file directly from its temporary local path without saving it permanently.
    #     1. Updates the tracking of processed file keys with the filename for readability.
    #     2. Loads the document content FROM THE TEMP PATH and injects metadata.
    #     3. Filters blank pages.
    #     4. Returns both all loaded docs and the filtered docs.
    #     """
    #     # 1. Update tracking with the new file_key and save the readable filename as its value.
    #     #    NOTE: We are NOT saving the file permanently anymore.
    #     original_filename = metadata.get('file_name', 'unknown_filename')
    #     self.processed_files[file_key] = original_filename
    #     self._save_processed_keys()
    #     logger.info(f"Added OSS key '{file_key}' (Filename: '{original_filename}') to processed files cache.")

    #     # 2. Load documents directly from the provided temporary path
    #     logger.info(f"Loading documents directly from temporary path: '{local_file_path}'")
    #     all_docs = self._load_documents_with_metadata(
    #         file_name=local_file_path, # <-- Use the temp path here
    #         extra_metadata=metadata
    #     )

    #     # 3. Filter blank pages
    #     filtered_docs, _ = self._filter_documents(all_docs)

    #     return all_docs, filtered_docs

    # def _load_documents_with_metadata(self, file_name: str, extra_metadata: Optional[dict] = None) -> List[LlamaDocument]:
    #     """Loads a single document and injects metadata into each page."""
    #     try:
    #         docs = SimpleDirectoryReader(input_files=[file_name]).load_data()

    #         for doc in docs:
    #             if doc.metadata is None:
    #                 doc.metadata = {}
    #             if extra_metadata:
    #                 doc.metadata.update(extra_metadata)
    #             if "page_label" not in doc.metadata:
    #                 doc.metadata["page_label"] = doc.metadata.get('page_label', '1')

    #         logger.debug(f"Metadata injected into {len(docs)} pages from file '{file_name}'.")
    #         return docs
    #     except FileNotFoundError as e:
    #         logger.error(f"File not found during document loading: {e}")
    #         raise ValueError(f"File not found: {e}")


    # --- 核心修改：这是一个新的、封装了完整流程的公共方法 ---
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
        metadata = request.metadata.dict()
        file_name = metadata.get("file_name", "unknown_file")
        collection_name = request.collection_name or "public_collection"

        # 1. 去重检查 (Deduplication Check)
        if file_key in self.processed_oss_keys:
            original_filename = self.processed_oss_keys[file_key]
            return {
                "message": f"This OSS file has already been processed (Original Filename: {original_filename}).",
                "status": "duplicate"
            }
            
        try:
            # 2. 下载文件到临时位置 (Download)
            logger.info(f"Downloading file from OSS for key: '{file_key}'")
            local_file_path = oss_service.download_file_to_temp(file_key)

            # 3. 从临时文件加载和处理文档 (Process)
            logger.info(f"Processing temporary file: '{local_file_path}'")
            all_docs, filtered_docs = self._prepare_documents_from_temp_file(local_file_path, metadata)
            
            if not filtered_docs:
                message = "No valid content found in the file after filtering."
                task_status = "error"
            else:
                total_pages = len(all_docs)
                pages_loaded = len(filtered_docs)
                
                # 4. 索引文档 (Index)
                logger.info(f"Indexing {pages_loaded} document chunks into collection '{collection_name}'...")
                query_service.update_index(filtered_docs, collection_name=collection_name)
                
                # 5. 成功后，更新处理记录
                self.processed_oss_keys[file_key] = file_name
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
        
        for doc in all_docs:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata.update(metadata)
            if "page_label" not in doc.metadata:
                doc.metadata["page_label"] = doc.metadata.get('page_label', '1')

        filtered_docs = self._filter_documents(all_docs)
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