# In services/document_oss_service.py

import datetime
import gc
import os
import shutil
import logging
from typing import List, Optional, Tuple

from fastapi import Depends

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument
import psutil
from core.rag_config import RagConfig
from models.schemas import UploadFromOssRequest
from core.config import settings
from services.indexer_service import IndexerService
from services.oss_service import OssService
from services.task_manager_service import TaskManagerService
from datetime import datetime

logger = logging.getLogger(__name__)
def _log_memory_usage(context: str):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory Usage ({context}): RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB")

class DocumentOssService:
    def __init__(self, 
                 indexer_service: IndexerService,
                 oss_service_instance: OssService,
                 task_manager_service: TaskManagerService):
        self.data_dir = settings.DATA_DIR
        self.data_config_dir = settings.DATA_CONFIG_DIR
        self.indexer_service = indexer_service
        self.oss_service = oss_service_instance
        self.task_manager = task_manager_service

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.data_config_dir, exist_ok=True)
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

    async def process_new_oss_file(self, 
                             request: UploadFromOssRequest, 
                             task_id: str,
                             rag_config: RagConfig) -> dict:
        """
        一个完整的、自包含的OSS文件处理流程。
        它不返回任何东西，而是通过task_manager更新任务进度。
        """
        local_file_path = None
        file_key = request.file_key
        display_file_name = request.metadata.file_name or "unknown_file"
         
        # 1. 去重检查
        if file_key in self.processed_files:
            logger.warning(f"[TASK_ID: {task_id}] Duplicate file key detected: {file_key}")
            result_payload = {
                "message": f"This OSS file has already been processed (Original Filename: {self.processed_files[file_key]}).",
                "file_key": file_key
            }
            self.task_manager.finish_task(task_id, "duplicate", result=result_payload)
            return
        
        try:
            # 2. 准备基础元数据
            base_metadata_payload = request.metadata.model_dump()
            
            # 提取并清理不需要作为普通字符串元数据导入的字段
            accessible_to_list = base_metadata_payload.pop('accessible_to', None) or []
            level_list = base_metadata_payload.pop('level_list', None) 
            base_metadata_payload.pop('file_name', None) 
            base_metadata_payload.pop('file_key', None) 
            
            # 3. 生成可导入的 URL (逻辑不变)
            self.task_manager.update_progress(task_id, 10, "Generating temporary URL from OSS...")
            
            # 确定存储桶
            target_bucket = settings.OSS_PUBLIC_BUCKET_NAME if "public" in accessible_to_list else settings.OSS_PRIVATE_BUCKET_NAME
            
            temp_url = self.oss_service.generate_presigned_url(
                object_key=file_key,
                bucket_name=target_bucket,
                expires_in=3600 # URL有效期（秒）
            )
            logger.info(f"Successfully generated pre-signed URL for {file_key}")

            import re
            # 正则表达式: 匹配所有不是字母、数字、_、- 的字符
            cleaned_doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', file_key)
            # 确保第一个字符是字母或下划线
            if not re.match(r'^[a-zA-Z_]', cleaned_doc_id):
                cleaned_doc_id = 'doc_' + cleaned_doc_id

            # 4. 准备通用文档元数据 (核心改造区域)
            
            # 4.1 初始化通用元数据字典
            generic_metadata = {}

            # 4.2 核心业务列表字段 (直接赋值，不进行供应商格式转换)
            
            # 1. 添加 accessible_to (权限控制字段)
            if accessible_to_list:
                generic_metadata["accessible_to"] = accessible_to_list 
                logger.debug(f"Added accessible_to meta: {accessible_to_list}")

            # 2. 添加 level_list (等级列表)
            if level_list:
                generic_metadata["level_list"] = level_list
                logger.debug(f"Added level_list meta: {level_list}")
                
            # 4.3 将其他自定义元数据添加到通用字典
            for key, value in base_metadata_payload.items():
                # 排除不再需要的或已在上面处理过的字段
                if key not in ["creation_date", "document_type"] and value is not None:
                    generic_metadata[key] = value

            # 5. 最终调用 IndexerService 的文档结构
            final_document_payload = {
                "url": temp_url,
                "doc_name": display_file_name,
                "doc_id": cleaned_doc_id, # 使用清理后的 doc_id
                "doc_type": os.path.splitext(file_key)[1].strip('.').lower(),
                # 传递通用字典，由底层服务（Bailian/Volcano）负责适配
                "meta": generic_metadata 
            }

            # 根据火山引擎文档，doc_type需要进行特殊处理 (此处保留，因为它与文档类型强相关)
            if final_document_payload["doc_type"] == 'jpg':
                final_document_payload["doc_type"] = 'jpeg'

            self.task_manager.update_progress(task_id, 30, "Document payload prepared. Indexing...")

            # 6. 调用 IndexerService 进行索引 (逻辑不变)
            logger.info(f"Indexing document '{display_file_name}' into knowledge base '{rag_config.knowledge_base_id}'...")
            
            # IndexerService 负责将这里的通用 meta 适配给底层 KB
            result = await self.indexer_service.add_documents_to_index(
                documents=[final_document_payload], 
                knowledge_base_id=rag_config.knowledge_base_id 
            )
            
            self.processed_files[file_key] = display_file_name
            self._save_processed_keys()
            
            success_result = {
                "message": "File from OSS has been processed and indexed successfully.",
                "file_key": file_key,
                "knowledge_base_id": rag_config.knowledge_base_id
            }
            self.task_manager.finish_task(task_id, "success", result=result.get('result'))
            return {"status": "success", "message": success_result["message"]}

        except Exception as e:
            logger.error(f"Error during processing of OSS key '{file_key}': {e}", exc_info=True)
            self.task_manager.finish_task(task_id, "error", result={"message": str(e)})
            return {"status": "error", "message": str(e)}
        finally:
            pass

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
