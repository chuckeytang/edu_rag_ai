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
            # --- [最终状态] 发现重复文件 ---
            result_payload = {
                "message": f"This OSS file has already been processed (Original Filename: {self.processed_files[file_key]}).",
                "file_key": file_key
            }
            self.task_manager.finish_task(task_id, "duplicate", result=result_payload)
            return
        
        try:
            # 2. 准备基础元数据
            base_metadata_payload = request.metadata.model_dump()
            base_metadata_payload['file_key'] = file_key

            # 提取并清理不需要作为普通字符串元数据导入的字段
            accessible_to_list = base_metadata_payload.pop('accessible_to', None) or []
            level_list = base_metadata_payload.pop('level_list', None) 
            base_metadata_payload.pop('file_name', None) # file_name 作为 doc_name
            base_metadata_payload.pop('file_key', None)  # file_key 已在上方使用
            
            # 3. 生成可导入的 URL
            self.task_manager.update_progress(task_id, 10, "Generating temporary URL from OSS...")
            
            # 确定存储桶
            accessible_to_list = request.metadata.accessible_to or []
            target_bucket = settings.OSS_PUBLIC_BUCKET_NAME if "public" in accessible_to_list else settings.OSS_PRIVATE_BUCKET_NAME
            
            # 调用 OssService 的新方法来生成预签名 URL
            # 这是一个关键步骤，因为火山引擎需要一个可公开访问的 URL
            import urllib.parse
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

            # 4. 准备火山引擎所需的文档元数据
            # 火山引擎的doc/add接口只处理一个文件URL，所以我们不再需要处理LlamaDocument的列表
            final_document_payload = {
                "url": temp_url,
                "doc_name": display_file_name,
                "doc_id": cleaned_doc_id, # 使用清理后的 doc_id
                "doc_type": os.path.splitext(file_key)[1].strip('.').lower(),
                "meta": []
            }

            # 根据火山引擎文档，doc_type需要进行特殊处理
            if final_document_payload["doc_type"] == 'jpg':
                final_document_payload["doc_type"] = 'jpeg'
            
            # 1. 添加 accessible_to (权限控制字段)
            if accessible_to_list:
                final_document_payload["meta"].append({
                    "field_name": "accessible_to",
                    # 明确指定类型为 list<string>
                    "field_type": "list<string>", 
                    # 直接传入列表
                    "field_value": accessible_to_list 
                })
                logger.debug(f"Added accessible_to meta: {accessible_to_list}")

            # 2. 添加 level_list (等级列表)
            if level_list:
                final_document_payload["meta"].append({
                    "field_name": "level_list",
                    # 明确指定类型为 list<string>
                    "field_type": "list<string>", 
                    # 直接传入列表
                    "field_value": level_list 
                })
                logger.debug(f"Added level_list meta: {level_list}")
                
            # 将自定义元数据转换为火山引擎的 "meta" 格式
            for key, value in base_metadata_payload.items():
                if key not in ["file_name", "file_key", "accessible_to", "creation_date", "document_type"] and value is not None:
                    # 注意: field_type 需要根据你创建知识库时的配置来确定
                    # 这里假设都是string类型，如果你的标签有其他类型，需要做相应判断
                    final_document_payload["meta"].append({
                        "field_name": key,
                        "field_type": "string",
                        "field_value": str(value)
                    })

            self.task_manager.update_progress(task_id, 30, "Document payload prepared. Indexing...")

            # 5. 调用 IndexerService 进行索引
            logger.info(f"Indexing document '{display_file_name}' into knowledge base '{rag_config.knowledge_base_id}'...")
            
            await self.indexer_service.add_documents_to_index(
                documents=[final_document_payload], # 传入包含单个文档字典的列表
                knowledge_base_id=rag_config.knowledge_base_id 
            )
            
            self.processed_files[file_key] = display_file_name
            self._save_processed_keys()
            
            success_result = {
                "message": "File from OSS has been processed and indexed successfully.",
                "file_key": file_key,
                "knowledge_base_id": rag_config.knowledge_base_id
            }
            self.task_manager.finish_task(task_id, "success", result=success_result)
            return {"status": "success", "message": success_result["message"]}

        except Exception as e:
            logger.error(f"Error during processing of OSS key '{file_key}': {e}", exc_info=True)
            self.task_manager.finish_task(task_id, "error", result={"message": str(e)})
            return {"status": "error", "message": str(e)}
        finally:
            # 清理，由于是URL导入，不再需要本地文件，所以这部分可以精简
            # 但如果你保留了下载到本地的逻辑，清理是必要的
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
