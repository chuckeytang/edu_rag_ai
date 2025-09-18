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

    def process_new_oss_file(self, 
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
        collection_name = request.collection_name or "public_collection"
        _log_memory_usage("Start of process_new_oss_file")
         
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
            # 根据文件类型决定加载方式
            loaded_original_docs: List[LlamaDocument] = []

            # 从 OSS 下载文件
            self.task_manager.update_progress(task_id, 10, "Downloading file from OSS...")
            
            # 确定存储桶
            accessible_to_list = request.metadata.accessible_to or []
            target_bucket = settings.OSS_PUBLIC_BUCKET_NAME if "public" in accessible_to_list else settings.OSS_PRIVATE_BUCKET_NAME

            local_file_path = self.oss_service.download_file_to_temp(
                object_key=file_key, 
                bucket_name=target_bucket
            )
            self.task_manager.update_progress(task_id, 30, "File download complete. Processing document...")
            _log_memory_usage("After file download, before document loading")
            
            # --- 2. 加载原始 LlamaDocument ---
            custom_file_extractor = {
                ".pdf": CamelotPDFReader(
                    flavor='stream', # 或 'lattice'
                    table_settings={},
                    extract_text_also=True,
                    chunk_tables_intelligently=True
                )
            }

            reader = SimpleDirectoryReader(
                input_files=[local_file_path], 
                recursive=True, 
                file_extractor=custom_file_extractor
            )
            loaded_original_docs = reader.load_data()
            _log_memory_usage("After document loading")
            
            if not loaded_original_docs:
                raise ValueError("Could not load any documents from the provided file.")

            # 获取原始总页数（仅对PDF文件有效，从第一个Document的metadata中推断或重新读取）
            total_pages = 1 # 默认值
            file_extension_lower = os.path.splitext(local_file_path)[1].lower()
            if file_extension_lower == '.pdf' and loaded_original_docs:
                try:
                    # 从原始PDF获取总页数，可能需要再次使用 pypdf
                    from pypdf import PdfReader
                    pdf_reader_obj = PdfReader(local_file_path)
                    total_pages = len(pdf_reader_obj.pages)
                except Exception:
                    logger.warning(f"Could not determine total pages for PDF: {local_file_path}. Defaulting to 1.")
                    total_pages = 1 # 降级处理
            elif loaded_original_docs: # 对于非PDF文件，一个doc可能就是一页或更多
                # 这是一个大致的页数，LlamaIndex默认会将单文件视为1页
                total_pages = max([int(doc.metadata.get("page_label", 1)) for doc in loaded_original_docs if doc.metadata and doc.metadata.get("page_label")], default=1)

            # --- 3. 准备基础元数据并附加到每个 LlamaDocument 上 ---
            # 将顶层的 file_key 也添加到元数据中
            base_metadata_payload = request.metadata.model_dump()
            base_metadata_payload['file_key'] = file_key

            base_metadata_payload.pop('accessible_to', None)
            base_metadata_payload.pop('level_list', None) 

            # 添加文件类型和时间戳
            file_extension = os.path.splitext(file_key)[1].lower()
            if file_extension == '.pdf':
                base_metadata_payload['document_type'] = 'PDF'
            elif file_extension in ('.doc', '.docx'):
                base_metadata_payload['document_type'] = 'Word'
            # 您可以根据需要添加更多文件类型判断
            else:
                base_metadata_payload['document_type'] = 'Unknown'

            base_metadata_payload['creation_date'] = datetime.now().isoformat()
            
            base_metadata_payload = {k: v for k, v in base_metadata_payload.items() if v is not None}
            logger.info(f"Constructed base metadata: {base_metadata_payload}")
            display_file_name = base_metadata_payload.get("file_name", "unknown_file")
            
            logger.info("Applying node duplication strategy based on ACLs...")
            # 如果权限列表为空，为了防止文档丢失，默认将其归属给作者
            if not accessible_to_list and 'author_id' in base_metadata_payload:
                acl_tag = str(base_metadata_payload['author_id'])
                accessible_to_list.append(acl_tag)
                logger.warning(f"ACL list was empty. Defaulting to author_id: {acl_tag}")

            # 为每一个权限标签，创建一套完整的文档节点副本
            final_documents_for_indexing: List[LlamaDocument] = []
            
            # 如果 accessible_to_list 仍然为空 (例如，没有 author_id 也没有指定 accessible_to)
            # 则默认使用 "private_default" 或直接跳过
            if not accessible_to_list:
                logger.warning(f"No specific ACL tags found for {file_key}. Document will be indexed with 'private_default' ACL.")
                accessible_to_list.append("private_default") # 默认一个私有标签
            
            # 为每一个权限标签，创建一套完整的文档节点副本
            final_nodes_to_index = []
            for acl_tag in accessible_to_list:
                for doc in loaded_original_docs:
                    # 复制基础元数据
                    metadata_copy = base_metadata_payload.copy()
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
            self.task_manager.update_progress(task_id, 75, f"Processing complete. Found {pages_loaded} nodes to index.")

            # 4. 索引所有生成的节点
            if not final_nodes_to_index:
                self.task_manager.finish_task(task_id, "error", result={"message": "No valid content or nodes generated for indexing."})
            else:
                logger.info(f"Indexing {pages_loaded} document chunks into collection '{collection_name}'...")
                self.indexer_service.add_documents_to_index(final_nodes_to_index, collection_name=collection_name, rag_config=rag_config)
                
                self.processed_files[file_key] = display_file_name
                self._save_processed_keys()
                
                success_result = {
                    "message": "File from OSS has been processed and indexed successfully.",
                    "pages_loaded": pages_loaded,
                    "total_pages": total_pages,
                    "file_key": file_key
                }
                self.task_manager.finish_task(task_id, "success", result=success_result)
                return {"status": "success", "message": success_result["message"]} # 明确返回

        except Exception as e:
            logger.error(f"Error during processing of OSS key '{file_key}': {e}", exc_info=True)
            self.task_manager.finish_task(task_id, "error", result={"message": str(e)})
            return {"status": "error", "message": str(e)}
        finally:
            # 清理临时文件
            if local_file_path:
                temp_dir = os.path.dirname(local_file_path)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: '{temp_dir}'")
            gc.collect() 
            _log_memory_usage("End of process_new_oss_file")

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
