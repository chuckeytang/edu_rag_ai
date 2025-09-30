# services/bailian_rag_service.py

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
import httpx 
import os 
from alibabacloud_bailian20231229.client import Client as BailianClient
from Tea.model import TeaModel
# 修正 1：Client Config 路径通常在 tea_openapi 或 tea_util 中
from alibabacloud_tea_openapi.models import Config as BailianConfig 
from alibabacloud_bailian20231229.models import (ApplyFileUploadLeaseRequest, AddFileRequest, 
    RetrieveRequest, DeleteIndexDocumentRequest, SubmitIndexAddDocumentsJobRequest
)
from alibabacloud_tea_util.models import RuntimeOptions

from core.config import settings 
from services.abstract_kb_service import AbstractKnowledgeBaseService

logger = logging.getLogger(__name__)

class BailianRagException(Exception):
    def __init__(self, code, request_id, message=None):
        self.code = code
        self.request_id = request_id
        self.message = f"{message}, code:{self.code}，request_id:{self.request_id}"
    def __str__(self):
        return self.message

class BailianRagService(AbstractKnowledgeBaseService): # 继承抽象接口
    
    def __init__(self, endpoint: str = 'bailian.cn-beijing.aliyuncs.com'):
        super().__init__()
        # 从 settings 或环境变量获取凭证 (假设已配置)
        access_key_id = os.environ.get('BAILIAN_ACCESS_KEY', settings.BAILIAN_ACCESS_KEY)
        access_key_secret = os.environ.get('BAILIAN_ACCESS_SECRET', settings.BAILIAN_ACCESS_SECRET)
        workspace_id = os.environ.get('BAILIAN_WORKSPACE_ID', settings.BAILIAN_WORKSPACE_ID)

        self.workspace_id = workspace_id
        self._client = self._create_client(access_key_id, access_key_secret, endpoint)
        self.default_category_id = 'default' 
        self.parser_type = 'DASHSCOPE_DOCMIND' 

        logger.info(f"Bailian RAG service initialized for Workspace: {self.workspace_id}")

    def _create_client(self, access_key_id: str, access_key_secret: str, endpoint: str) -> BailianClient:
        config = BailianConfig(access_key_id=access_key_id, access_key_secret=access_key_secret)
        config.endpoint = endpoint
        return BailianClient(config)
    
    def _parse_bailian_response(self, response: TeaModel) -> Dict[str, Any]:
        """解析百炼SDK响应，检查HTTP状态码和业务错误。"""
        if not response:
            raise BailianRagException("N/A", "N/A", "Empty response from Bailian API")
        
        response_dict = response.to_map()
        
        if response_dict.get('statusCode') and response_dict.get('statusCode') >= 400:
            error_body = response_dict.get('body', {})
            error_msg = f"HTTP Error {response_dict.get('statusCode')}. Message: {error_body.get('Message', 'N/A')}"
            raise BailianRagException(response_dict.get('statusCode'), response_dict.get('headers', {}).get('x-acs-request-id', 'N/A'), error_msg)

        body = response_dict.get('body', {})
        data = body.get('Data')
        
        if not data:
            if 'Message' in body and body.get('Message') != 'Success':
                 raise BailianRagException(body.get('Code', 'N/A'), body.get('RequestId', 'N/A'), body.get('Message'))
            return {"status": "success", "request_id": body.get('RequestId')}

        return data

    async def _async_bailian_call(self, method, *args, **kwargs) -> Dict[str, Any]:
        """将同步的 Bailian SDK 调用包装成异步方法。"""
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, lambda: method(*args, **kwargs))
            return self._parse_bailian_response(response)
        except Exception as e:
            logger.error(f"Bailian API call failed: {e}", exc_info=True)
            if isinstance(e, BailianRagException):
                raise e
            raise BailianRagException(1000030, "sdk_error", str(e))

    async def _upload_file_content_from_url(self, pre_signed_url: str, source_url: str, headers: Dict[str, str]):
        """异步从 source_url 下载文件内容，并流式上传到 pre_signed_url。"""
        logger.info(f"Fetching content from source URL: {source_url} and uploading to presigned URL.")
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                async with client.stream("GET", source_url) as source_response:
                    source_response.raise_for_status() 
                    upload_response = await client.request(
                        method="PUT", 
                        url=pre_signed_url,
                        headers=headers,
                        content=source_response.aiter_bytes()
                    )
                    upload_response.raise_for_status()
                    logger.info("File content successfully uploaded to Bailian temporary storage.")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error during file fetch/upload: {e.response.status_code} - {e.response.text[:100]}...")
                raise BailianRagException(e.response.status_code, "upload_error", f"File fetch/upload failed: {e.response.status_code}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during file fetch/upload: {e}")
                raise BailianRagException(1000031, "upload_error", f"Unexpected error during file fetch/upload: {str(e)}")


    async def import_document_url(self, 
                                  url: str, 
                                  doc_name: str, 
                                  knowledge_base_id: str, # Index ID
                                  doc_id: str, 
                                  doc_type: str, 
                                  meta: Optional[List[Dict[str, Any]]] = None
                                  ) -> Dict[str, Any]:
        """
        实现百炼的文件导入流程: 1. 申请租约 -> 2. 客户端上传文件 -> 3. 添加文件到类目 -> 4. 提交索引追加任务
        """
        index_id = knowledge_base_id
        
        # 1. 获取文件大小并申请租约
        file_size = 1024 # 简化处理，实际生产应尝试精确获取
        file_md5 = "ignored" 
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                head_resp = await client.head(url)
                head_resp.raise_for_status()
                file_size = int(head_resp.headers.get('content-length'))
        except Exception:
            logger.warning(f"Failed to get file size via HEAD request for {url}. Using default size 1024.")

        lease_request = ApplyFileUploadLeaseRequest(file_name=doc_name, md_5=file_md5, size_in_bytes=file_size)
        lease_response = await self._async_bailian_call(self._client.apply_file_upload_lease_with_options,
            self.default_category_id, self.workspace_id, lease_request, {}, RuntimeOptions())

        lease_id = lease_response.get('FileUploadLeaseId')
        upload_param = lease_response.get('Param', {})
        pre_signed_url = upload_param.get('Url')
        upload_headers_dict = {k: v for k, v in upload_param.get('Headers', {}).items()}
        
        # 2. 客户端上传文件内容到 presigned URL
        await self._upload_file_content_from_url(pre_signed_url, url, upload_headers_dict)
        
        # 3. 添加文件到类目
        add_file_request = AddFileRequest(lease_id=lease_id, parser=self.parser_type, category_id=self.default_category_id)
        add_file_response = await self._async_bailian_call(self._client.add_file_with_options,
            self.workspace_id, add_file_request, {}, RuntimeOptions())
        
        file_id = add_file_response.get('FileId')
        
        # 4. 提交索引追加任务
        submit_request = SubmitIndexAddDocumentsJobRequest(index_id=index_id, document_ids=[file_id], source_type='DATA_CENTER_FILE')
        submit_response = await self._async_bailian_call(self._client.submit_index_add_documents_job_with_options,
            self.workspace_id, submit_request, {}, RuntimeOptions())
        
        job_id = submit_response.get('Id')
        
        return {
            "file_id": file_id,
            "job_id": job_id,
            "status": "Submitted for indexing",
            "request_id": submit_response.get('RequestId')
        }

    async def delete_document(self, knowledge_base_id: str, doc_id: str) -> Dict[str, Any]:
        """删除文件。doc_id 映射为 Bailian File ID。"""
        index_id = knowledge_base_id
        file_id = doc_id 
        
        delete_request = DeleteIndexDocumentRequest(index_id=index_id, document_ids=[file_id])
        
        response = await self._async_bailian_call(self._client.delete_index_document_with_options,
            self.workspace_id, delete_request, {}, RuntimeOptions())
        
        return {"status": "success", "message": f"Delete request submitted for file ID {file_id}", "request_id": response.get('RequestId')}

    async def retrieve_documents(self,
                                 query_text: str,
                                 knowledge_base_id: str,
                                 limit: int = 10,
                                 rerank_switch: bool = True,
                                 filters: Optional[Dict[str, Any]] = None,
                                 dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        
        index_id = knowledge_base_id
        
        retrieve_request = RetrieveRequest(index_id=index_id, query=query_text)
        
        if filters:
            logger.warning("Bailian Retrieve API does not support complex filters. Ignoring filters.")
        
        response_data = await self._async_bailian_call(self._client.retrieve_with_options,
            self.workspace_id, retrieve_request, {}, RuntimeOptions())
        
        result_list = response_data.get("Docs", [])

        mapped_results = []
        for item in result_list[:limit]:
            source = item.get('Source', {})
            mapped_results.append({
                "content": item.get('Content', ''),
                "score": 1.0, # Bailian Retrieve 不提供分数，设为 1.0 兼容
                "rerank_score": 1.0, 
                "source": source.get('DocumentName', '未知文件'),
                "docId": source.get('DocumentId', 'N/A'), # FileId
                "chunkId": item.get('ChunkId', 'N/A'),
                "user_data": source.get('CustomField', {}) 
            })
        return mapped_results

    async def update_document_meta(self, 
                               knowledge_base_id: str, 
                               doc_id: str, 
                               meta_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """更新文档元数据。阿里百炼无直接 API，返回 Not Implemented。"""
        logger.warning(f"Update is not directly supported by Bailian Indexing API for doc_id {doc_id}.")
        return {
            "message": "Bailian Indexing API does not support direct metadata updates on existing files. Please re-ingest the document.",
            "status": "not_implemented",
        }