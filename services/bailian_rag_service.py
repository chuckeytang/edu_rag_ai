# services/bailian_rag_service.py

import json
import logging
import asyncio
import re
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


    def _sanitize_tag(self, tag: str) -> str:
        """
        清除标签中的非法字符，将其替换为下划线。
        通常，阿里云标签只允许字母、数字和下划线。
        """
        # 移除所有非字母、数字、下划线的字符。特别是冒号、点号、连字符等。
        # 保持中文。这里我们假设只保留字母、数字、中文和下划线。
        # 假设业务 Doc ID/File Key 中的连字符 '-' 是允许的，但为安全起见，这里统一清理。
        
        # 将所有非法字符替换为下划线
        cleaned_tag = re.sub(r'[^\w\u4e00-\u9fa5]+', '_', tag).strip('_')
        
        # 确保标签不为空且不超过长度限制（假设不超过 32 个字符）
        return cleaned_tag if cleaned_tag else 'tag_empty'

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
        
        # 3. 添加文件到类目 (核心：注入 doc_id 作为 tags)
        
        # 构造 Tags 列表，将业务 doc_id 作为第一个标签
        all_tags = [self._sanitize_tag(doc_id)]
        if meta and isinstance(meta, dict):
            for key, value in meta.items():
                if isinstance(value, list):
                    # 如果值是列表，将列表中的每个元素作为标签，并进行清洗
                    all_tags.extend([self._sanitize_tag(str(v)) for v in value])
                elif isinstance(value, (str, int, float)):
                    # 如果值是单个值，格式化为 key_value 形式作为标签，并进行清洗
                    tag_value = f"{key}_{value}" 
                    all_tags.append(self._sanitize_tag(tag_value))
        # 去重并去除 None/空字符串
        all_tags = list(set([t for t in all_tags if t and t != 'tag_empty']))
        
        # 实例化 AddFileRequest 并传入 tags
        add_file_request = AddFileRequest(
            lease_id=lease_id, 
            parser=self.parser_type, 
            category_id=self.default_category_id,
            tags=all_tags
        )
        logger.info(f"Adding file {doc_name} with tags/metadata: {all_tags}")
        
        add_file_response = await self._async_bailian_call(self._client.add_file_with_options,
            self.workspace_id, add_file_request, {}, RuntimeOptions())
        
        file_id = add_file_response.get('FileId')
        
        # 4. 提交索引追加任务
        # SubmitIndexAddDocumentsJobRequest 不包含元数据字段，仅传入文件 ID
        submit_request = SubmitIndexAddDocumentsJobRequest(
            index_id=index_id, 
            document_ids=[file_id], 
            source_type='DATA_CENTER_FILE'
        )
        submit_response = await self._async_bailian_call(self._client.submit_index_add_documents_job_with_options,
            self.workspace_id, submit_request, {}, RuntimeOptions())
        
        job_id = submit_response.get('Id')
        
        return {
            "file_id": file_id, # 百炼返回的 File ID
            "business_doc_id": doc_id, # 业务传入的 Doc ID
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
        
        bailian_search_filters = []
        
        # 1. 过滤器适配逻辑：将上层业务 filter 转换为 Bailian SearchFilters
        if filters and isinstance(filters, dict):
            
            # --- 适配 Doc ID 列表过滤 (业务最常见需求) ---
            # 传入格式示例: {"doc_id_list": ["id1", "id2"]}
            doc_id_list = filters.get("doc_id_list")
            
            if doc_id_list and isinstance(doc_id_list, list):
                logger.info(f"Mapping doc_id_list filter for Bailian: {doc_id_list}")
                
                # 核心修正：必须对传入的 doc_id 进行清洗，以匹配存储时的标签格式
                for doc_id in doc_id_list:
                    
                    # 关键：对传入的 doc_id 进行清洗，确保与存储时的标签格式一致
                    sanitized_tag = self._sanitize_tag(str(doc_id)) 
                    
                    # 实现逻辑 AND 过滤（文档必须包含所有标签）
                    tag_filter = {
                        "tags": [sanitized_tag]  # 标签必须是 JSON 字符串列表
                    }
                    bailian_search_filters.append(tag_filter)
                
                logger.info(f"Generated Bailian tags filters for Doc ID: {bailian_search_filters}")

            else:
                # 如果传入的不是 doc_id_list 而是其他复杂的 K/V 结构，
                # 假设它已经是简单的 Bailian 子分组结构，直接作为唯一分组传入。
                # ⚠️ 警告：这部分适配存在风险，因为它依赖于业务层传入的 K/V 键名与百炼索引字段名一致。
                if filters.get("op") is None: # 排除 Volcano 复杂的 op/conds 结构
                    bailian_search_filters.append(filters)
                    logger.warning("Treating complex filter as a single Bailian SearchFilter subgroup.")
                else:
                    logger.warning(f"Complex Volcano-style filter structure detected: {json.dumps(filters)}. Ignoring for Bailian API.")

        # 2. 构造 Retrieve 请求体
        retrieve_request = RetrieveRequest(
            index_id=index_id, 
            query=query_text,
            enable_reranking=rerank_switch, 
            dense_similarity_top_k=limit,
            search_filters=bailian_search_filters if bailian_search_filters else None # 传入过滤器
        )
        
        # 3. 如果成功生成了 SearchFilters，则添加到 Request 中
        if bailian_search_filters:
            retrieve_request.search_filters = bailian_search_filters
            logger.info(f"Final Bailian SearchFilters: {json.dumps(bailian_search_filters)}")
        
        # 4. 调用 API
        response_data = await self._async_bailian_call(self._client.retrieve_with_options,
            self.workspace_id, retrieve_request, {}, RuntimeOptions())
        
        result_list = response_data.get("Docs", [])

        # 5. 结果映射 (使用 Score 字段，不再固定为 1.0)
        mapped_results = []
        for item in result_list[:limit]:
            source = item.get('Source', {})
            metadata = item.get('Metadata', {})
            
            # 从 Metadata 中提取 docId (FileId)
            doc_id_from_metadata = metadata.get('doc_id') or source.get('DocumentId', 'N/A')
            
            # 将 Metadata 字段映射到 user_data
            user_data = metadata.copy()
            
            mapped_results.append({
                "content": item.get('Content', ''),
                "score": item.get('Score', 0.0), # 读取真实的 Score 字段，默认 0.0
                "rerank_score": item.get('Score', 0.0), 
                "source": source.get('DocumentName', '未知文件'),
                "docId": doc_id_from_metadata, 
                "chunkId": item.get('ChunkId', 'N/A'),
                "user_data": user_data 
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

    async def list_knowledge_points(self, 
                                    knowledge_base_id: str,
                                    doc_ids: Optional[List[str]] = None, # 接收 doc_id 列表
                                    limit: int = 100,
                                    offset: int = 0) -> List[Dict[str, Any]]:
        """
        实现 AbstractKnowledgeBaseService.list_knowledge_points，调用阿里百炼 ListChunks 接口。
        
        注意：
        1. Bailian ListChunks 接口不支持 doc_id 列表，只支持单个 FileId 过滤。
        2. doc_id 必须是 Bailian 的 FileId。
        """
        index_id = knowledge_base_id
        
        # 1. 确定 FileId (ListChunks 只支持单个 FileId)
        file_id_filter = None
        if doc_ids and len(doc_ids) > 0:
            # ⚠️ 假设传入的 doc_ids 已经是 Bailian 的 FileId，并且我们只取第一个进行过滤
            file_id_filter = doc_ids[0]
            logger.warning(f"Bailian ListChunks only supports filtering by one FileId. Using the first ID: {file_id_filter}")
        
        page_num = offset // limit + 1
        page_size = min(limit, 100) # Bailian PageSize 最大为 100

        list_chunks_request = ListChunksRequest(
            index_id=index_id,
            # FileId 是文档搜索类知识库的必要过滤条件
            file_id=file_id_filter, 
            page_num=page_num,
            page_size=page_size,
            # Filed 字段是旧版本 SDK 字段，不需要传入
        )
        
        logger.info(f"Listing chunks for Index '{index_id}' with FileId: {file_id_filter}, PageNum: {page_num}")
        
        try:
            response_data = await self._async_bailian_call(
                self._client.list_chunks_with_options,
                self.workspace_id,
                list_chunks_request,
                {},
                RuntimeOptions()
            )
            
            result_list = response_data.get("Nodes", [])
            
            mapped_results = []
            for item in result_list:
                metadata = item.get('Metadata', {})
                
                # 兼容 Volcano 的返回字段
                mapped_results.append({
                    "content": item.get('Text', item.get('Content')), # Text 是切片内容
                    "point_id": metadata.get('nid', 'N/A'), # nid 是切片 ID
                    "doc_id": metadata.get('doc_id', 'N/A'), # doc_id 是 FileId
                    "doc_name": metadata.get('doc_name', 'N/A'),
                    "source": metadata.get('doc_name', 'N/A'),
                    "metadata": metadata, # 返回完整的元数据字典
                    "text_snippet": item.get('Text', '')[:500] + "..."
                })

            return mapped_results

        except Exception as e:
            logger.error(f"Failed to list chunks from Bailian KB: {e}", exc_info=True)
            # 为了让上层 DEBUG 接口能捕获到错误，这里重新抛出
            raise
