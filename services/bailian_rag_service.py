# services/bailian_rag_service.py

import json
import logging
import asyncio
import random
import re
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
import httpx 
import os 
from urllib.parse import quote
from alibabacloud_bailian20231229.client import Client as BailianClient
from Tea.model import TeaModel
from alibabacloud_tea_openapi.models import Config as BailianConfig 
from alibabacloud_bailian20231229.models import (ApplyFileUploadLeaseRequest, AddFileRequest, 
    RetrieveRequest, DeleteIndexDocumentRequest, SubmitIndexAddDocumentsJobRequest, ListChunksRequest
)
from alibabacloud_tea_openapi.exceptions._throttling import ThrottlingException
from alibabacloud_tea_util.models import RuntimeOptions

from core.config import settings 
from services.abstract_kb_service import AbstractKnowledgeBaseService

logger = logging.getLogger(__name__)

from alibabacloud_bailian20231229.models import (
    ApplyFileUploadLeaseRequest, AddFileRequest, RetrieveRequest, DeleteIndexDocumentRequest, 
    SubmitIndexAddDocumentsJobRequest, ListChunksRequest, ListFileRequest # <--- 新增 ListFileRequest
)

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

        request_id = response_dict.get('headers', {}).get('x-acs-request-id', 'N/A')
        
        if response_dict.get('statusCode') and response_dict.get('statusCode') >= 400:
            error_body = response_dict.get('body', {})
            error_msg = f"HTTP Error {response_dict.get('statusCode')}. Message: {error_body.get('Message', 'N/A')}"
            raise BailianRagException(response_dict.get('statusCode'), request_id, error_msg)
        
        body = response_dict.get('body', {})
        data = body.get('Data')
        
        # 成功路径 A: Body 中没有 Data
        if not data:
            if 'Message' in body and body.get('Message') != 'Success':
                # 这是一个业务错误，但 HTTP 状态码可能是 200，所以仍需检查
                raise BailianRagException(body.get('Code', 'N/A'), request_id, body.get('Message'))
            
            # 纯粹的成功返回，将 Request ID 包含在结果中
            return {"status": "success", "request_id": request_id} # <--- 修正点：使用提取的 request_id

        # 成功路径 B: 返回 Data
        # 确保 Data 字典中包含 Request ID (虽然通常 Request ID 在顶层，但加在这里可以保证)
        if isinstance(data, dict):
            data['RequestId'] = data.get('RequestId', request_id)
            
        return data

    async def _async_bailian_call(self, method, *args) -> Dict[str, Any]:
        """
        将同步的 Bailian SDK 调用包装成异步方法。
        
        Args:
            method: 要调用的 SDK 方法。
            *args: 传递给 SDK 方法的所有位置参数（不包括 RuntimeOptions）。
            runtime_options: 额外的运行时选项，用于覆盖默认值。
        """
        MAX_RETRIES = 5
        BASE_DELAY = 1.0  # 初始等待 1 秒

        # 1. 确保至少有一个参数 (RuntimeOptions)
        if not args:
            raise ValueError("Bailian SDK call requires at least one argument (RuntimeOptions).")
            
        # 2. 提取并覆盖 RuntimeOptions
        runtime_index = len(args) - 1
        original_runtime = args[runtime_index]

        options_dict = {
            'read_timeout': 30000, # 30秒
            'connect_timeout': 15000, # 15秒
        }
        
        # 这里我们使用原始的 RuntimeOptions 作为模板，并覆盖其值
        if isinstance(original_runtime, RuntimeOptions):
            # 将 RuntimeOptions 转换为字典，并用我们的定制值覆盖
            new_runtime_dict = original_runtime.to_map()
            new_runtime_dict.update(options_dict)
            custom_runtime = RuntimeOptions(**new_runtime_dict)
        else:
            # 如果最后一个参数不是 RuntimeOptions (理论上不应发生)，则抛出错误或使用默认
            custom_runtime = RuntimeOptions(**options_dict)
        
        # 3. 重建 *args，用 custom_runtime 替换原始的 RuntimeOptions
        final_args = list(args[:runtime_index])
        final_args.append(custom_runtime)

        loop = asyncio.get_event_loop()
        for attempt in range(MAX_RETRIES):
            try:
                response = await loop.run_in_executor(
                    None, 
                    lambda: method(*args, custom_runtime) 
                )
                return self._parse_bailian_response(response)
            
            except ThrottlingException as e:
                # 遇到限流错误
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5) 
                    logger.warning(f"Bailian API Throttled. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(delay)
                else:
                    # 最后一次重试失败，抛出异常
                    logger.error(f"Bailian API failed after {MAX_RETRIES} retries due to Throttling: {e}", exc_info=True)
                    raise BailianRagException(e.code, e.request_id, e.message)
            
            except Exception as e:
                # 捕获其他非 Throttling 错误，并重新包装/抛出
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
                        headers=headers, # 使用传入的 headers
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
        return cleaned_tag[:32] if cleaned_tag else 'tag_empty'

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
        将 material_id 拼接到 doc_name 前面。
        """

        # 清理 URL，移除查询参数
        from urllib.parse import urlparse, urlunparse
        parsed_url = urlparse(url)
        # 只保留 scheme, netloc, path，去除 params, query, fragment
        cleaned_oss_path_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))
        
        logger.info(f"Original URL: {url}")
        logger.info(f"Cleaned OSS Path URL for Bailian Check: {cleaned_oss_path_url}")
        
        index_id = knowledge_base_id
        final_doc_name = doc_name
        
        # 检查 doc_name 是否以 doc_type 结尾（忽略大小写和点）
        if doc_type and not final_doc_name.lower().endswith(f'.{doc_type}'.lower()):
            # 补全文件后缀，例如：'document' + '.pdf'
            final_doc_name = f"{final_doc_name}.{doc_type}"
            logger.info(f"Appended doc_type '{doc_type}' to doc_name. Final name: {final_doc_name}")
        
        # --- 拼接 doc_name 以包含 material_id ---
        material_id = meta.get('material_id') if meta and isinstance(meta, dict) else None
        # 构造 Bailian 侧实际使用的文件名
        bailian_doc_name = final_doc_name
        if material_id is not None:
             # 确保 material_id 是字符串，并拼接
            bailian_doc_name = f"{material_id}_{final_doc_name}" 
            logger.info(f"New Bailian Document Name with material_id: {bailian_doc_name}")
        
        # 1. 获取文件大小并申请租约
        file_size = 1024 # 简化处理，实际生产应尝试精确获取
        file_md5 = "ignored" 
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                head_resp = await client.head(cleaned_oss_path_url)
                head_resp.raise_for_status()
                file_size = int(head_resp.headers.get('content-length'))
        except Exception:
            logger.warning(f"Failed to get file size via HEAD request for {cleaned_oss_path_url}. Using default size 1024.")

        lease_request = ApplyFileUploadLeaseRequest(file_name=bailian_doc_name, md_5=file_md5, size_in_bytes=file_size)
        lease_response = await self._async_bailian_call(self._client.apply_file_upload_lease_with_options,
            self.default_category_id, self.workspace_id, lease_request, RuntimeOptions())

        lease_id = lease_response.get('FileUploadLeaseId')
        upload_param = lease_response.get('Param', {})
        pre_signed_url = upload_param.get('Url')
        # 1. 获取 Bailian/OSS 预设的 Header
        upload_headers_dict = {k: v for k, v in upload_param.get('Headers', {}).items()}
        
        # 2. 客户端上传文件内容到 presigned URL
        await self._upload_file_content_from_url(pre_signed_url, url, upload_headers_dict)
        
        # 4. 调用公共方法完成 AddFileRequest 和索引提交
        return await self._process_import_job(
            lease_id=lease_id,
            bailian_doc_name=bailian_doc_name,
            knowledge_base_id=knowledge_base_id,
            doc_id=doc_id,
            meta=meta,
            original_file_url=cleaned_oss_path_url # 传入清理后的 URL
        )

    async def import_document_text(self, 
                                text_content: str,
                                doc_name: str, 
                                knowledge_base_id: str, # Index ID
                                doc_id: str, 
                                doc_type: str, # 应该是 'txt'
                                meta: Optional[Dict[str, Any]] = None
                                ) -> Dict[str, Any]:
        """
        将文本内容作为文件上传到百炼知识库 (PaperCut)。
        """
        
        # 1. 准备文本数据和大小
        text_bytes = text_content.encode('utf-8')
        file_size = len(text_bytes)
        file_md5 = "ignored" 
        
        # 确保 doc_type 为 txt/md，并补全文件名
        doc_type = 'txt' 
        final_doc_name = doc_name
        if not final_doc_name.lower().endswith(f'.{doc_type}'):
            final_doc_name = f"{final_doc_name}.{doc_type}"
        
        # 拼接 material_id 
        material_id = meta.get('material_id') if meta and isinstance(meta, dict) else None
        bailian_doc_name = final_doc_name
        if material_id is not None:
            bailian_doc_name = f"{material_id}_{final_doc_name}" 
            logger.info(f"Bailian Text Document Name: {bailian_doc_name}")

        # 2. 申请租约
        lease_request = ApplyFileUploadLeaseRequest(file_name=bailian_doc_name, md_5=file_md5, size_in_bytes=file_size)
        lease_response = await self._async_bailian_call(self._client.apply_file_upload_lease_with_options,
            self.default_category_id, self.workspace_id, lease_request, RuntimeOptions())

        lease_id = lease_response.get('FileUploadLeaseId')
        upload_param = lease_response.get('Param', {})
        pre_signed_url = upload_param.get('Url')
        upload_headers_dict = {k: v for k, v in upload_param.get('Headers', {}).items()}
        
        # 3. 直接上传文本内容到 presigned URL
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                upload_response = await client.request(
                    method="PUT", 
                    url=pre_signed_url,
                    headers=upload_headers_dict, 
                    content=text_bytes
                )
                upload_response.raise_for_status()
                logger.info("Text content successfully uploaded to Bailian temporary storage.")
            except Exception as e:
                raise BailianRagException(1000031, "upload_error", f"Unexpected error during text upload: {str(e)}")

        # 4. 调用公共方法完成 AddFileRequest 和索引提交
        return await self._process_import_job(
            lease_id=lease_id,
            bailian_doc_name=bailian_doc_name,
            knowledge_base_id=knowledge_base_id,
            doc_id=doc_id,
            meta=meta,
            original_file_url="" # 纯文本上传，不提供源 URL
        )

    async def _process_import_job(self, 
                                lease_id: str, 
                                bailian_doc_name: str,
                                knowledge_base_id: str, 
                                doc_id: str, 
                                meta: Optional[Dict[str, Any]] = None,
                                original_file_url: str = "") -> Dict[str, Any]:
        """
        处理文件上传后的后续步骤：AddFileRequest 和 SubmitIndexAddDocumentsJobRequest。
        
        Args:
            lease_id: 申请文件上传租约返回的 ID。
            bailian_doc_name: 百炼实际使用的文件名（已包含 material_id 和后缀）。
            knowledge_base_id: 索引 ID。
            doc_id: 业务文档 ID。
            meta: 业务元数据字典。
            original_file_url: 文件的源 URL (如果存在，用于 AddFileRequest)。
        """
        index_id = knowledge_base_id

        # 3. 构造 Tags 列表 (从原方法中抽取，确保 meta 是字典)
        all_tags = []
        sanitized_doc_tag = self._sanitize_tag(doc_id)
        if sanitized_doc_tag and sanitized_doc_tag != 'tag_empty':
            all_tags.append(sanitized_doc_tag)
        
        FIELDS_TO_TAG = [
            'accessible_to', 'level_list', 'clazz', 'exam', 'subject', 'type'
        ]
        
        if meta and isinstance(meta, dict):
            for field in FIELDS_TO_TAG:
                value = meta.get(field)
                if value is None: continue
                value_list = value if isinstance(value, list) else [value] 
                for item in value_list:
                    item_str = str(item)
                    if not item_str: continue
                    final_tag = self._sanitize_tag(f"{field}_{item_str}")
                    all_tags.append(final_tag)
                        
        all_tags = list(set([t for t in all_tags if t and t != 'tag_empty']))
        
        # 实例化 AddFileRequest 并传入 tags
        add_file_request = AddFileRequest(
            lease_id=lease_id, 
            parser=self.parser_type, 
            category_id=self.default_category_id,
            tags=all_tags,
            original_file_url=original_file_url 
        )
        logger.info(f"Adding file {bailian_doc_name} with tags/metadata: {all_tags}, URL: {original_file_url[:50]}...")
        
        add_file_response = await self._async_bailian_call(self._client.add_file_with_options,
            self.workspace_id, add_file_request, RuntimeOptions())
        
        file_id = add_file_response.get('FileId')
        
        # 5. 提交索引追加任务
        submit_request = SubmitIndexAddDocumentsJobRequest(
            index_id=index_id, 
            document_ids=[file_id], 
            source_type='DATA_CENTER_FILE'
        )
        submit_response = await self._async_bailian_call(self._client.submit_index_add_documents_job_with_options,
            self.workspace_id, submit_request, RuntimeOptions())
        
        job_id = submit_response.get('Id')
        
        return {
            "kb_doc_id": file_id, # 百炼返回的 File ID
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
            self.workspace_id, delete_request)
        
        return {"status": "success", "message": f"Delete request submitted for file ID {file_id}", "request_id": response.get('RequestId')}

    async def delete_file_permanently(self, file_id: str) -> Dict[str, Any]:
        """
        调用 DeleteFile 接口，永久删除应用数据中的文件。
        """
        
        logger.warning(f"Attempting to permanently delete file from data center: {file_id}")
        
        try:
            # 实际调用时，请求体应该传递 None 或空字典
            response = await self._async_bailian_call(
                self._client.delete_file_with_options,
                file_id,           # 路径参数 1: FileId
                self.workspace_id, # 路径参数 2: WorkspaceId
                None,              # 请求体: None (API 无请求体)
            )
            
            # DeleteFile 成功返回的 Data 中包含 FileId
            deleted_file_id = response.get('FileId')
            
            return {
                "status": "success", 
                "message": f"Successfully deleted file {deleted_file_id} from data center.", 
                "file_id": deleted_file_id, 
                "request_id": response.get('RequestId')
            }
        
        except BailianRagException as e:
            # 捕获并记录 API 错误，例如文件状态不支持删除等
            logger.error(f"Bailian API Error when deleting file {file_id}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during permanent file deletion for {file_id}: {e}", exc_info=True)
            raise BailianRagException(1000033, "delete_file_error", f"Unexpected error: {str(e)}")
        
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
            
            # 复制一份过滤器，以便在处理过程中移除已识别的字段
            remaining_filters = filters.copy()
            
            # --- 适配 Doc ID 列表过滤 ---
            doc_id_list = remaining_filters.pop("doc_id_list", None)
            
            if doc_id_list and isinstance(doc_id_list, list):
                logger.info(f"Mapping doc_id_list filter for Bailian: {doc_id_list}")
                
                # Doc ID 过滤: 每一个 Doc ID 都是一个 Tag
                for doc_id in doc_id_list:
                    sanitized_tag = self._sanitize_tag(str(doc_id))
                    if sanitized_tag:
                        # 生成 Tag 过滤器，Doc ID 之间是 AND 关系 (百炼默认行为)
                        tag_list = [sanitized_tag]
                        tag_filter = {
                            "tags": json.dumps(tag_list)
                        }
                        bailian_search_filters.append(tag_filter)
                
                logger.info(f"Generated Bailian tags filters for Doc ID: {bailian_search_filters}")

            # --- 适配 accessible_to 权限过滤 (OR 关系) ---
            accessible_to_list = remaining_filters.pop("accessible_to", None)

            if accessible_to_list and isinstance(accessible_to_list, list):
                logger.info(f"Mapping accessible_to filter for Bailian: {accessible_to_list}")
                
                sanitized_acl_tags = []
                # 权限 Tag 格式也是 key_value 模式：accessible_to_value
                for access_id in accessible_to_list:
                    # 关键：在这里将权限值格式化为 key_value Tag
                    final_tag = self._sanitize_tag(f"accessible_to_{str(access_id)}")
                    if final_tag:
                        sanitized_acl_tags.append(final_tag)
                
                if sanitized_acl_tags:
                    # 构造单个 Tag 过滤器，包含所有权限标签，实现 OR 关系
                    acl_filter = {
                        "tags": json.dumps(sanitized_acl_tags) # 将列表序列化为 JSON 字符串
                    }
                    bailian_search_filters.append(acl_filter)
                    logger.info(f"Generated Bailian ACL filter: {acl_filter}")

            # --- 适配剩余的 K/V 元数据过滤器 (AND/OR Tag 过滤) ---
            
            # 遍历剩余的 filters 字典。这些字段都是 clazz, subject, levelList 等业务元数据。
            if remaining_filters:
                logger.info(f"Mapping remaining metadata filters for Bailian: {remaining_filters}")
                
                for field, value in remaining_filters.items():
                    # 确定要过滤的值列表 (如果单值，转换为单元素列表)
                    if value is None:
                        continue
                    elif isinstance(value, list):
                        filter_values = value
                    else:
                        filter_values = [value]
                    
                    # 构造 Tag 列表：key_value 格式
                    field_tags = []
                    for val in filter_values:
                        item_str = str(val)
                        if not item_str: continue

                        # 关键：将 Field Key 和 Value 格式化为 key_value Tag
                        final_tag = self._sanitize_tag(f"{field}_{item_str}")
                        field_tags.append(final_tag)
                        
                    if field_tags:
                        # 构造 Tag 过滤器。对于单个字段的多个值，通常我们希望是 OR 关系。
                        # 例如 Subject: ['Biology', 'Physics'] 应该是匹配 Biology_Tag OR Physics_Tag。
                        # 百炼的单个 'tags' 字段列表实现了这个 OR 逻辑。
                        meta_filter = {
                            "tags": json.dumps(field_tags)
                        }
                        # 由于 bailian_search_filters 列表之间是 AND 关系，这保证了
                        # 多个不同字段（如 clazz 和 subject）之间的 AND 关系。
                        bailian_search_filters.append(meta_filter)
                        logger.info(f"Generated meta filter for {field}: {meta_filter}")


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
            self.workspace_id, retrieve_request, RuntimeOptions())
        
        result_list = response_data.get("Nodes", [])

        # 5. 结果映射 (使用 Score 字段，不再固定为 1.0)
        mapped_results = []
        for item in result_list[:limit]:
            source = item.get('Source', {})
            metadata = item.get('Metadata', {})
            
            # 从 Metadata 中提取 docId (FileId)
            doc_id_from_metadata = metadata.get('doc_id') or source.get('DocumentId', 'N/A')
            document_name = metadata.get('doc_name', '未知文件')
            
            # --- 改造点 2: 从 DocumentName 中解析 material_id ---
            material_id = None
            try:
                # 尝试从 DocumentName 中解析 {material_id}_{file_name} 格式
                match = re.match(r'^(\d+)_', document_name)
                if match:
                    material_id = match.group(1)
                    # 可以在这里移除前缀，只保留原始文件名
                    document_name = document_name[len(match.group(0)):]
            except Exception:
                logger.warning(f"Could not parse material_id from DocumentName: {document_name}")
            
            # 将 Metadata 字段映射到 user_data
            user_data = metadata.copy()
            user_data['material_id'] = material_id
            
            mapped_results.append({
                "content": item.get('Text', ''),
                "score": item.get('Score', 0.0), # 读取真实的 Score 字段，默认 0.0
                "rerank_score": item.get('Score', 0.0), 
                "source": document_name,
                "docId": doc_id_from_metadata, 
                "chunkId": metadata.get('_id', 'N/A'),
                "user_data": user_data,
                "material_id": material_id
            })
        return mapped_results


    async def update_document_tags(self, 
                                   file_id: str, 
                                   tags: List[str]) -> Dict[str, Any]:
        """
        调用百炼 UpdateFileTag API 更新指定文件的标签。
        
        Args:
            file_id: 百炼的文件 ID (对应业务 doc_id)。
            tags: 要更新的完整标签列表。
        """
        from alibabacloud_bailian20231229.models import UpdateFileTagRequest

        # 1. 对传入的标签进行清洗
        sanitized_tags = []
        for tag in tags:
            sanitized_tag = self._sanitize_tag(str(tag))
            if sanitized_tag and sanitized_tag != 'tag_empty':
                sanitized_tags.append(sanitized_tag)
        
        # 2. 去重并限制数量 (百炼单个文件最多支持 10 个标签)
        final_tags = list(set(sanitized_tags))[:10]
        
        if not final_tags:
            logger.warning(f"No valid tags generated for file_id {file_id}. Tags will be cleared.")
            
        # 1. 构造请求体 (Body) DTO: 
        #    UpdateFileTagRequest 只需要 tags，因为它只代表 HTTP Body 内容。
        update_request = UpdateFileTagRequest(
            tags=final_tags 
        )
        
        logger.info(f"Updating tags for file {file_id}. Final tags: {final_tags}")
        
        try:
            # 2. 调用 SDK 方法：File ID 作为路径参数传入
            #    SDK 的方法签名通常是: update_file_tag_with_options(WorkspaceId, FileId, UpdateFileTagRequest, ...)
            response = await self._async_bailian_call(
                self._client.update_file_tag_with_options,
                self.workspace_id,      # 路径参数 1: WorkspaceId
                file_id,                # 路径参数 2: FileId (您的 SDK 封装会在这里找到它)
                update_request,         # 请求体: UpdateFileTagRequest
                RuntimeOptions()
            )
            
            logger.info(f"Successfully submitted tag update for file ID {file_id}.")
            
            return {"status": "success", "message": "File tags updated successfully.", "request_id": response.get('RequestId'), "file_id": file_id}

        except Exception as e:
            logger.error(f"Failed to update file tags for {file_id}: {e}", exc_info=True)
            if isinstance(e, BailianRagException):
                raise e
            raise BailianRagException(1000032, "tag_update_error", f"SDK error during tag update: {str(e)}")

    async def update_document_meta(self, 
                               knowledge_base_id: str, 
                               doc_id: str, 
                               file_key: str,
                               meta_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        更新文档元数据。针对百炼，我们将其适配为更新文件标签 (Tags)。
        此方法将 meta_updates 中所有支持的字段（权限和四属性）格式化为 key_value Tag，
        并执行完整的覆盖式 Tag 更新。
        
        注意：doc_id 在此方法中是 Bailian 的 File ID。
        """
        file_id = doc_id 
        
        # --- 1. 业务标识 Tag (File Key 清洗后的格式) ---
        import re
        # 匹配所有不是字母、数字、_、- 的字符，并替换为 _
        cleaned_file_key_tag_raw = re.sub(r'[^a-zA-Z0-9_-]', '_', file_key)
        # 确保第一个字符是字母或下划线
        if not re.match(r'^[a-zA-Z_]', cleaned_file_key_tag_raw):
            cleaned_file_key_tag_raw = 'doc_' + cleaned_file_key_tag_raw
        
        # 确保 Doc ID Tag（这是用于检索的原始 File Key 标识）和 File Key Tag 一致
        business_id_tag = self._sanitize_tag(cleaned_file_key_tag_raw)
        
        # 初始化标签列表，必须保留业务标识 Tag
        # new_tags = [self._sanitize_tag(cleaned_file_key_tag)] # 原始代码有误，应使用 business_id_tag
        new_tags = [business_id_tag] 
        
        logger.info(f"Retaining business identifier tag: {business_id_tag}")

        
        # --- 2. 收集并格式化所有要更新的业务元数据 Tag ---
        
        # 定义需要转化为 key_value Tag 的字段集合
        FIELDS_TO_TAG = [
            'accessible_to', 'level_list', 'clazz', 'exam', 'subject', 'type'
        ]
        
        # 收集更新中的所有 Tag
        updated_meta_tags = []
        
        # 遍历 Spring 传入的 meta_updates 列表
        for item in meta_updates:
            field = item.get('field_name')
            value = item.get('field_value')
            
            if field in FIELDS_TO_TAG:
                
                # 确定要处理的值列表 (如果是单值，转换为单元素列表)
                if value is None:
                    continue
                elif isinstance(value, list):
                    value_list = value
                else:
                    value_list = [value] 
                
                for item_value in value_list:
                    item_str = str(item_value)
                    if not item_str:
                        continue
                        
                    # 格式化 Field Key 和 Value
                    sanitized_field_key = self._sanitize_tag(field) 
                    sanitized_item_value = self._sanitize_tag(item_str)
                    
                    if sanitized_item_value and sanitized_field_key:
                        # 构建最终的 Tag: key_value
                        final_tag = f"{sanitized_field_key}_{sanitized_item_value}"
                        updated_meta_tags.append(final_tag)

        # 3. 合并所有 Tag 并去重
        # new_tags 现在包含：[业务标识 Tag] + [所有格式化的元数据 Tag]
        new_tags.extend(updated_meta_tags)
        
        final_tags_to_apply = list(set(new_tags))
        
        # 4. 调用更新 tags 的 API
        try:
            logger.info(f"Updating tags for file {file_id}. Tags to be applied: {final_tags_to_apply}")
            return await self.update_document_tags(
                file_id=file_id, 
                tags=final_tags_to_apply # 执行完整的覆盖式更新
            )
        except Exception as e:
            logger.error(f"Failed to update file tags (metadata update failed): {e}", exc_info=True)
            return {
                "message": f"Failed to update file tags (metadata update): {str(e)}",
                "status": "error",
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
        
    async def list_files_iterator(self, category_id: str, max_results: int = 100) -> AsyncGenerator[Tuple[List[Dict[str, Any]], Optional[str], int], None]:
        """
        异步生成器：分页循环调用 ListFile 接口，每次 yield 一页的文件列表。
        
        Yields:
            Tuple[List[Dict[str, Any]], Optional[str], int]: (文件列表, 下一页Token, 页码)
        """
        from alibabacloud_bailian20231229.models import ListFileRequest
        
        next_token: Optional[str] = None
        page_count = 0
        
        # 确保 max_results 在 [1, 200] 范围内
        page_size = min(max_results, 200)
        
        while True:
            page_count += 1
            logger.info(f"Listing files for category '{category_id}', Page: {page_count}, MaxResults: {page_size}, NextToken: {next_token}")
            
            list_request = ListFileRequest(
                category_id=category_id,
                max_results=page_size,
                next_token=next_token
            )

            try:
                # 假设 _async_bailian_call 已经更新了更长的超时时间
                response_data = await self._async_bailian_call(
                    self._client.list_file_with_options,
                    self.workspace_id,
                    list_request,
                )
            except Exception as e:
                logger.error(f"Failed to list files on page {page_count}: {e}")
                # 如果查询失败，终止迭代器
                return 
            
            file_list = response_data.get('FileList', [])
            has_next = response_data.get('HasNext', False)
            next_token = response_data.get('NextToken')
            total_count = response_data.get('TotalCount') # 记录总数，以便上层追踪进度

            # 立即返回当前页的文件列表、下一页的 token 和页码
            yield file_list, next_token, page_count, total_count
            
            if not has_next or not next_token:
                logger.info("Finished listing files.")
                break
                
            # 限流：API 限频 5 次/秒，等待 0.25 秒
            await asyncio.sleep(0.25)
