# services/bailian_rag_service.py

import json
import logging
import asyncio
import re
from typing import Dict, Any, List, Optional
import httpx 
import os 
from urllib.parse import quote
from alibabacloud_bailian20231229.client import Client as BailianClient
from Tea.model import TeaModel
from alibabacloud_tea_openapi.models import Config as BailianConfig 
from alibabacloud_bailian20231229.models import (ApplyFileUploadLeaseRequest, AddFileRequest, 
    RetrieveRequest, DeleteIndexDocumentRequest, SubmitIndexAddDocumentsJobRequest, ListChunksRequest
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
        index_id = knowledge_base_id
        
        # --- 拼接 doc_name 以包含 material_id ---
        material_id = meta.get('material_id') if meta and isinstance(meta, dict) else None
        # 构造 Bailian 侧实际使用的文件名
        bailian_doc_name = doc_name
        if material_id is not None:
             # 确保 material_id 是字符串，并拼接
            bailian_doc_name = f"{material_id}_{doc_name}" 
            logger.info(f"New Bailian Document Name with material_id: {bailian_doc_name}")
        
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

        lease_request = ApplyFileUploadLeaseRequest(file_name=bailian_doc_name, md_5=file_md5, size_in_bytes=file_size)
        lease_response = await self._async_bailian_call(self._client.apply_file_upload_lease_with_options,
            self.default_category_id, self.workspace_id, lease_request, {}, RuntimeOptions())

        lease_id = lease_response.get('FileUploadLeaseId')
        upload_param = lease_response.get('Param', {})
        pre_signed_url = upload_param.get('Url')
        # 1. 获取 Bailian/OSS 预设的 Header
        upload_headers_dict = {k: v for k, v in upload_param.get('Headers', {}).items()}
        
        # 2. 客户端上传文件内容到 presigned URL
        await self._upload_file_content_from_url(pre_signed_url, url, upload_headers_dict)
        
        # 3. 添加文件到类目
        # 构造 Tags 列表
        all_tags = []
        # 3.1 业务 Doc ID 作为第一个标签 (保留)
        sanitized_doc_tag = self._sanitize_tag(doc_id)
        if sanitized_doc_tag and sanitized_doc_tag != 'tag_empty':
            all_tags.append(sanitized_doc_tag)
        
        # 3.2 仅对 accessible_to 列表字段进行 Tagging (新增/修改)
        if meta and isinstance(meta, dict):
            # 获取 accessible_to 列表
            accessible_to_list = meta.get('accessible_to')
            
            if accessible_to_list and isinstance(accessible_to_list, list):
                for item in accessible_to_list:
                    # 将列表中的每个元素作为标签，并进行清洗
                    sanitized_tag = self._sanitize_tag(str(item))
                    if sanitized_tag and sanitized_tag != 'tag_empty':
                        all_tags.append(sanitized_tag)
        # 去重并去除 None/空字符串
        all_tags = list(set([t for t in all_tags if t and t != 'tag_empty']))
        
        # 实例化 AddFileRequest 并传入 tags
        add_file_request = AddFileRequest(
            lease_id=lease_id, 
            parser=self.parser_type, 
            category_id=self.default_category_id,
            tags=all_tags,
            original_file_url=url 
        )
        logger.info(f"Adding file {bailian_doc_name} with tags/metadata: {all_tags}")
        
        add_file_response = await self._async_bailian_call(self._client.add_file_with_options,
            self.workspace_id, add_file_request, {}, RuntimeOptions())
        
        file_id = add_file_response.get('FileId')
        
        # 5. 提交索引追加任务
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
                
                for doc_id in doc_id_list:
                    
                    sanitized_tag = self._sanitize_tag(str(doc_id))
                    
                    # 关键检查：确保清洗后的标签非空，避免传入 [""] 或 [null]
                    if sanitized_tag:
                        # 修正：恢复官方示例中的 JSON 字符串格式
                        tag_list = [sanitized_tag]
                        tag_filter = {
                            "tags": json.dumps(tag_list)
                        }
                        bailian_search_filters.append(tag_filter)
                    else:
                        logger.warning(f"Skipping empty tag generated from doc_id: {doc_id}")
                
                logger.info(f"Generated Bailian tags filters for Doc ID: {bailian_search_filters}")

            accessible_to_list = filters.get("accessible_to")

            if accessible_to_list and isinstance(accessible_to_list, list):
                logger.info(f"Mapping accessible_to filter for Bailian: {accessible_to_list}")
                
                # 对于权限过滤，我们希望文档匹配列表中的任一标签 (OR 关系)，例如 ['80', 'public']。
                # 百炼的 SearchFilters 是一个列表，其元素之间是 AND 关系。
                # 但是单个过滤器内部的 'tags' 字段，如果传入多个，则是 OR 关系。
                # ⚠️ 关键：百炼 SDK 要求 tags 字段值是一个 JSON 字符串，代表一个列表。
                
                sanitized_acl_tags = []
                for access_id in accessible_to_list:
                    sanitized_tag = self._sanitize_tag(str(access_id))
                    if sanitized_tag:
                        sanitized_acl_tags.append(sanitized_tag)
                
                if sanitized_acl_tags:
                    # 构造单个 Tag 过滤器，包含所有权限标签，实现 OR 关系
                    acl_filter = {
                        "tags": json.dumps(sanitized_acl_tags) # 将列表序列化为 JSON 字符串
                    }
                    bailian_search_filters.append(acl_filter)
                    logger.info(f"Generated Bailian ACL filter: {acl_filter}")

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
                {}, 
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
        我们假设 meta_updates 中包含了要更新的完整权限列表。
        
        注意：doc_id 在此方法中通常是 Bailian 的 File ID。
        """
        file_id = doc_id 
        
        # 期望 meta_updates 结构是 [{'key': 'accessible_to', 'value': ['user1', 'public']}]
        
        accessible_to_new_list = None
        
        # 1. 提取新的 accessible_to 列表
        for item in meta_updates:
            if isinstance(item, dict) and item.get('field_name') == 'accessible_to':
                accessible_to_new_list = item.get('field_value')
                break
        
        if accessible_to_new_list is None:
             # 如果没有可更新的 accessible_to 字段，则返回不支持其他元数据更新
            logger.error(f"Bailian only supports updating 'accessible_to' via file tags. Other metadata updates are not supported.")
            return {
                "message": "Bailian only supports updating 'accessible_to' via file tags. Other metadata updates are not supported.",
                "status": "not_implemented",
            }

        import re
        # 1. 正则表达式: 匹配所有不是字母、数字、_、- 的字符
        cleaned_file_key_tag = re.sub(r'[^a-zA-Z0-9_-]', '_', file_key)
        # 2. 确保第一个字符是字母或下划线
        if not re.match(r'^[a-zA-Z_]', cleaned_file_key_tag):
            cleaned_file_key_tag = 'doc_' + cleaned_file_key_tag
            
        # 2.2 初始化标签列表，将清洗后的 File Key 作为业务标识 Tag 放在第一位
        new_tags = [self._sanitize_tag(cleaned_file_key_tag)] 
        logger.info(f"Retaining business tag based on file_key: {cleaned_file_key_tag}")
        
        # 添加新的权限标签
        if accessible_to_new_list and isinstance(accessible_to_new_list, list):
            for item in accessible_to_new_list:
                sanitized_tag = self._sanitize_tag(str(item))
                if sanitized_tag and sanitized_tag != 'tag_empty':
                    new_tags.append(sanitized_tag)

        # 3. 调用更新 tags 的 API
        try:
            return await self.update_document_tags(
                file_id=file_id, 
                tags=list(set(new_tags)) # 再次去重
            )
        except Exception as e:
            return {
                "message": f"Failed to update file tags (permissions): {str(e)}",
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
