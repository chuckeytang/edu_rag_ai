import json
import logging
import threading
from typing import Dict, Any, List, Optional
import requests
import httpx # 用于异步调用

from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.base.Service import Service
from volcengine.ServiceInfo import ServiceInfo
from volcengine.auth.SignerV4 import SignerV4
from volcengine.base.Request import Request

from core.config import settings
from services.abstract_kb_service import AbstractKnowledgeBaseService

logger = logging.getLogger(__name__)

class VolcanoEngineRagException(Exception):
    def __init__(self, code, request_id, message=None):
        self.code = code
        self.request_id = request_id
        self.message = f"{message}, code:{self.code}，request_id:{self.request_id}"
    def __str__(self):
        return self.message
    
# 继承抽象接口
class VolcanoEngineRagService(Service, AbstractKnowledgeBaseService):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(VolcanoEngineRagService, "_instance"):
            with VolcanoEngineRagService._instance_lock:
                if not hasattr(VolcanoEngineRagService, "_instance"):
                    VolcanoEngineRagService._instance = object.__new__(cls)
        return VolcanoEngineRagService._instance

    def __init__(self, ak="", sk="", sts_token=""):
        host = settings.VOLCANO_ENGINE_RAG_API_URL.replace("https://", "").replace("http://", "")
        region = settings.VOLCANO_ENGINE_REGION
        scheme = "https"
        connection_timeout = 30
        socket_timeout = 30

        self.service_info = VolcanoEngineRagService.get_service_info(host, region, scheme, connection_timeout, socket_timeout)
        self.api_info = VolcanoEngineRagService.get_api_info()
        super(VolcanoEngineRagService, self).__init__(self.service_info, self.api_info)
        
        self.set_ak(settings.VOLCANO_ENGINE_AK_ID)
        self.set_sk(settings.VOLCANO_ENGINE_SK_SECRET)
        if sts_token:
            self.set_session_token(session_token=sts_token)
        
        logger.info("Volcano Engine RAG service initialized. Now conforming to AbstractKnowledgeBaseService interface.")

    @staticmethod
    def get_service_info(host, region, scheme, connection_timeout, socket_timeout):
        service_info = ServiceInfo(host, {"Host": host},
                                   Credentials('', '', 'air', region), connection_timeout, socket_timeout,
                                   scheme=scheme)
        return service_info

    @staticmethod
    def get_api_info():
        api_info = {
            "SearchKnowledge": ApiInfo("POST", "/api/knowledge/collection/search_knowledge", {}, {}, {'Accept': 'application/json', 'Content-Type': 'application/json'}),
            "DeleteDocument": ApiInfo("POST", "/api/knowledge/collection/delete_document", {}, {}, {'Accept': 'application/json', 'Content-Type': 'application/json'}),
            "ImportDocumentUrl": ApiInfo("POST", "/api/knowledge/doc/add", {}, {}, {'Accept': 'application/json', 'Content-Type': 'application/json'}),
            "Ping": ApiInfo("GET", "/api/knowledge/ping", {}, {}, {'Accept': 'application/json', 'Content-Type': 'application/json'}),
            "UpdateDocumentMeta": ApiInfo("POST", "/api/knowledge/doc/update_meta", {}, {}, {'Accept': 'application/json', 'Content-Type': 'application/json'}),
            "ListKnowledgePoints": ApiInfo("POST", "/api/knowledge/point/list", {}, {}, {'Accept': 'application/json', 'Content-Type': 'application/json'}),
        }
        return api_info

    def _adapt_meta_to_volcano_list(self, generic_meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将业务层传递的通用元数据字典转换为火山引擎 API 期望的列表格式。
        通用格式: {'key': 'value'} 或 {'key': ['v1', 'v2']}
        火山格式: [{'field_name': 'key', 'field_type': 'string'/'list<string>', 'field_value': value}]
        """
        if not generic_meta:
            return []

        volcano_meta_list = []
        for key, value in generic_meta.items():
            if value is None:
                continue

            field_type = "string"
            field_value = value

            if isinstance(value, list):
                # 假设列表字段需要明确声明为 list<string>
                field_type = "list<string>"
                field_value = value
            elif isinstance(value, (int, float, bool)):
                # 简单类型转换为字符串，以便兼容
                field_value = str(value)
            
            volcano_meta_list.append({
                "field_name": key,
                "field_type": field_type,
                "field_value": field_value
            })
            
        return volcano_meta_list
    
    # 抽象方法实现 1: 导入文档
    async def import_document_url(self, 
                                  url: str, 
                                  doc_name: str, 
                                  knowledge_base_id: str, 
                                  doc_id: str, 
                                  doc_type: str, 
                                  meta: Optional[Dict[str, Any]] = None 
                                  ) -> Dict[str, Any]:
        """
        火山引擎的文档导入接口实现，适配 AbstractKnowledgeBaseService。
        """
        logger.info(f"Importing document '{doc_name}' from URL to knowledge base '{knowledge_base_id}'.")
        
        # 核心改造点：将通用字典转换为火山引擎需要的列表格式
        volcano_meta_list = self._adapt_meta_to_volcano_list(meta)

        payload = {
            "collection_name": knowledge_base_id,
            "resource_id": knowledge_base_id,
            "add_type": "url",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_type": doc_type,
            "url": url,
        }
        
        if volcano_meta_list:
            payload["meta"] = volcano_meta_list
            logger.debug(f"Volcano meta payload: {payload['meta']}")
            
        try:
            return await self._async_make_request("ImportDocumentUrl", {}, payload)
        except Exception as e:
            logger.error(f"Error during API call with payload: {payload}", exc_info=True)
            raise e

    # 抽象方法实现 2: 删除文档
    async def delete_document(self, knowledge_base_id: str, doc_id: str) -> Dict[str, Any]:
        """
        火山引擎的文档删除接口实现，适配 AbstractKnowledgeBaseService。
        """
        logger.info(f"Deleting document '{doc_id}' from knowledge base '{knowledge_base_id}'.")
        payload = {
            "resource_id": knowledge_base_id,
            "doc_id": doc_id,
        }
        return await self._async_make_request("DeleteDocument", {}, payload)

    # 抽象方法实现 3: 检索文档 (核心)
    async def retrieve_documents(self,
                                 query_text: str,
                                 knowledge_base_id: str,
                                 limit: int = 10,
                                 rerank_switch: bool = True,
                                 filters: Optional[Dict[str, Any]] = None, # 适配抽象接口
                                 dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        火山引擎的检索接口实现，适配 AbstractKnowledgeBaseService，并支持 filters 传入。
        """
        
        logger.info(f"Retrieving for query '{query_text[:50]}...' from knowledge base '{knowledge_base_id}'.")
        
        # 1. 初始化 payload
        payload = {
            "resource_id": knowledge_base_id,
            "query": query_text,
            "limit": limit,
            "dense_weight": dense_weight,
            "post_processing": {
                "rerank_switch": rerank_switch
            },
            "query_param": {}
        }
        
        # 2. 如果存在 filters，将其作为 doc_filter 添加到 query_param 中 (恢复的逻辑)
        if filters:
            # filters 预期是 Volcano 引擎支持的结构，例如 {"op": "must", "field": "doc_id", "conds": ["id1", "id2"]}
            payload["query_param"]["doc_filter"] = filters
         
        # 3. 打印最终的 payload 以便调试
        logger.info(f"Final retrieval payload: {json.dumps(payload, indent=2)}")

        try:
            response_data = await self._async_make_request("SearchKnowledge", {}, payload)
            
            result_list = response_data.get("data", {}).get("result_list", [])
            logger.info(f"Successfully retrieved {len(result_list)} chunks.")

            mapped_results = []
            for item in result_list:
                doc_info = item.get('doc_info', {})
                user_data_str = doc_info.get('user_data', '{}')
                try:
                    user_data = json.loads(user_data_str)
                except json.JSONDecodeError:
                    user_data = {"error": "Invalid JSON in user_data"}

                # 保持 Volcano 原始的映射结构
                mapped_results.append({
                    "content": item.get('content'),
                    "score": item.get('score'),
                    "rerank_score": item.get('rerank_score'),
                    "source": doc_info.get('doc_name'),
                    "docId": doc_info.get('doc_id'),
                    "chunkId": item.get('id'),
                    "url": doc_info.get('url'),
                    "user_data": user_data
                })

            return mapped_results

        except Exception as e:
            logger.error(f"Failed to retrieve from Volcano Engine RAG: {e}")
            return []

    async def import_document_text(self, 
                                text_content: str,
                                doc_name: str, 
                                knowledge_base_id: str, # Index ID
                                doc_id: str, 
                                doc_type: str, # 应该是 'txt'
                                meta: Optional[Dict[str, Any]] = None
                                ) -> Dict[str, Any]:
        pass
    
    # 抽象方法实现 4: 更新元数据
    async def update_document_meta(self, 
                               knowledge_base_id: str, 
                               file_key: str,
                               doc_id: str, 
                               meta_updates: List[Dict[str, Any]]) -> Dict[str, Any]: 
        """
        调用火山引擎 /api/knowledge/doc/update_meta 接口更新文档的元数据，适配 AbstractKnowledgeBaseService。
        """
        logger.info(f"Updating meta for doc '{doc_id}' in KB '{knowledge_base_id}' with updates: {meta_updates}")

        payload = {
            "resource_id": knowledge_base_id,
            "doc_id": doc_id,
            "meta": meta_updates # 传入要更新的元数据列表
        }
        
        try:
            # _async_make_request 是原类中的私有方法，用于进行签名和请求
            return await self._async_make_request("UpdateDocumentMeta", {}, payload)
        except Exception as e:
            logger.error(f"Error during meta update for doc '{doc_id}': {e}", exc_info=True)
            raise e
    

    async def list_knowledge_points(self, 
                                    knowledge_base_id: str,
                                    doc_ids: Optional[List[str]] = None,
                                    limit: int = 100,
                                    offset: int = 0) -> List[Dict[str, Any]]:
        """
        调用 /api/knowledge/point/list 接口获取知识库中的切片列表及其完整元数据。
        """
        logger.info(f"Listing knowledge points for KB '{knowledge_base_id}'. Doc IDs: {doc_ids}, Limit: {limit}")

        payload = {
            "resource_id": knowledge_base_id,
            "limit": limit,
            "offset": offset
        }
        
        if doc_ids:
            payload["doc_ids"] = doc_ids # 筛选文档 ID
            
        try:
            response_data = await self._async_make_request("ListKnowledgePoints", {}, payload)
            
            point_list = response_data.get("data", {}).get("point_list", [])
            
            mapped_results = []
            for point in point_list:
                doc_info = point.get('doc_info', {})
                doc_meta_str = doc_info.get('doc_meta', '[]')
                
                # 尝试解析 doc_meta 字符串
                custom_metadata = {}
                try:
                    # doc_meta 是一个包含 {"field_name": ..., "field_value": ...} 列表的字符串
                    doc_meta_list = json.loads(doc_meta_str)
                    for item in doc_meta_list:
                        # 转换成 {field_name: field_value} 的字典格式
                        custom_metadata[item['field_name']] = item['field_value']
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode doc_meta for point {point.get('point_id')}")

                mapped_results.append({
                    "content": point.get('content'),
                    "point_id": point.get('point_id'),
                    "doc_id": doc_info.get('doc_id'),
                    "doc_name": doc_info.get('doc_name'),
                    "doc_type": doc_info.get('doc_type'),
                    "source": doc_info.get('source'),
                    "process_time": point.get('process_time'),
                    "metadata": custom_metadata
                })

            return mapped_results

        except Exception as e:
            logger.error(f"Failed to list knowledge points from Volcano Engine: {e}")
            return []
        
    async def _async_make_request(self, api: str, params: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
        if not (api in self.api_info):
            raise Exception("no such api")
        api_info = self.api_info[api]
        
        request = self.prepare_request(api_info, params)
        request.headers['Content-Type'] = 'application/json'
        request.headers['Traffic-Source'] = 'SDK'
        request.body = json.dumps(body)
        
        SignerV4.sign(request, self.service_info.credentials)
        
        url = request.build()
        
        async with httpx.AsyncClient(timeout=self.service_info.connection_timeout) as client:
            try:
                response = await client.request(
                    request.method,
                    url,
                    headers=request.headers,
                    content=request.body,
                )
                
                # 在这里，先检查非200状态码，并尝试解析错误详情
                if response.status_code >= 400:
                    try:
                        error_data = response.json()
                        # 打印火山引擎返回的 JSON 错误详情
                        logger.error(f"Volcano Engine API returned an error. Status Code: {response.status_code}, Response Body: {json.dumps(error_data, indent=2)}")
                        # 抛出包含详细信息的自定义异常
                        message = error_data.get("message", "Unknown API error")
                        request_id = error_data.get("request_id", "N/A")
                        raise VolcanoEngineRagException(error_data.get("code", "N/A"), request_id, message)
                    except json.JSONDecodeError:
                        # 如果响应不是 JSON 格式，就打印原始文本
                        logger.error(f"HTTP error occurred. Status Code: {response.status_code}, Response Body: {response.text}")
                        raise VolcanoEngineRagException(response.status_code, "httpx_error", response.text)

                # 如果状态码是200，则继续处理响应
                response_data = response.json()
                
                if response_data.get("code") != 0:
                    message = response_data.get("message", "Unknown error from Volcano Engine API")
                    raise VolcanoEngineRagException(response_data.get("code"), response_data.get("request_id"), message)
                    
                return response_data

            except Exception as e:
                # 捕获其他任何请求失败异常
                logger.error(f"Request failed: {e}", exc_info=True)
                raise VolcanoEngineRagException(1000029, "request_error", str(e))