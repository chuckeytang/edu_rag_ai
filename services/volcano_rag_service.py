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

logger = logging.getLogger(__name__)

class VolcanoEngineRagException(Exception):
    def __init__(self, code, request_id, message=None):
        self.code = code
        self.request_id = request_id
        self.message = f"{message}, code:{self.code}，request_id:{self.request_id}"
    def __str__(self):
        return self.message

class VolcanoEngineRagService(Service):
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
        
        # ⚠️ 移除有问题的同步 Ping 测试，以解决启动错误
        logger.info("Volcano Engine RAG service initialized. Ping test skipped.")

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
        }
        return api_info

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

    async def import_document_url(self, 
                                  url: str, 
                                  doc_name: str, 
                                  knowledge_base_id: str, 
                                  doc_id: str, # 新增 doc_id 参数
                                  doc_type: str, # 新增 doc_type 参数
                                  meta: Optional[List[Dict[str, Any]]] = None # 新增 meta 参数
                                  ) -> Dict[str, Any]:
        
        logger.info(f"Importing document '{doc_name}' from URL to knowledge base '{knowledge_base_id}'.")
        
        # 参照火山引擎官方文档 doc/add 的请求参数来构建 payload
        payload = {
            "collection_name": knowledge_base_id, # 火山引擎的文档中 collection_name 对应知识库名称
            "resource_id": knowledge_base_id,
            "add_type": "url",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_type": doc_type,
            "url": url,
        }
        
        if meta:
            payload["meta"] = meta
            
        try:
            # 调用 _async_make_request 并确保 API 名称正确
            return await self._async_make_request("ImportDocumentUrl", {}, payload)
        except Exception as e:
            logger.error(f"Error during API call with payload: {payload}", exc_info=True)
            raise e

    async def delete_document(self, knowledge_base_id: str, doc_id: str) -> Dict[str, Any]:
        logger.info(f"Deleting document '{doc_id}' from knowledge base '{knowledge_base_id}'.")
        payload = {
            "resource_id": knowledge_base_id,
            "doc_id": doc_id,
        }
        return await self._async_make_request("DeleteDocument", {}, payload)

    async def retrieve_documents(self,
                                 query_text: str,
                                 knowledge_base_id: str,
                                 limit: int = 10,
                                 rerank_switch: bool = True,
                                 filters: Optional[Dict[str, Any]] = None,
                                 dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving for query '{query_text[:50]}...' from knowledge base '{knowledge_base_id}'.")
        
        payload = {
            "resource_id": knowledge_base_id,
            "query": query_text,
            "limit": limit,
            "dense_weight": dense_weight,
            "post_processing": {
                "rerank_switch": rerank_switch
            }
        }
        
        if filters:
            payload["filters"] = filters
        
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