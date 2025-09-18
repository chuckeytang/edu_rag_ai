import logging
import json
import time
from typing import List, Dict, Any, Optional

import requests
from requests.exceptions import RequestException

# 导入 volcengine 的签名 SDK
# 你可能需要先安装：pip install volcengine-python-sdk
try:
    from volcengine.auth.SignerV4 import Signer as SignerV4
except ImportError:
    logging.error("Volcengine Python SDK not found. Please install with 'pip install volcengine-python-sdk'.")
    SignerV4 = None

from core.config import settings

logger = logging.getLogger(__name__)

class VolcanoEngineRagService:
    """
    封装了与火山引擎知识库服务 API 的交互。
    负责文档的上传、更新、删除和检索。
    """
    def __init__(self):
        self.access_key_id = settings.VOLCANO_ENGINE_AK_ID
        self.secret_key = settings.VOLCANO_ENGINE_SK_SECRET
        self.api_url = settings.VOLCANO_ENGINE_RAG_API_URL
        self.region = settings.VOLCANO_ENGINE_REGION
        self.service = "air" # 知识库服务的 service name

        if not all([self.access_key_id, self.secret_key, self.api_url, self.region]):
            logger.warning("Volcano Engine RAG configuration is missing. This service will be non-functional.")

    def _create_signed_headers(self, method: str, pathname: str, body: str) -> Dict[str, str]:
        """
        使用 volcengine SDK 为请求生成带有鉴权的头部。
        """
        if not SignerV4:
            raise ImportError("Volcengine Python SDK is not installed.")

        # 构建请求对象
        req = {
            'method': method,
            'scheme': 'https',
            'host': self.api_url.replace("https://", ""),
            'pathname': pathname,
            'query': {},
            'header': {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            'body': body
        }

        # 使用 SignerV4 进行签名
        signer = SignerV4(req, self.service, self.region)
        signer.sign(self.access_key_id, self.secret_key)

        return req['header']

    def _make_request(self, method: str, pathname: str, payload: dict) -> Dict[str, Any]:
        """
        通用的请求执行器，处理签名和发送请求。
        """
        if not self.access_key_id:
            logger.error("Volcano Engine RAG service is not configured.")
            raise RuntimeError("Volcano Engine RAG service is not configured.")

        url = f"{self.api_url}{pathname}"
        body = json.dumps(payload, ensure_ascii=False)
        
        try:
            headers = self._create_signed_headers(method, pathname, body)
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=body,
                timeout=60
            )
            response.raise_for_status()
            
            response_data = response.json()
            if response_data.get("code") != 0:
                message = response_data.get("message", "Unknown error from Volcano Engine API")
                raise RequestException(f"Volcano Engine API Error: {message}")
            
            return response_data
            
        except RequestException as e:
            logger.error(f"Request to Volcano Engine API failed: {e}")
            raise

    # --- 文档管理 API ---

    def upload_document(self, file_content: bytes, filename: str, knowledge_base_id: str,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        上传单个文件到知识库。
        
        Args:
            file_content: 文件内容的字节流。
            filename: 文件名。
            knowledge_base_id: 知识库 ID。
            metadata: 额外元数据。
            
        Returns:
            API 响应数据。
        """
        logger.info(f"Uploading document '{filename}' to knowledge base '{knowledge_base_id}'.")
        
        # 假设上传 API 是通过 form-data 格式
        # 但 volcengine 知识库的API文档显示是先上传文件到对象存储，再通过 URL 导入。
        # 这里为了简化，我们假设有一个直接的上传 API，但实际生产环境应遵循文档。
        # 实际操作可能需要调用火山引擎 OSS SDK 先上传文件，再将 OSS URL 传给 RAG 服务的 `import_document` API。
        
        # 为了演示，我们使用一个假设的 `import_document_url` API
        # 实际的 API 路径和参数可能不同，请参照最新的火山引擎文档进行调整
        pathname = "/api/knowledge/collection/import_document_url"
        
        # 这里需要将文件上传到火山引擎的对象存储（TOS），然后将 TOS URL 传递给这个 API
        # 伪代码：
        # tos_url = tos_client.upload(file_content, filename)
        # payload = {
        #     "resource_id": knowledge_base_id,
        #     "url": tos_url,
        #     "doc_name": filename,
        #     "callback_url": "your_callback_url", # 可选
        #     "user_data": metadata # 可选
        # }
        
        # 这是一个简化的示例 payload
        payload = {
            "resource_id": knowledge_base_id,
            "urls": [
                # 这里应该填写 TOS 上的文件 URL
            ],
            "doc_name": filename,
            "user_data": json.dumps(metadata) if metadata else ""
        }
        
        # 实际上，这里需要调用一个批量上传 API，或者先将文件上传到 TOS。
        # 这个方法需要你根据实际 API 调整
        raise NotImplementedError("Direct file upload is not the standard way. Please implement logic to upload file to VolcanoEngine TOS first, then import the TOS URL.")


    def delete_document(self, knowledge_base_id: str, doc_id: str) -> Dict[str, Any]:
        """
        从知识库中删除一个文档。
        """
        logger.info(f"Deleting document '{doc_id}' from knowledge base '{knowledge_base_id}'.")
        pathname = "/api/knowledge/collection/delete_document"
        payload = {
            "resource_id": knowledge_base_id,
            "doc_id": doc_id,
        }
        return self._make_request("POST", pathname, payload)

    # --- 文档检索 API ---

    async def retrieve_documents(self,
                                 query_text: str,
                                 knowledge_base_id: str,
                                 limit: int = 10,
                                 rerank_switch: bool = True,
                                 dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        从知识库中检索知识切片。
        
        Args:
            query_text: 用户查询文本。
            knowledge_base_id: 知识库 ID。
            limit: 返回的知识切片数量。
            rerank_switch: 是否启用重排。
            dense_weight: 向量召回权重。
            
        Returns:
            包含 content, score, doc_info 等信息的知识切片列表。
        """
        logger.info(f"Retrieving for query '{query_text[:50]}...' from knowledge base '{knowledge_base_id}'.")
        
        pathname = "/api/knowledge/collection/search_knowledge"
        
        payload = {
            "resource_id": knowledge_base_id,
            "query": query_text,
            "limit": limit,
            "dense_weight": dense_weight,
            "post_processing": {
                "rerank_switch": rerank_switch
            }
        }
        
        try:
            # Volcano Engine API requests are synchronous, so we'll treat them as such.
            # You can wrap this in an executor for true async behavior if needed,
            # but for API calls, a direct call is often sufficient and cleaner.
            response_data = self._make_request("POST", pathname, payload)
            
            result_list = response_data.get("data", {}).get("result_list", [])
            logger.info(f"Successfully retrieved {len(result_list)} chunks.")

            # 映射结果格式，与你提供的 TypeScript 代码保持一致
            mapped_results = []
            for item in result_list:
                doc_info = item.get('doc_info', {})
                mapped_results.append({
                    "content": item.get('content'),
                    "score": item.get('score'),
                    "rerank_score": item.get('rerank_score'),
                    "source": doc_info.get('doc_name'),
                    "docId": doc_info.get('doc_id'),
                    "chunkId": item.get('id'),
                    "url": doc_info.get('url'),
                    # 你可能还需要添加 page_label 和 material_id 等自定义元数据
                    "material_id": doc_info.get('user_data', {}).get('material_id')
                })

            return mapped_results

        except Exception as e:
            logger.error(f"Failed to retrieve from Volcano Engine RAG: {e}")
            return []