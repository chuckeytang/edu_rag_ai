# services/abstract_kb_service.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Awaitable

class AbstractKnowledgeBaseService(ABC):
    """
    知识库服务抽象接口：定义了 RAG 后端所需的核心知识库操作。
    """

    @abstractmethod
    async def import_document_url(self, 
                                  url: str, 
                                  doc_name: str, 
                                  knowledge_base_id: str, 
                                  doc_id: str, 
                                  doc_type: str, 
                                  meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """导入外部 URL 文档到知识库。"""
        pass
    
    @abstractmethod
    async def delete_document(self, knowledge_base_id: str, doc_id: str) -> Dict[str, Any]:
        """根据文档 ID (File ID) 从知识库中删除文档。"""
        pass

    @abstractmethod
    async def retrieve_documents(self, 
                                 query_text: str, 
                                 knowledge_base_id: str, 
                                 limit: int = 10,
                                 rerank_switch: bool = True,
                                 filters: Optional[Dict[str, Any]] = None, 
                                 dense_weight: float = 0.5) -> List[Dict[str, Any]]:
        """根据查询文本检索相关的文档切片。"""
        pass
        
    @abstractmethod
    async def update_document_meta(self, doc_id: str, knowledge_base_id: str, meta_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """更新现有文档的元数据。"""
        pass

    @abstractmethod
    async def list_knowledge_points(self, 
                                    knowledge_base_id: str,
                                    doc_ids: Optional[List[str]] = None, 
                                    limit: int = 100,
                                    offset: int = 0) -> List[Dict[str, Any]]:
        """
        列出指定知识库/文档下的所有知识点（切片/Chunk）。
        """
        pass