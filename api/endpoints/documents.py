from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from api.dependencies import get_query_service
from services.document_service import document_service
from models.schemas import Document, DocumentMetadata, DebugRequest, DocumentChunkResponse, RAGMetadata
from services.document_oss_service import document_oss_service
import os
import logging

from services.query_service import QueryService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
router = APIRouter()

from fastapi import Query

@router.get("/documents", response_model=List[dict])
async def list_documents(keyword: str = Query(None, description="关键词过滤文件名（忽略大小写）")):
    try:
        # 加载 hash -> path 映射
        hashes = document_service.file_hashes  # dict[str, str]

        # 构造过滤后列表
        results = []
        for file_hash, file_path in hashes.items():
            file_name = os.path.basename(file_path)
            if keyword is None or keyword.lower() in file_name.lower():
                results.append({
                    "file_hash": file_hash,
                    "file_name": file_name
                })

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/doc-metadata", response_model=List[Document])
async def get_document_metadata(
    file_hash: str = Query(..., description="目标文档的 hash 值")
):
    try:
        file_path = document_service.file_hashes.get(file_hash)
        if not file_path:
            raise HTTPException(status_code=404, detail="File hash not found")

        docs = document_service.load_documents([file_path])
        filtered_docs, _ = document_service.filter_documents(docs)

        return [
            Document(
                text=doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                metadata=DocumentMetadata(
                    file_name=doc.metadata.get("file_name", "unknown"),
                    page_label=doc.metadata.get("page_label", "unknown")
                )
            )
            for doc in filtered_docs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@router.get("/documents-oss", response_model=List[dict], summary="[OSS上传] 列出通过OSS Key索引的文件")
async def list_oss_documents(keyword: str = Query(None, description="关键词过滤文件名（忽略大小写）")):
    """
    列出所有通过 OSS 流程处理过的文件。
    数据来源于 document_oss_service 的持久化缓存。
    """
    try:
        # 从 document_oss_service 加载 file_key -> filename 映射
        processed_map = document_oss_service.processed_files

        results = []
        for file_key, file_name in processed_map.items():
            if keyword is None or keyword.lower() in file_name.lower():
                results.append({
                    "file_key": file_key,
                    "file_name": file_name
                })
        return results
    except Exception as e:
        logger.error(f"Failed to list OSS documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/doc-metadata-oss",
    response_model=List[DocumentChunkResponse], 
    summary="[ChromaDB查询] 按OSS Key获取文档的元数据和内容片段"
)
async def get_oss_document_metadata(
    file_key: str = Query(..., description="目标文档的 OSS file_key"),
    query_service: QueryService = Depends(get_query_service)
):
    """
    根据 OSS file_key 从 ChromaDB 中直接获取已索引的文档的
    每一页（chunk）的详细信息。
    """
    try:
        # 1. 定义过滤器，使用我们已存入的 file_key 字段
        filters = {"file_key": file_key}
        collection_name = "public_collection" # 或根据业务逻辑决定

        # 2. 调用 QueryService 的新方法来获取所有匹配的节点
        retrieved_nodes = query_service.get_nodes_by_metadata_filter(
            collection_name=collection_name,
            filters=filters
        )

        if not retrieved_nodes:
            raise HTTPException(status_code=404, detail=f"No processed data found in DB for file key '{file_key}'.")

        # 3. 将节点数据转换为 API 响应模型
        response_chunks = []
        for node in retrieved_nodes:
            # Pydantic 可以直接从字典创建模型实例
            metadata_from_db = node.extra_info.copy()
            # 2. 转换 accessible_to 字段（如果存在且为字符串）
            if 'accessible_to' in metadata_from_db and isinstance(metadata_from_db['accessible_to'], str):
                raw_str = metadata_from_db['accessible_to'].strip(',')
                # 如果 strip 后为空字符串（对应原始的空列表），则返回空列表，否则切割
                metadata_from_db['accessible_to'] = raw_str.split(',') if raw_str else []

            # 3. 转换 level_list 字段（如果存在且为字符串）
            if 'level_list' in metadata_from_db and isinstance(metadata_from_db['level_list'], str):
                raw_str = metadata_from_db['level_list'].strip(',')
                metadata_from_db['level_list'] = raw_str.split(',') if raw_str else []

            # node.metadata 是一个包含了所有文件级和页面级元数据的字典
            file_meta_obj = RAGMetadata(**metadata_from_db)
            
            chunk_response = DocumentChunkResponse(
                page_label=node.extra_info.get("page_label", "N/A"),
                text_snippet=node.text[:500] + "..." if len(node.text) > 500 else node.text,
                metadata=file_meta_obj
            )
            response_chunks.append(chunk_response)
        
        return response_chunks

    except Exception as e:
        logger.error(f"Error retrieving document metadata for key '{file_key}': {e}", exc_info=True)
        # 抛出一个通用的服务器错误，而不是误导性的404
        raise HTTPException(status_code=500, detail="An internal error occurred while retrieving document metadata.")