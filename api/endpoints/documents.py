from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from services.document_service import document_service
from models.schemas import Document, DocumentMetadata, DebugRequest, DocumentChunkResponse
from services.document_oss_service import document_oss_service
import os
import logging

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
    summary="[OSS上传] 按OSS Key获取文档的完整元数据和内容片段"
)
async def get_oss_document_metadata(
    file_key: str = Query(..., description="目标文档的 OSS file_key")
):
    """
    根据 OSS file_key 获取文档每一页（chunk）的详细信息，
    其中包含完整的、文件级别的 RAGMetadata。
    """
    try:
        # 1. 根据 key 查找永久存储路径 (逻辑不变)
        permanent_path = document_oss_service.processed_oss_keys.get(file_key)
        if not permanent_path or not os.path.exists(permanent_path):
            raise HTTPException(status_code=404, detail=f"File for key '{file_key}' not found on disk.")

        # 2. 加载和过滤文档 (逻辑不变)
        docs = document_oss_service._load_documents_with_metadata(file_name=permanent_path)
        filtered_docs = document_oss_service._filter_documents(docs)

        if not filtered_docs:
            return []

        # --- 核心修改 2: 构造新的、结构更丰富的响应体 ---
        response_chunks = []
        for doc in filtered_docs:
            # Pydantic 可以直接从字典创建模型实例
            # doc.metadata 是一个包含了所有文件级和页面级元数据的字典
            file_meta_obj = RAGMetadata(**doc.metadata)
            
            chunk_response = DocumentChunkResponse(
                page_label=doc.metadata.get("page_label", "N/A"),
                text_snippet=doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                file_metadata=file_meta_obj
            )
            response_chunks.append(chunk_response)
        
        return response_chunks

    except Exception as e:
        logger.error(f"Failed to get metadata for OSS key '{file_key}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug-question")
async def debug_question_search(request: DebugRequest):
    return document_service.debug_question_search(
        request.filename,
        request.question
    )