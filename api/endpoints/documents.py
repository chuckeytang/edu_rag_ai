from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from services.document_service import document_service
from models.schemas import Document, DocumentMetadata, DebugRequest
import os
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
        
@router.post("/debug-question")
async def debug_question_search(request: DebugRequest):
    return document_service.debug_question_search(
        request.filename,
        request.question
    )