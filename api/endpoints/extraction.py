# endpoints/extraction.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from models.schemas import ExtractionRequest
from services.ai_extraction_service import ai_extraction_service, DocumentMetadata, Flashcard # 导入新服务和模型

router = APIRouter()

# --- 元数据提取 API ---
@router.post("/extract_metadata", response_model=DocumentMetadata, summary="从文档或文本中提取元数据")
async def extract_metadata_endpoint(request: ExtractionRequest):
    """
    接收一个 OSS 文件键或直接的文本内容，并使用 AI 提取文档的元数据。
    """
    try:
        metadata = await ai_extraction_service.extract_document_metadata(
            file_key=request.file_key, 
            text_content=request.content,
            is_public=request.is_public
        )
        return metadata
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"元数据提取失败: {e}")

# --- 知识点（记忆卡）提炼 API ---
@router.post("/extract_flashcards", response_model=List[Flashcard], summary="从文档或文本中提炼知识点（记忆卡）")
async def extract_flashcards_endpoint(request: ExtractionRequest):
    """
    接收一个 OSS 文件键或直接的文本内容，并使用 AI 提炼知识点，形成记忆卡。
    """
    try:
        flashcards = await ai_extraction_service.extract_knowledge_flashcards(
            file_key=request.file_key, 
            text_content=request.content,
            is_public=request.is_public
        )
        return flashcards
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记忆卡提炼失败: {e}")