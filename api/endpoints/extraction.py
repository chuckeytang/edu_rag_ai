# endpoints/extraction.py
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from api.dependencies import get_ai_extraction_service
from models.schemas import ExtractedDocumentMetadata, ExtractionRequest, Flashcard
from services.ai_extraction_service import AIExtractionService

router = APIRouter()
logger = logging.getLogger(__name__)

# --- 元数据提取 API ---
@router.post("/extract_metadata", response_model=ExtractedDocumentMetadata, summary="从文档或文本中提取元数据")
async def extract_metadata_endpoint(
    request: ExtractionRequest,
    ai_extraction_service: AIExtractionService = Depends(get_ai_extraction_service)
):
    """
    接收一个 OSS 文件键或直接的文本内容，并使用 AI 提取文档的元数据。
    """
    # 增加日志来检查接收到的请求
    logger.info(f"Received request for /extract_metadata: {request.model_dump_json(indent=2)}")
    
    # --- 在业务层进行数据校验和类型转换 ---
    # 检查是否提供了至少一个内容源
    if not request.file_key and not request.content:
        raise HTTPException(status_code=400, detail="必须提供 'file_key' 或 'content' 中的至少一个。")
    if request.file_key and request.content:
        raise HTTPException(status_code=400, detail="不能同时提供 'file_key' 和 'content'。")

    # 手动处理 user_provided_level 的类型转换
    processed_levels = []
    if isinstance(request.user_provided_level, str):
        processed_levels = [request.user_provided_level]
    elif isinstance(request.user_provided_level, list):
        processed_levels = request.user_provided_level
        
    # 手动处理 subscribed_subjects 的 None 值
    processed_subscribed_subjects = request.subscribed_subjects if request.subscribed_subjects else []

    try:
        metadata = await ai_extraction_service.extract_document_metadata(
            file_key=request.file_key, 
            text_content=request.content,
            is_public=request.is_public,
            user_provided_clazz=request.user_provided_clazz,       
            user_provided_subject=request.user_provided_subject,     
            user_provided_exam=request.user_provided_exam,        
            user_provided_level=processed_levels,  
            subscribed_subjects=processed_subscribed_subjects
        )
        return metadata
    except ValueError as e:
        # 捕捉业务校验错误，返回 400
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        # 捕捉从服务层抛出的 HTTPException，直接透传
        raise e
    except Exception as e:
        # 捕捉其他未知错误，返回 500
        raise HTTPException(status_code=500, detail=f"元数据提取失败: {e}")

# --- 知识点（记忆卡）提炼 API ---
@router.post("/extract_flashcards", response_model=List[Flashcard], summary="从文档或文本中提炼知识点（记忆卡）")
async def extract_flashcards_endpoint(request: ExtractionRequest,
                                      ai_extraction_service: AIExtractionService = Depends(get_ai_extraction_service)):
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