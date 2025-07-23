from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_query_service
from services.query_service import QueryService
from models.schemas import ChatQueryRequest, QueryResponse, QueryResponseNode, StreamChunk, StreamingResponseWrapper
from fastapi.responses import StreamingResponse
import logging

logger = logging.getLogger(__name__)
    
router = APIRouter()
@router.post("/rag-query-with-context")
async def rag_query_with_context_api(request: ChatQueryRequest,
                                     query_service: QueryService = Depends(get_query_service)):
    """
    统一的流式 RAG 查询接口，支持：
    - 聊天历史上下文的语义检索
    - 指定文件 (material_id) 范围内的 RAG
    - 额外的元数据过滤 (filters)
    """
    try:
        # 直接返回 query_service.rag_query_with_context(request) 产生的异步生成器
        # FastAPI 的 StreamingResponse 会自动接管这个生成器，并将其作为 SSE 流发送
        return StreamingResponse(
            query_service.rag_query_with_context(request),
            media_type="text/event-stream"
        )

    except ValueError as ve:
        logger.error(f"Validation error in /rag-query-with-context: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unhandled error in /rag-query-with-context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during RAG processing.")
