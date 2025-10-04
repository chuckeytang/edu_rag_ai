from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_query_service
from core.rag_config import RagConfig
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
            query_service.rag_query_with_context(request, request.rag_config),
            media_type="text/event-stream"
        )

    except ValueError as ve:
        logger.error(f"Validation error in /rag-query-with-context: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unhandled error in /rag-query-with-context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during RAG processing.")
    
@router.get("/doc-id/bailian/{knowledge_base_id}/{doc_id}", 
            summary="根据业务ID查找百炼KB的文档ID")
async def get_bailian_doc_id(
    knowledge_base_id: str,
    doc_id: str, # 业务侧的filekey
    rag_service: QueryService = Depends(get_query_service) 
):
    """
    通过业务 Doc ID (Material ID) 检索阿里百炼的 File ID (kb_doc_id)。
    """
    try:
        bailian_doc_id = await rag_service.get_bailian_doc_id_by_doc_id(
            knowledge_base_id=knowledge_base_id, 
            doc_id=doc_id
        )
        
        if bailian_doc_id:
            return {"kb_doc_id": bailian_doc_id}
        else:
            # 如果找不到，返回 404
            raise HTTPException(status_code=404, detail=f"Bailian KB ID not found for Doc ID: {doc_id}")
            
    except Exception as e:
        logger.error(f"Failed to retrieve Bailian Doc ID for {doc_id}: {e}", exc_info=True)
        # 向上抛出更具体的错误
        raise HTTPException(status_code=500, detail=f"Internal RAG Service Error: {str(e)}")
