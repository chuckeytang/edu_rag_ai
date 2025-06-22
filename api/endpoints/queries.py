from fastapi import APIRouter, HTTPException
from services.query_service import query_service
from models.schemas import QueryRequest, QueryResponse, QueryResponseNode, StreamChunk, StreamingResponseWrapper
from fastapi.responses import StreamingResponse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
router = APIRouter()
@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        response = query_service.query(
            question=request.question,
            similarity_top_k=request.similarity_top_k,
            prompt=request.prompt
        )
        nodes = []
        for node in response.source_nodes:
            try:
                file_name = node.metadata.get("file_name", "unknown")
                page_label = node.metadata.get("page_label", "unknown")
                if hasattr(node, 'text'):
                    node_text = node.text
                elif hasattr(node, 'node') and hasattr(node.node, 'text'):
                    node_text = node.node.text
                elif hasattr(node, 'get_content'):
                    node_text = node.get_content()
                else:
                    node_text = "[Node text not accessible]"
                nodes.append(QueryResponseNode(
                    file_name=file_name,
                    page_label=page_label,
                    content=node_text,
                    score=node.score
                ))
            except Exception as e:
                continue
        return QueryResponse(
            response=response.response,
            nodes=nodes
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing error: {str(e)}"
        )
@router.post("/stream-query", response_class=StreamingResponse)
async def stream_query(request: QueryRequest):
    async def generate():
        try:
            async for chunk in query_service.stream_query(
                    question=request.question,
                    similarity_top_k=request.similarity_top_k,
                    prompt=request.prompt
            ):
                yield chunk
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_last=True
            ).json() + "\n"
    return StreamingResponse(
        StreamingResponseWrapper(generate()),
        media_type="text/event-stream"
    )
@router.post("/query-with-files")
async def query_with_files(
        request: QueryRequest,
        response_class=StreamingResponse
):
    async def generate():
        try:
            if not request.target_file_ids:
                async for chunk in query_service.stream_query(
                        question=request.question,
                        similarity_top_k=request.similarity_top_k,
                        prompt=request.prompt
                ):
                    yield chunk
            else:
                async for chunk in query_service.query_with_files(
                        question=request.question,
                        file_hashes=request.target_file_ids,
                        similarity_top_k=request.similarity_top_k,
                        prompt=request.prompt
                ):
                    yield StreamChunk(content=chunk).json() + "\n"
                yield StreamChunk(content="", is_last=True).json() + "\n"
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_last=True
            ).json() + "\n"
    return StreamingResponse(
        StreamingResponseWrapper(generate()),
        media_type="text/event-stream"
    )

@router.post("/query-with-filters")
def perform_query(request: QueryRequest):
    """
    Performs a query against the RAG system, with optional metadata filters.
    """
    try:
        response = query_service.query_with_filters(
            question=request.question,
            filters=request.filters
        )
        return {
            "answer": response.response,
            "source_nodes": [
                {
                    "text": node.get_text()[:300] + "...",
                    "score": node.get_score(),
                    "metadata": node.metadata
                } for node in response.source_nodes
            ]
        }
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))