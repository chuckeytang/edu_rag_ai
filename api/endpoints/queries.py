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
    """
    处理不带元数据过滤的查询请求。
    """
    try:
        response = query_service.query(
            question=request.question,
            collection_name=request.collection_name, 
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

@router.post("/query-with-filters", response_model=QueryResponse)
async def query_with_filters(request: QueryRequest):
    """
    处理带元数据过滤的查询请求。
    使用与 /query 相同的请求模型，但会利用其中的 filters 字段。
    """
    try:
        # 调用 query_service.query_with_filters 方法
        response = query_service.query_with_filters(
            question=request.question,
            collection_name=request.collection_name,
            filters=request.filters, # 传递 filters
            similarity_top_k=request.similarity_top_k,
            prompt=request.prompt
        )
        
        # 响应处理逻辑与上面完全相同
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
        return QueryResponse(response=response.response, nodes=nodes)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@router.post("/stream-query", response_class=StreamingResponse)
async def stream_query(request: QueryRequest):
    async def generate():
        try:
            async for chunk in query_service.stream_query(
                    question=request.question,
                    collection_name=request.collection_name,
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

@router.post("/stream-query-with-filters", response_class=StreamingResponse)
async def stream_query_with_filters(request: QueryRequest):
    async def generate():
        try:
            async for chunk in query_service.stream_query_with_filters(
                    question=request.question,
                    collection_name=request.collection_name,
                    filters=request.filters, # 传递 filters
                    similarity_top_k=request.similarity_top_k,
                    prompt=request.prompt
            ):
                yield chunk
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_last=True
            ).json() + "\n"
    return StreamingResponse(StreamingResponseWrapper(generate()), media_type="text/event-stream")

@router.post("/stream-query-with-files")
async def query_with_files(
        request: QueryRequest,
        response_class=StreamingResponse
):
    async def generate():
        try:
            if not request.target_file_ids:
                logger.info(f"No target_file_ids provided. Performing a regular stream query on collection '{request.collection_name}'.")
                if not request.collection_name:
                    raise ValueError("collection_name is required for a general query.")
                async for chunk in query_service.stream_query(
                        question=request.question,
                        collection_name=request.collection_name,
                        similarity_top_k=request.similarity_top_k,
                        prompt=request.prompt
                ):
                    yield chunk
            else:
                logger.info(f"target_file_ids provided. Performing a query within specific files.")
                async for chunk in query_service.query_with_files(
                        question=request.question,
                        file_identifiers=request.target_file_ids,
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
