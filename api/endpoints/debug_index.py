import logging
import json
import time
import asyncio
from typing import Any, Dict, List, Optional
import numpy as np
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

# --- 引入项目服务和配置 ---
from services.query_service import QueryService
from services.volcano_rag_service import VolcanoEngineRagService 
from services.indexer_service import IndexerService
from api.dependencies import get_query_service, get_indexer_service
from models.schemas import ChatQueryRequest, QueryRequest, DeleteByMetadataRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# --- 辅助函数：获取 Volcano RAG Service 实例 ---
# 假设 VolcanoEngineRagService 已经注入到 QueryService 或 IndexerService 中
def get_volcano_rag_service(query_service: QueryService = Depends(get_query_service)) -> VolcanoEngineRagService:
    # 假设 VolcanoEngineRagService 实例被存储在 QueryService 中
    return query_service.volcano_rag_service 

# --- Schema 定义 ---

# 定义一个用于返回调试信息的Schema
class DebugQueryResponse(BaseModel):
    query_text: str
    final_llm_prompt: str
    original_retrieved_nodes: List[Dict[str, Any]]
    final_retrieved_nodes: List[Dict[str, Any]]
    final_response_content: str
    citations: List[Dict[str, Any]]
    
    
@router.get("/list-chunks-volcano", summary="[DEBUG-VOLCANO] 按 Doc ID 查看知识库切片")
async def list_chunks_volcano(
    knowledge_base_id: str = Query(..., description="要查询的知识库唯一 ID (resource_id)"),
    doc_id: Optional[str] = Query(
        None, description="按文档 ID (doc_id) 精确过滤。例如: material-import_xxx_pdf"
    ),
    limit: int = Query(50, ge=1, le=100, description="最多返回的切片数量（火山 API 最大为 100）"),
    offset: int = Query(0, ge=0, description="分页起始位置"),
    volcano_rag_service: VolcanoEngineRagService = Depends(get_volcano_rag_service)
) -> List[Dict[str, Any]]:
    """
    通过调用 /api/knowledge/point/list 接口，获取知识库中切片的完整内容和元数据，用于验证索引和权限字段。
    """
    logger.info(f"Received request to list chunks for KB '{knowledge_base_id}' with doc_id: '{doc_id}', limit: {limit}, offset: {offset}")

    doc_ids_list = [doc_id] if doc_id else None
    
    try:
        # 调用新的切片列表方法
        chunks_with_meta = await volcano_rag_service.list_knowledge_points(
            knowledge_base_id=knowledge_base_id,
            doc_ids=doc_ids_list,
            limit=limit,
            offset=offset
        )

        # 格式化结果以返回给前端
        results = []
        for chunk in chunks_with_meta:
            results.append({
                # 这里的 score 不可用，因为这不是检索 API
                "doc_id": chunk.get('doc_id'),
                "point_id": chunk.get('point_id'),
                "doc_name": chunk.get('doc_name'),
                "source": chunk.get('source'),
                "text_snippet": chunk.get('content', '')[:500] + "...",
                "metadata": chunk.get('metadata', {}) 
            })

        logger.info(f"Successfully retrieved {len(results)} knowledge points.")
        return results

    except Exception as e:
        logger.error(f"Failed to list knowledge points from Volcano KB: {e}", exc_info=True)
        # 统一异常处理
        if hasattr(e, 'code') and hasattr(e, 'message'):
            raise HTTPException(status_code=500, detail=f"Volcano API Error (Code {e.code}): {e.message}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
@router.post("/retrieve-with-filters-volcano", summary="[DEBUG-VOLCANO] 测试带过滤的节点召回")
async def debug_retrieve_with_filters_volcano(
        request: QueryRequest,
        volcano_rag_service: VolcanoEngineRagService = Depends(get_volcano_rag_service),
        query_service: QueryService = Depends(get_query_service)
) -> List[Dict[str, Any]]:
    """
    只执行火山引擎的检索步骤，并返回召回的文档块列表及其分数。
    - **filters**: 传递 Volcano API 格式的元数据过滤（op, field, conds）。
    - **target_file_ids**: 传递火山 Doc ID 字符串列表进行精确过滤。
    """
    try:
        # 1. 初始化过滤器
        volcano_filters = request.filters.copy() if request.filters else {}
        filter_expressions = []

        # 2. 处理 target_file_ids (DocID 过滤)
        if request.target_file_ids:
            doc_id_filter = {
                "op": "must",
                "field": "doc_id",
                "conds": request.target_file_ids # 假设 Java 端已转换为 Volcano Doc ID 格式
            }
            filter_expressions.append(doc_id_filter)

        # 3. 合并其他元数据过滤器 (假设 request.filters 已是 Volcano 格式)
        if volcano_filters:
            filter_expressions.append(volcano_filters)
            
        # 4. 组合所有过滤器（使用 AND 逻辑）
        final_doc_filter = None
        if len(filter_expressions) == 1:
            final_doc_filter = filter_expressions[0]
        elif len(filter_expressions) > 1:
            final_doc_filter = {"op": "and", "conds": filter_expressions}

        logger.info(f"Final Volcano Doc Filter: {json.dumps(final_doc_filter)}")
        
        final_top_k = request.similarity_top_k if request.similarity_top_k is not None else query_service.rag_config.retrieval_top_k

        # 5. 调用 Volcano Engine 检索 API (rerank_switch=True 确保返回 rerank score)
        retrieved_chunks = await volcano_rag_service.retrieve_documents(
            query_text=request.question,
            knowledge_base_id=request.knowledge_base_id,
            limit=final_top_k,
            rerank_switch=True,
            filters=final_doc_filter
        )

        # 6. 格式化结果
        results = []
        for chunk in retrieved_chunks:
            # Volcano API 返回的已经是字典格式，可以直接使用
            results.append({
                "score": chunk.get('rerank_score', chunk.get('score', 0.0)),
                "raw_score": chunk.get('score', 0.0),
                "doc_id": chunk.get('docId'),
                "chunk_id": chunk.get('chunkId'),
                "source": chunk.get('source'),
                "text_snippet": chunk.get('content', '')[:300] + "...",
                "metadata": chunk.get('user_data', {}) 
            })

        return results

    except Exception as e:
        logger.error(f"Debug Volcano retrieval failed: {e}", exc_info=True)
        if hasattr(e, 'code') and hasattr(e, 'message'):
            raise HTTPException(status_code=500, detail=f"Volcano API Error (Code {e.code}): {e.message}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/debug-rag-flow-volcano", summary="[DEBUG-VOLCANO] 执行并展示完整的RAG流程")
async def debug_rag_flow_volcano(
        request: ChatQueryRequest,
        query_service: QueryService = Depends(get_query_service),
) -> DebugQueryResponse:
    """
    执行基于火山知识库的完整 RAG 流程，并返回每个关键步骤的详细结果。
    通过消费 QueryService.rag_query_with_context 的 SSE 流来提取信息。
    """
    logger.info(f"Starting debug RAG flow for KB '{request.knowledge_base_id}' with query: '{request.question}'")

    response_stream = query_service.rag_query_with_context(request, request.rag_config)
    
    full_response_content = ""
    final_metadata = {}
    
    # 模拟流式消费，收集最终的回复内容和 metadata
    try:
        async for chunk_bytes in response_stream:
            chunk_line = chunk_bytes.decode('utf-8').strip()
            if not chunk_line.startswith("data:"):
                continue

            chunk_json = chunk_line[5:].strip()
            chunk = json.loads(chunk_json)

            if chunk.get("is_last"):
                final_metadata = chunk.get("metadata", {})
            else:
                full_response_content += chunk.get("content", "")

    except Exception as e:
        logger.error(f"Error during streaming RAG query: {e}", exc_info=True)
        if hasattr(e, 'code'):
            raise HTTPException(status_code=500, detail=f"Volcano Retrieval Error (Code {e.code}): {e.message}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    # --- 提取调试信息 ---
    
    # 1. 最终回复内容 (LLM output)
    final_response_content = full_response_content

    # 2. 引用和召回节点信息 (从 final_metadata 中提取)
    citations = final_metadata.get("sentence_citations", [])
    
    # 无法直接获取 Original/Final 召回节点分数和内容，只能通过引用信息反推
    # 我们将 Final Retrieved Nodes 定义为所有被引用的文档。
    final_retrieved_nodes_map = {}
    
    for citation in citations:
        chunk_id = citation['referenced_chunk_id']
        if chunk_id not in final_retrieved_nodes_map:
            final_retrieved_nodes_map[chunk_id] = {
                "score": "N/A (Citations)",
                "node_id": chunk_id,
                "text_snippet": citation['referenced_chunk_text'][:300] + "...",
                "metadata": {
                    "document_id": citation.get('document_id'),
                    "material_id": citation.get('material_id'),
                    "file_name": citation.get('file_name'),
                    "source_type": citation.get('source_type'),
                }
            }

    final_retrieved_nodes = list(final_retrieved_nodes_map.values())
    original_retrieved_nodes = [] # 无法从最终 metadata 中反推，留空

    # 3. Final LLM Prompt (需要从服务日志中查找)
    final_llm_prompt_placeholder = "The full prompt is constructed internally by QueryService. Please check service logs for 'final_prompt_for_llm'."
    
    # 4. 封装并返回所有调试信息
    return DebugQueryResponse(
        query_text=request.question,
        final_llm_prompt=final_llm_prompt_placeholder,
        original_retrieved_nodes=original_retrieved_nodes,
        final_retrieved_nodes=final_retrieved_nodes,
        final_response_content=final_response_content,
        citations=citations
    )

@router.delete("/delete-by-metadata-volcano", summary="[DEBUG-VOLCANO] 根据元数据删除文档")
def delete_by_metadata_volcano(request: DeleteByMetadataRequest,
                               indexer_service: IndexerService = Depends(get_indexer_service)):
    """
    接收来自后端的指令，根据元数据过滤器删除火山知识库中的文档。
    该接口直接调用 IndexerService。
    """
    logger.info(f"Received request to delete documents from Volcano KB with filters: {request.filters}")
    try:
        # 假设 IndexerService.delete_nodes_by_metadata 已经适配了 Volcano API
        # 注意: 火山 API 通常只支持按 doc_id 删除
        result = indexer_service.delete_nodes_by_metadata(
            knowledge_base_id=request.knowledge_base_id,
            filters=request.filters
        )
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    except Exception as e:
        logger.error(f"Volcano Deletion request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during deletion: {str(e)}")