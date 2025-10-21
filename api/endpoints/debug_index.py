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
from services.abstract_kb_service import AbstractKnowledgeBaseService 
from services.indexer_service import IndexerService
from api.dependencies import get_query_service, get_indexer_service
from models.schemas import ChatQueryRequest, QueryRequest, DeleteByMetadataRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# --- 辅助函数：获取抽象 KB Service 实例 (通用化) ---
# 假设抽象 KB Service 实例被注入到 QueryService 或 IndexerService 中
# 注意：在您的 QueryService 构造函数中，抽象服务被命名为 self.kb_service
def get_kb_service(query_service: QueryService = Depends(get_query_service)) -> AbstractKnowledgeBaseService:
    """获取当前配置的抽象知识库服务实例。"""
    return query_service.kb_service 

# --- Schema 定义 ---

# 定义一个用于返回调试信息的Schema
class DebugQueryResponse(BaseModel):
    query_text: str
    final_llm_prompt: str
    original_retrieved_nodes: List[Dict[str, Any]]
    final_retrieved_nodes: List[Dict[str, Any]]
    final_response_content: str
    citations: List[Dict[str, Any]]
    
class CleanAllFilesRequest(BaseModel):
    knowledge_base_id: str = Query(..., description="要清理的知识库唯一 ID (Index ID/resource_id)")
    category_id: str = Query(..., description="知识库所属的类目 ID (CategoryId)，例如: default")
    max_results_per_page: int = Query(100, ge=1, le=200, description="分页查询时每页行数")
    force_index_delete: bool = Query(True, description="是否同时删除知识库索引中的文档")

@router.get("/list-chunks", summary="[DEBUG-KB] 按 Doc ID 查看知识库切片")
async def list_chunks_kb(
    knowledge_base_id: str = Query(..., description="要查询的知识库唯一 ID (Index ID/resource_id)"),
    doc_id: Optional[str] = Query(
        None, description="按文档 ID (doc_id) 精确过滤。"
    ),
    limit: int = Query(50, ge=1, le=100, description="最多返回的切片数量"),
    offset: int = Query(0, ge=0, description="分页起始位置"),
    kb_service: AbstractKnowledgeBaseService = Depends(get_kb_service) # 依赖注入抽象接口
) -> List[Dict[str, Any]]:
    """
    通过调用 list_knowledge_points 接口，获取知识库中切片的完整内容和元数据，用于验证索引和权限字段。
    """
    logger.info(f"Received request to list chunks for KB '{knowledge_base_id}' with doc_id: '{doc_id}', limit: {limit}, offset: {offset}")

    doc_ids_list = [doc_id] if doc_id else None
    
    try:
        # 调用抽象接口
        # 注意：阿里百炼的实现中，此方法返回的是文件列表，而非切片列表
        chunks_with_meta = await kb_service.list_knowledge_points(
            knowledge_base_id=knowledge_base_id,
            doc_ids=doc_ids_list,
            limit=limit,
            offset=offset
        )

        # 格式化结果以返回给前端
        results = []
        for chunk in chunks_with_meta:
            results.append({
                # 兼容 Volcano 的返回字段，但注意百炼可能返回 File ID/File Name
                "doc_id": chunk.get('doc_id', chunk.get('file_id', 'N/A')),
                "point_id": chunk.get('point_id', 'N/A'),
                "doc_name": chunk.get('doc_name', chunk.get('file_name', 'N/A')),
                "source": chunk.get('source', 'N/A'),
                "text_snippet": chunk.get('content', chunk.get('status', 'N/A'))[:500] + "...",
                "metadata": chunk.get('metadata', {}) 
            })

        logger.info(f"Successfully retrieved {len(results)} knowledge points.")
        return results

    except Exception as e:
        logger.error(f"Failed to list knowledge points from KB: {e}", exc_info=True)
        # 统一异常处理
        if hasattr(e, 'code') and hasattr(e, 'message'):
            raise HTTPException(status_code=500, detail=f"KB API Error (Code {e.code}): {e.message}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
@router.post("/retrieve-with-filters", summary="[DEBUG-KB] 测试带过滤的节点召回")
async def debug_retrieve_with_filters_kb(
        request: QueryRequest,
        kb_service: AbstractKnowledgeBaseService = Depends(get_kb_service), # 依赖注入抽象接口
        query_service: QueryService = Depends(get_query_service)
) -> List[Dict[str, Any]]:
    """
    只执行知识库的检索步骤，并返回召回的文档块列表及其分数。
    - **filters**: 传递简化元数据过滤（例如 {"accessible_to": ["80", "public"]}）。
    - **target_file_ids**: 传递文档 ID 字符串列表进行精确过滤。
    """
    try:
        # 1. 初始化过滤器：从请求中获取所有自定义过滤器
        # 确保 filters 字段存在且是可变字典
        combined_rag_filters = request.filters.copy() if request.filters else {}
        
        # 2. 处理 target_file_ids (DocID 过滤)
        if request.target_file_ids:
            logger.info(f"Applying doc_id filter: {request.target_file_ids}. Merging with existing filters.")
            # **核心修改：** 将 target_file_ids 直接合并为 'doc_id_list'
            combined_rag_filters["doc_id_list"] = request.target_file_ids
        
        # 3. 最终的过滤器就是这个扁平化字典
        final_doc_filter = combined_rag_filters 

        logger.info(f"Final KB Doc Filter (Simplified format): {json.dumps(final_doc_filter)}")
        
        # 4. 确定 top_k
        final_top_k = request.similarity_top_k if request.similarity_top_k is not None else query_service.rag_config.retrieval_top_k

        # 5. 调用抽象接口。底层 kb_service (如 BailianRagService) 会负责将简化格式
        #    (如 doc_id_list, accessible_to) 转换为其 API 所需的格式 (SearchFilters/Tags)。
        retrieved_chunks = await kb_service.retrieve_documents(
            query_text=request.question,
            knowledge_base_id=request.knowledge_base_id,
            limit=final_top_k,
            rerank_switch=True,
            filters=final_doc_filter # 传入整合后的过滤器
        )

        # 6. 格式化结果 (依赖底层服务返回的统一字典结构)
        results = []
        for chunk in retrieved_chunks:
            # chunk.get('user_data', {}) 包含了所有的元数据，包括 accessible_to 等
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
        logger.error(f"Debug KB retrieval failed: {e}", exc_info=True)
        if hasattr(e, 'code') and hasattr(e, 'message'):
            # 捕获 BailianRagException 或其他底层服务错误
            raise HTTPException(status_code=500, detail=f"KB API Error (Code {e.code}): {e.message}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/debug-rag-flow", summary="[DEBUG-KB] 执行并展示完整的RAG流程")
async def debug_rag_flow_kb(
        request: ChatQueryRequest,
        query_service: QueryService = Depends(get_query_service),
) -> DebugQueryResponse:
    """
    执行基于抽象知识库的完整 RAG 流程，并返回每个关键步骤的详细结果。
    通过消费 QueryService.rag_query_with_context 的 SSE 流来提取信息。
    """
    logger.info(f"Starting debug RAG flow for KB '{request.knowledge_base_id}' with query: '{request.question}'")

    # 注意：QueryService.rag_query_with_context 已经使用抽象 KB Service
    # 确保传入请求的 rag_config 包含最新的配置
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
            # 使用通用的 KB Retrieval Error
            # 假设异常有一个 code 属性 (如 BailianRagException)
            error_code = getattr(e, 'code', 'UNKNOWN')
            error_message = getattr(e, 'message', str(e))
            raise HTTPException(status_code=500, detail=f"KB Retrieval Error (Code {error_code}): {error_message}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    # --- 核心修改：重构最终引用的节点信息 ---
    
    final_response_content = full_response_content
    # 获取新的解耦引用结构
    final_referenced_docs = final_metadata.get("referenced_docs", [])
    citations = final_metadata.get("sentence_citations", [])
    
    # 1. 将所有引用的文档块详情映射到字典，以便通过 _id 查找
    doc_detail_map = {doc["_id"]: doc for doc in final_referenced_docs}
    
    # 2. 构建 final_retrieved_nodes (用于调试面板展示的节点列表)
    final_retrieved_nodes = []
    
    # 使用 final_referenced_docs 列表中的数据来构建调试节点
    for doc_id, doc_detail in doc_detail_map.items():
        # 注意：这里直接使用 doc_detail 的信息，因为它已经是清洗和截断后的
        final_retrieved_nodes.append({
            "score": "N/A (Citation)", # 分数在句子引用中无法精确获取
            "raw_score": "N/A",
            "node_id": doc_detail["_id"],
            "chunk_id": doc_detail["_id"],
            "source": doc_detail.get('file_name', 'N/A'),
            "text_snippet": doc_detail['referenced_chunk_text'], # 使用截断后的引用文本
            "metadata": {
                "kb_doc_id": doc_detail.get('kb_doc_id'),
                "material_id": doc_detail.get('material_id'),
                "source_type": doc_detail.get('source_type'),
                "page_label": doc_detail.get('page_label', 'N/A'),
            }
        })

    original_retrieved_nodes = [] # 保持为空，因为完整的检索结果在流式架构中难以捕获
    final_llm_prompt_placeholder = "The full prompt is constructed internally by QueryService. Please check service logs for 'final_prompt_for_llm'."
    
    # 4. 封装并返回所有调试信息
    return DebugQueryResponse(
        query_text=request.question,
        final_llm_prompt=final_llm_prompt_placeholder,
        original_retrieved_nodes=original_retrieved_nodes,
        final_retrieved_nodes=final_retrieved_nodes, # <--- 包含了解耦后的引用详情
        final_response_content=final_response_content,
        citations=citations # <--- 包含 refid 的句子引用列表
    )

@router.delete("/delete-by-metadata-kb", summary="[DEBUG-KB] 根据元数据删除文档")
def delete_by_metadata_kb(request: DeleteByMetadataRequest,
                               indexer_service: IndexerService = Depends(get_indexer_service)):
    """
    接收来自后端的指令，根据元数据过滤器删除知识库中的文档。
    该接口直接调用 IndexerService。
    """
    logger.info(f"Received request to delete documents from KB with filters: {request.filters}")
    try:
        # IndexerService.delete_nodes_by_metadata 已经适配了抽象接口
        result = indexer_service.delete_nodes_by_metadata(
            knowledge_base_id=request.knowledge_base_id,
            filters=request.filters
        )
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    except Exception as e:
        logger.error(f"KB Deletion request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during deletion: {str(e)}")
    
@router.delete("/clean-all-files", summary="[ADMIN] 批量删除所有文件和索引文档")
async def clean_all_files_kb(
    request: CleanAllFilesRequest,
    kb_service: AbstractKnowledgeBaseService = Depends(get_kb_service)
):
    """
    分页查询指定 CategoryId 下的所有文件，并执行：查一批、删一批。
    """
    
    index_id = request.knowledge_base_id
    category_id = request.category_id
    
    logger.warning(f"!!! STARTING DESTRUCTIVE CLEANUP (Paged)!!! Index: {index_id}, Category: {category_id}")
    
    total_found = 0
    total_deleted = 0
    delete_results = {"index_delete": [], "file_delete": [], "errors": []}
    
    try:
        # 1. 使用异步生成器进行分批查询和处理
        async for file_list, next_token, page_count, total_count in kb_service.list_files_iterator(
            category_id, 
            request.max_results_per_page
        ):
            if not file_list:
                logger.info(f"Page {page_count} returned no files.")
                continue

            current_page_count = len(file_list)
            total_found += current_page_count
            file_ids = [f.get('FileId') for f in file_list if f.get('FileId')]
            
            logger.info(f"Processing Page {page_count}: Found {current_page_count} files. Total processed so far: {total_found}")

            # 2. 优先删除知识库索引中的文档 (DeleteIndexDocument)
            if request.force_index_delete and file_ids:
                logger.info(f"Submitting DeleteIndexDocument jobs for {len(file_ids)} documents...")
                for file_id in file_ids:
                    try:
                        # kb_service.delete_document 实现的是 DeleteIndexDocument
                        result = await kb_service.delete_document(index_id, file_id) 
                        delete_results["index_delete"].append({"file_id": file_id, "status": "submitted", "req_id": result.get('request_id')})
                        # API 限流：10 次/秒，等待 0.1 秒，保证删除速度
                        await asyncio.sleep(0.2) 
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        logger.error(f"Index Delete Error for {file_id}: {error_msg}")
                        delete_results["errors"].append({"step": "index_delete", "file_id": file_id, "error": error_msg})

            # 3. 永久删除应用数据中的文件 (DeleteFile)
            logger.info(f"Starting permanent file deletion (DeleteFile) for {len(file_ids)} files...")
            
            for file_id in file_ids:
                try:
                    # kb_service.delete_file_permanently 实现的是 DeleteFile
                    result = await kb_service.delete_file_permanently(file_id)
                    delete_results["file_delete"].append({"file_id": file_id, "status": "success", "req_id": result.get('request_id')})
                    total_deleted += 1
                    # API 限流：10 次/秒，等待 0.1 秒
                    await asyncio.sleep(0.2) 
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"File Delete Error for {file_id}: {error_msg}")
                    delete_results["errors"].append({"step": "file_delete", "file_id": file_id, "error": error_msg})
            
            # 查完一页，删完一页，继续下一页的查询，无需等待。

        final_message = (
            f"Cleanup finished. Total files found: {total_found}. "
            f"Total files deleted from data center: {total_deleted}. "
            f"Errors encountered: {len(delete_results['errors'])}."
        )
        logger.warning(f"!!! CLEANUP REPORT !!! {final_message}")
        
        return {
            "status": "success", 
            "message": final_message, 
            "total_files_found": total_found,
            "total_files_deleted": total_deleted,
            "results": delete_results # 返回详细结果，方便调试
        }

    except Exception as e:
        logger.error(f"Critical error during file cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Critical cleanup error: {str(e)}")