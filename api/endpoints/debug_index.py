import logging
import json

from api.dependencies import get_indexer_service, get_query_service
from services.indexer_service import IndexerService

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional
from models.schemas import ChatQueryRequest, QueryRequest
from fastapi import APIRouter, Depends, HTTPException, Path, Query
import numpy as np
from services.query_service import QueryService  

router = APIRouter()
@router.get("/indexed", summary="列出索引中的节点（chunk）")
def list_indexed(
    collection_name: str = Query(..., description="要查询的Collection名称"),
    limit: int = Query(20, ge=1, le=1000, description="最多返回多少个节点"),
    file_name: Optional[str] = Query(
        None, description="按 metadata.file_name 关键字模糊过滤（不区分大小写）"
    ),
    # 修改参数名为 chroma_id
    chroma_id: Optional[str] = Query(
        None, description="要查询的特定节点在 ChromaDB 中的内部ID（通常与LlamaIndex的node_id一致）" # 更新描述
    ),
    title: Optional[str] = Query(
        None, description="按 metadata.title 关键字模糊过滤（不区分大小写）"
    ),
    query_service: Any = Depends(get_query_service) 
) -> List[Dict[str, Any]]:
    """
    从指定的ChromaDB Collection中直接“窥视”数据，用于调试。
    支持按Collection名称、返回数量、文件名/标题模糊过滤，以及特定节点ID精确查询。
    """
    try:
        col = query_service.chroma_client.get_collection(name=collection_name)
    except ValueError:
        logger.warning(f"Debug endpoint: Collection '{collection_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found in ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get collection '{collection_name}': {e}")

    items = []

    if chroma_id:
        # 如果提供了 chroma_id，则进行精确查询
        logger.info(f"Querying collection '{collection_name}' for chroma_id: {chroma_id}")
        
        try:
            get_results = col.get(
                ids=[chroma_id], 
                include=['documents', 'metadatas']
            )

            if get_results and get_results.get("ids") and get_results["ids"][0]:
                _id = get_results["ids"][0] 
                doc = get_results["documents"][0]
                meta = get_results["metadatas"][0]

                node_content_meta = {}
                if '_node_content' in meta and isinstance(meta['_node_content'], str):
                    try:
                        node_content_data = json.loads(meta['_node_content'])
                        node_content_meta = node_content_data.get('metadata', {})
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse _node_content as JSON for chroma_id: {chroma_id}")
                
                effective_node_id = meta.get("doc_id") or meta.get("ref_doc_id") or _id

                formatted_item = {
                    "node_id": effective_node_id,
                    "chroma_id": _id,
                    "metadata": meta,
                    "file_name": meta.get("file_name") or node_content_meta.get("file_name"), 
                    "page_label": meta.get("page_label") or node_content_meta.get("page_label"), 
                    "text": doc
                }
                items.append(formatted_item)
            else:
                raise HTTPException(status_code=404, detail=f"Node with ChromaDB ID '{chroma_id}' not found in collection '{collection_name}'.")

        except Exception as e:
            logger.error(f"Failed to retrieve specific node '{chroma_id}' from collection '{collection_name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to retrieve specific node: {e}")

    else:
        # 如果没有提供 chroma_id，则执行原有的 peek 和过滤逻辑
        logger.info(f"Peeking into collection '{collection_name}' with limit {limit}...")
        peek_result = col.peek(limit=limit * 5) # Peek more to allow for filtering
        
        ids = peek_result.get("ids") or []
        docs = peek_result.get("documents") or []
        metas = peek_result.get("metadatas") or []

        for _id, doc, meta in zip(ids, docs, metas):
            # 同样解析 _node_content 中的元数据，以获取更完整的file_name和title
            node_content_meta = {}
            if '_node_content' in meta and isinstance(meta['_node_content'], str):
                try:
                    node_content_data = json.loads(meta['_node_content'])
                    node_content_meta = node_content_data.get('metadata', {})
                except json.JSONDecodeError:
                    pass 
            
            # 获取实际用于过滤的文件名和标题 (优先 metadata，其次 _node_content)
            current_file_name = (meta.get("file_name") or node_content_meta.get("file_name") or "").lower()
            current_title = (meta.get("title") or node_content_meta.get("title") or "").lower()

            # --- 应用文件名过滤 ---
            if file_name and file_name.lower() not in current_file_name:
                continue
            
            # --- 应用标题过滤 ---
            if title and title.lower() not in current_title:
                continue
                
            effective_node_id = meta.get("doc_id") or meta.get("ref_doc_id") or _id

            items.append({
                "node_id": effective_node_id,
                "chroma_id": _id,
                "metadata": meta,
                "file_name": current_file_name, # 返回时使用已处理的
                "page_label": meta.get("page_label") or node_content_meta.get("page_label"),
                "text": doc
            })
            
            if len(items) >= limit:
                break
                
    return items

@router.get("/indexed/{chroma_id}", summary="查看单个节点的完整内容与元数据")
def get_node(chroma_id: str,
            collection_name: str = Query(..., description="该节点所在的Collection名称"),
            query_service: QueryService = Depends(get_query_service)):
    """
    从指定的ChromaDB Collection中，按ID获取单个节点的详细信息。
    """
    try:
        # --- 核心修改 2: 同样，先按名称获取指定的 collection ---
        col = query_service.chroma_client.get_collection(name=collection_name)
    except ValueError:
        logger.warning(f"Debug endpoint: Collection '{collection_name}' not found for node id '{chroma_id}'.")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found in ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get collection '{collection_name}': {e}")

    # --- 后续的 get 和数据处理逻辑保持不变 ---
    res = col.get(ids=[chroma_id], include=["documents", "metadatas"])

    if not res.get("ids"):
        raise HTTPException(status_code=404, detail=f"Node with chroma_id '{chroma_id}' not found in collection '{collection_name}'.")

    return {
        "chroma_id": res["ids"][0],
        "metadata":  res["metadatas"][0],
        "text":      res["documents"][0],
    }


@router.post("/retrieve-with-filters", summary="[DEBUG] 测试带过滤的节点召回")
def debug_retrieve_with_filters(request: QueryRequest, 
                            query_service: QueryService = Depends(get_query_service),
                            indexer_service: IndexerService = Depends(get_indexer_service)):
    """
    一个用于调试的端点。
    它只执行召回步骤，并返回召回的节点列表及其元数据，
    帮助您确认Filter是否按预期工作。
    """
    try:
        # 创建一个可变的 filters 字典，以便合并
        combined_filters = request.filters.copy() if request.filters else {}

        if request.target_file_ids:
            try:
                material_ids_int = [int(mid) for mid in request.target_file_ids]
                # 将 target_file_ids 合并到 material_id 过滤器中
                if "material_id" in combined_filters and isinstance(combined_filters["material_id"], dict) and "$in" in combined_filters["material_id"]:
                    current_material_ids = combined_filters["material_id"]["$in"]
                    combined_filters["material_id"]["$in"] = list(set(current_material_ids + material_ids_int))
                else:
                    combined_filters["material_id"] = {"$in": material_ids_int}
            except ValueError:
                logger.warning("Debug endpoint: Invalid material_id in target_file_ids. Ignoring file filter.")

        retrieved_nodes = query_service.retrieve_with_filters(
            question=request.question,
            collection_name=request.collection_name,
            filters=combined_filters, 
            similarity_top_k=request.similarity_top_k
        )

        results = []
        for node_with_score in retrieved_nodes:
            current_node_metadata = node_with_score.node.metadata 

            try:
                col = indexer_service.chroma_client.get_collection(name=request.collection_name)
                latest_meta_from_db = col.get(ids=[node_with_score.node.node_id], include=["metadatas"])['metadatas'][0]
                current_node_metadata = latest_meta_from_db
            except Exception as debug_e:
                logger.error(f"DEBUG: Failed to get latest metadata from DB for {node_with_score.node.node_id}: {debug_e}")

            score_to_append = float(node_with_score.score) # <--- 将 score 强制转换为 Python float
            
            # 对 metadata 进行深度拷贝，并转换其中的 NumPy 类型
            cleaned_metadata = {}
            for key, value in current_node_metadata.items():
                if isinstance(value, np.ndarray): # 如果是 NumPy 数组
                    cleaned_metadata[key] = value.tolist() # 转换为 Python 列表
                elif isinstance(value, (np.float32, np.float64)): # <--- 移除 np.float
                    cleaned_metadata[key] = float(value) 
                elif isinstance(value, (np.int32, np.int64)): 
                    cleaned_metadata[key] = int(value) 
                else:
                    cleaned_metadata[key] = value 
                    
            results.append({
                "score": score_to_append,
                "node_id": node_with_score.node.node_id,
                "text_snippet": node_with_score.get_text()[:300] + "...",
                "metadata": cleaned_metadata
            })

        return results

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Debug retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/delete-material/{material_id}", summary="根据 material_id 删除所有相关文档块（chunk）")
def delete_chunks_by_material_id(
    material_id: int = Path(..., description="要删除的文档的 material_id"),
    collection_name: str = Query(..., description="目标Collection的名称"),
    indexer_service: IndexerService = Depends(get_indexer_service) # 依赖注入 IndexerService
) -> Dict[str, Any]:
    """
    根据给定的 material_id，从指定 Collection 中删除所有相关的文档块（chunk）。
    """
    logger.info(f"Received request to delete chunks for material_id: {material_id} in collection: '{collection_name}'")
    
    try:
        # IndexerService 中的 delete_nodes_by_metadata 方法已经处理了过滤逻辑
        filters = {"material_id": material_id} # material_id 存储为整数，直接传递整数
        
        result = indexer_service.delete_nodes_by_metadata(collection_name=collection_name, filters=filters)
        
        # delete_nodes_by_metadata 已经返回了清晰的状态和信息
        if result.get("status") == "success":
            logger.info(f"Successfully processed delete request for material_id {material_id}: {result.get('message')}")
            return {"status": "success", "message": result.get("message")}
        else:
            logger.error(f"Failed to delete chunks for material_id {material_id}: {result.get('message')}")
            raise HTTPException(status_code=500, detail=result.get("message"))
            
    except HTTPException: # 如果 delete_nodes_by_metadata 内部抛出 HTTPException，直接向上抛
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during deletion for material_id {material_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
