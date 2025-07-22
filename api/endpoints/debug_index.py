import logging
import json

from api.dependencies import get_indexer_service, get_query_service
from services.indexer_service import IndexerService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from typing import Any, Dict, List, Optional
from models.schemas import QueryRequest
from fastapi import APIRouter, Depends, HTTPException, Query

from services.query_service import QueryService  

router = APIRouter()
@router.get("/indexed", summary="列出索引中的节点（chunk）")
def list_indexed(
    collection_name: str = Query(..., description="要查询的Collection名称"),
    limit: int = Query(20, ge=1, le=1000, description="最多返回多少个节点"),
    file_name: Optional[str] = Query(
        None, description="按 metadata.file_name 关键字模糊过滤（不区分大小写）"
    ),
    node_id: Optional[str] = Query(
        None, description="要查询的特定节点ID（精确匹配doc_id或ref_doc_id）" # 新增参数
    ),
    query_service: Any = Depends(get_query_service) # 暂时用Any，实际应该用QueryService类型
) -> List[Dict[str, Any]]:
    """
    从指定的ChromaDB Collection中直接“窥视”数据，用于调试。
    支持按Collection名称、返回数量、文件名模糊过滤，以及特定节点ID精确查询。
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

    if node_id:
        # 如果提供了 node_id，则进行精确查询
        logger.info(f"Querying collection '{collection_name}' for node_id: {node_id}")
        
        try:
            # ChromaDB 的 get 方法接受一个 ID 列表
            get_results = col.get(
                ids=[node_id], # 尝试直接用 node_id 作为 ChromaDB 内部 ID
                include=['documents', 'metadatas']
            )

            if get_results and get_results.get("ids") and get_results["ids"][0]:
                _id = get_results["ids"][0] # ChromaDB 的内部ID
                doc = get_results["documents"][0]
                meta = get_results["metadatas"][0]

                # 提取 LlamaIndex 的 doc_id / ref_doc_id
                # 有些 metadata 中的 '_node_content' 可能是 JSON 字符串，需要解析
                node_content_meta = {}
                if '_node_content' in meta and isinstance(meta['_node_content'], str):
                    try:
                        node_content_data = json.loads(meta['_node_content'])
                        node_content_meta = node_content_data.get('metadata', {})
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse _node_content as JSON for node_id: {node_id}")
                
                # 优先使用 metadata 中的 doc_id 或 ref_doc_id，如果LlamaIndex有提供
                # 否则使用 ChromaDB 内部的 _id 作为 node_id 的 fallback
                effective_node_id = meta.get("doc_id") or meta.get("ref_doc_id") or _id

                formatted_item = {
                    "node_id": effective_node_id,
                    "chroma_id": _id,
                    "metadata": meta,
                    "file_name": meta.get("file_name") or node_content_meta.get("file_name"), # 从内外层都找文件名
                    "page_label": meta.get("page_label") or node_content_meta.get("page_label"), # 从内外层都找页码
                    "text_snippet": (doc or "")[:200] + ("..." if doc and len(doc) > 200 else "")
                }
                items.append(formatted_item)
            else:
                # 如果直接用 node_id 没找到，尝试通过 metadata 查找 (更通用但可能需要 query)
                # 由于 col.query 要求 non-empty query_texts，这里不便直接用。
                # 只能依赖 col.get(ids=[node_id]) 的匹配能力。
                raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found in collection '{collection_name}'. (Note: ID must be ChromaDB's internal ID or directly mapped)")

        except Exception as e:
            logger.error(f"Failed to retrieve specific node '{node_id}' from collection '{collection_name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to retrieve specific node: {e}")

    else:
        # 如果没有提供 node_id，则执行原有的 peek 和过滤逻辑
        logger.info(f"Peeking into collection '{collection_name}' with limit {limit}...")
        peek_result = col.peek(limit=limit * 5) # Peek more to allow for filtering
        
        # 安全地解包，避免因数据为空而出错
        ids = peek_result.get("ids") or []
        docs = peek_result.get("documents") or []
        metas = peek_result.get("metadatas") or []

        for _id, doc, meta in zip(ids, docs, metas):
            # 如果需要文件名过滤，并且当前项不匹配，则跳过
            if file_name and file_name.lower() not in (meta.get("file_name") or "").lower():
                continue
            
            # 同样解析 _node_content 中的元数据
            node_content_meta = {}
            if '_node_content' in meta and isinstance(meta['_node_content'], str):
                try:
                    node_content_data = json.loads(meta['_node_content'])
                    node_content_meta = node_content_data.get('metadata', {})
                except json.JSONDecodeError:
                    pass # 不解析失败的

            # 优先使用 metadata 中的 doc_id 或 ref_doc_id，如果LlamaIndex有提供
            effective_node_id = meta.get("doc_id") or meta.get("ref_doc_id") or _id

            items.append({
                "node_id": effective_node_id,
                "chroma_id": _id,
                "metadata": meta,
                "file_name": meta.get("file_name") or node_content_meta.get("file_name"),
                "page_label": meta.get("page_label") or node_content_meta.get("page_label"),
                "text_snippet": (doc or "")[:200] + ("..." if doc and len(doc) > 200 else "")
            })
            
            # 因为我们 peek 了更多数据用于过滤，所以在这里手动限制返回数量
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
def debug_retrieve_with_filters(request: QueryRequest, # 确保这里的 QueryRequest 包含了 target_file_ids
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
            filters=combined_filters, # <--- 传递合并后的 filters
            similarity_top_k=request.similarity_top_k
        )

        # ... (后续的 results 格式化和返回逻辑保持不变)
        results = []
        for node_with_score in retrieved_nodes:
            current_node_metadata = node_with_score.node.metadata 

            try:
                col = indexer_service.chroma_client.get_collection(name=request.collection_name)
                latest_meta_from_db = col.get(ids=[node_with_score.node.node_id], include=["metadatas"])['metadatas'][0]
                current_node_metadata = latest_meta_from_db
            except Exception as debug_e:
                logger.error(f"DEBUG: Failed to get latest metadata from DB for {node_with_score.node.node_id}: {debug_e}")

            results.append({
                "score": node_with_score.score,
                "node_id": node_with_score.node.node_id,
                "text_snippet": node_with_score.get_text()[:300] + "...",
                "metadata": current_node_metadata
            })

        return results

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Debug retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
