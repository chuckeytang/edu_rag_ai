import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from services.query_service import query_service  

router = APIRouter()
@router.get("/indexed", summary="列出索引中的节点（chunk）")
def list_indexed(
    limit: int = Query(20, ge=1, le=1000, description="最多返回多少个节点"),
    file_name: Optional[str] = Query(
        None, description="按 metadata.file_name 关键字模糊过滤（不区分大小写）"
    ),
):
    """
    适配 llama-index ≥0.11（node_store）和 ≤0.10（docstore）两套结构。
    - 若两处都取不到节点，则直接到 Chroma collection 里 peek。
    """
    if query_service.index is None:
        raise HTTPException(status_code=404, detail="Index not initialized")

    sc = query_service.index.storage_context
    nodes: List = []
    vs = sc.vector_store
    if hasattr(vs, "_collection"):                # ChromaVectorStore
        col = vs._collection                      # type: ignore
        peek = col.peek(limit) 
        docs = peek.get("documents", [])
        metas = peek.get("metadatas", [])
        ids = peek.get("ids", [])
        logger.info(f"[DEBUG] Chroma peek 返回 {len(docs)} 条")
        # 直接构造返回
        items = []
        for _id, doc, meta in zip(ids, docs, metas):
            if file_name and file_name.lower() not in (meta.get("file_name") or "").lower():
                continue
            items.append({
                "node_id": meta.get("doc_id") or meta.get("id_") or "",
                "chroma_id": _id,
                "file_name": meta.get("file_name"),
                "page_label": meta.get("page_label"),
                "text_snippet": (doc or "")[:200] + ("..." if doc and len(doc) > 200 else "")
            })
        return items[:limit]

@router.get("/indexed/{chroma_id}", summary="查看单个节点的完整内容与元数据")
def get_node(chroma_id: str):
    if query_service.index is None:
        raise HTTPException(status_code=404, detail="Index not initialized")

    col = query_service.index.storage_context.vector_store._collection  # type: ignore
    res = col.get(ids=[chroma_id], include=["documents", "metadatas"])

    if not res["ids"]:
        raise HTTPException(status_code=404, detail="chroma_id not found")

    return {
        "chroma_id": res["ids"][0],
        "metadata":  res["metadatas"][0],
        "text":      res["documents"][0],
    }
