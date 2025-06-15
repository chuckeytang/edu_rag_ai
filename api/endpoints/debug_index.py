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
    if query_service.index is None:
        raise HTTPException(status_code=404, detail="Index not initialized")

    docstore = query_service.index.storage_context.docstore

    node_dict = docstore.docs                   # type: Dict[str, BaseNode]
    nodes = list(node_dict.values())
    logger.info(f"[DEBUG] docstore 共 {len(nodes)} 个节点")

    # 关键字过滤
    if file_name:
        keyword = file_name.lower()
        before = len(nodes)
        nodes = [
            n for n in nodes
            if keyword in (n.metadata.get("file_name") or "").lower()
        ]
        logger.info(f"[DEBUG] 过滤 file_name='{file_name}' 后剩 {len(nodes)} / {before}")

    # 截断
    nodes = nodes[:limit]
    logger.info(f"[DEBUG] 最终返回 {len(nodes)} 个节点")

    # 如果还是空，记录一次示例 metadata，方便排查
    if not nodes:
        sample_meta = next(iter(docstore.docs.values()), None)
        logger.warning(f"[DEBUG] 返回空。样例 metadata: {getattr(sample_meta, 'metadata', {})}")

    return [
        {
            "node_id": n.node_id,
            "file_name": n.metadata.get("file_name"),
            "page_label": n.metadata.get("page_label"),
            "text_snippet": n.text[:200] + ("..." if len(n.text) > 200 else ""),
        }
        for n in nodes
    ]

@router.get(
    "/indexed/{node_id}", summary="查看单个节点的完整内容与元数据"
)
def get_node(node_id: str):
    if query_service.index is None:
        raise HTTPException(status_code=404, detail="Index not initialized")

    docstore = query_service.index.storage_context.docstore
    node = docstore.get_node(node_id)

    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")

    return {
        "node_id": node.node_id,
        "metadata": node.metadata,
        "text": node.text,
    }
