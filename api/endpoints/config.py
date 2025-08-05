# api/endpoints/config.py (新增)
from fastapi import APIRouter, Body
from api.dependencies import update_rag_config
from core.rag_config import RagConfig
import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/update-rag-config")
async def update_rag_configuration(config: RagConfig):
    """
    接收并更新RAG的运行时配置。
    """
    global _current_rag_config
    try:
        _current_rag_config = config
        logger.info(f"RAG configuration updated to: {config.model_dump_json(indent=2)}")
        return {"status": "success", "message": "RAG configuration updated successfully."}
    except Exception as e:
        logger.error(f"Failed to update RAG configuration: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid configuration format: {e}")