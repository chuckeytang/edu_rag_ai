# api/endpoints/config.py
from fastapi import APIRouter, Body, HTTPException
from api.dependencies import update_rag_config
from core.rag_config import RagConfig
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/update-rag-config")
async def update_rag_configuration(config: RagConfig):
    """
    接收并更新RAG的运行时配置。
    """
    try:
        # 将Pydantic模型转换为JSON字符串，传递给依赖函数
        config_json_str = config.model_dump_json()
        
        # 调用位于 dependencies.py 的更新函数
        success = update_rag_config(config_json_str)

        if success:
            logger.info(f"RAG configuration updated successfully.")
            return {"status": "success", "message": "RAG configuration updated successfully."}
        else:
            raise HTTPException(status_code=400, detail="Failed to update RAG configuration due to an internal error.")

    except Exception as e:
        logger.error(f"Failed to update RAG configuration: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid configuration format: {e}")