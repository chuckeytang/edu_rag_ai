# api/endpoints/chat_history.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from models.schemas import AddChatMessageRequest # 从统一的 schemas 文件导入
from services.chat_history_service import chat_history_service 
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/add_message")
async def add_chat_message_to_chroma_api(request: AddChatMessageRequest):
    """
    将聊天消息同步到 ChromaDB 的聊天历史Collection。
    由 Spring 后端调用。
    """
    try:
        chat_history_service.add_chat_message_to_chroma(request.dict())
        return {"status": "success", "message": "Chat message added to ChromaDB."}
    except Exception as e:
        logger.error(f"Error adding chat message to ChromaDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add chat message to ChromaDB.")