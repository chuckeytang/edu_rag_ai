# api/endpoints/chat_history.py
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any

from models.schemas import AddChatMessageRequest, DeleteChatMessagesRequest, UpdateChatMessageRequest 
from services.chat_history_service import ChatHistoryService
from api.dependencies import get_chat_history_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/add-chat-message")
async def add_chat_message_to_chroma_api(request: AddChatMessageRequest,
                                         chat_history_service: ChatHistoryService = Depends(get_chat_history_service)):
    """
    将聊天消息同步到 ChromaDB 的聊天历史Collection。
    由 Spring 后端调用。
    """
    try:
        request_dict = request.dict()
        rag_config = request_dict.pop("rag_config", None)
        chat_history_service.add_chat_message_to_chroma(request_dict, rag_config=rag_config)
        return {"status": "success", "message": "Chat message added to ChromaDB."}
    except Exception as e:
        logger.error(f"Error adding chat message to ChromaDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add chat message to ChromaDB.")
    
@router.post("/delete-chat-messages") 
async def delete_chat_messages_api(request: DeleteChatMessagesRequest,
                                   chat_history_service: ChatHistoryService = Depends(get_chat_history_service)):
    """
    根据会话ID和用户ID删除 ChromaDB 中的所有聊天消息。
    由 Spring 后端调用。
    注意：此接口使用POST方法以方便传递请求体。
    """
    try:
        result = chat_history_service.delete_chat_messages(
            session_id=request.session_id,
            account_id=request.account_id
        )
        if result.get("status") == "success":
            return {"status": "success", "message": result.get("message")}
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get("message", "Failed to delete chat messages."))
    except Exception as e:
        logger.error(f"Error deleting chat messages from ChromaDB: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete chat messages from ChromaDB.")