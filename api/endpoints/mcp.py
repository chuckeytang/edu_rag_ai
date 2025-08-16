# api/endpoints/mcp.py
from fastapi import APIRouter, Depends, HTTPException, Body
from services.mcp_service import MCPService
import logging
from api.dependencies import get_mcp_service
from models.schemas import MCPQueryRequest # 稍后新增这个 Pydantic 模型

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/dispatch-command")
async def dispatch_mcp_command(request: MCPQueryRequest,
                               mcp_service: MCPService = Depends(get_mcp_service)):
    """
    一个通用的 MCP 调度接口，接收来自 Java 端的请求，并返回一个 MCP 指令。
    """
    logger.info(f"Received request for MCP dispatch. Question: '{request.question}'")
    
    try:
        # 调用 MCPService 生成指令
        mcp_command = await mcp_service.generate_mcp_command(request.question)
        
        # 返回 JSON 格式的指令
        return mcp_command

    except Exception as e:
        logger.error(f"Error processing MCP dispatch request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process request and generate MCP command.")