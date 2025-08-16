# api/api_v1.py
from logging import config
from fastapi import APIRouter

# 现有 endpoint
from api.endpoints import chat_history, documents, extraction, queries, uploads, config, mcp
# 新增调试 endpoint
from api.endpoints import debug_index

api_router = APIRouter()

api_router.include_router(
    documents.router, prefix="/documents", tags=["documents"]
)
api_router.include_router(
    queries.router, prefix="/queries", tags=["queries"]
)
api_router.include_router(
    uploads.router, prefix="/uploads", tags=["uploads"]
)
api_router.include_router(
    debug_index.router, prefix="/debug", tags=["debug"]
)
api_router.include_router(
    extraction.router, prefix="/extraction", tags=["extraction"]
)
api_router.include_router(
    chat_history.router, prefix="/chat_history", tags=["chat_history"]
)
api_router.include_router(
    config.router, prefix="/config", tags=["config"]
)
api_router.include_router(
    mcp.router, prefix="/mcp", tags=["mcp"]
)