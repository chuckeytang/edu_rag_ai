# api/api_v1.py
from fastapi import APIRouter

from api.endpoints import chat_history, extraction, queries, uploads, mcp, debug_index

api_router = APIRouter()

api_router.include_router(
    queries.router, prefix="/queries", tags=["queries"]
)

api_router.include_router(
    uploads.router, prefix="/uploads", tags=["uploads"]
)

api_router.include_router(
    extraction.router, prefix="/extraction", tags=["extraction"]
)

api_router.include_router(
    chat_history.router, prefix="/chat_history", tags=["chat_history"]
)

api_router.include_router(
    mcp.router, prefix="/mcp", tags=["mcp"]
)

api_router.include_router(
    debug_index.router, prefix="/debug", tags=["debug"]
)