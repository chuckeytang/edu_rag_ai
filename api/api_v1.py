# api/api_v1.py
from fastapi import APIRouter

# 现有 endpoint
from api.endpoints import documents, extraction, queries, uploads
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