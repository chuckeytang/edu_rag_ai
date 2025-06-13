from fastapi import APIRouter
from api.endpoints import documents, queries, uploads
api_router = APIRouter()
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(queries.router, prefix="/queries", tags=["queries"])
api_router.include_router(uploads.router, prefix="/uploads", tags=["uploads"])