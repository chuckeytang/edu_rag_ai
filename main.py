from fastapi import FastAPI
from core.config import settings
from api.api_v1 import api_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for querying exam papers and syllabus documents using LLM RAG",
    version=settings.VERSION
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.API_V1_STR)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)