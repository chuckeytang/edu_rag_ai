from fastapi import FastAPI
from core.config import settings
from api.api_v1 import api_router
from fastapi.middleware.cors import CORSMiddleware
import logging
from core.logging_config import setup_app_logging
setup_app_logging(level=logging.INFO, log_file="logs/app.log")
app_logger = logging.getLogger(__name__) 
app_logger.info("Starting FastAPI application...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for querying exam papers and syllabus documents using LLM RAG",
    version=settings.VERSION,
    docs_url="/docs",        # Swagger UI 路径
    redoc_url="/redoc"       # ReDoc 文档路径
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
    # =================================================================
    # 把日志配置移动到这里
    # 这将只在直接运行时生效，为 uvicorn.run 提供日志支持
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )

    import uvicorn
    # uvicorn.run 会使用上面刚刚配置好的日志设置
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=False)