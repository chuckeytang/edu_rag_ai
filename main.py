# main.py
from fastapi import FastAPI
from core.config import settings
from api.api_v1 import api_router
from fastapi.middleware.cors import CORSMiddleware
import logging
from core.logging_config import setup_app_logging # 假设这个模块存在并配置了日志
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

# 导入 asynccontextmanager 用于 FastAPI 的生命周期事件
from contextlib import asynccontextmanager

# 导入所有需要预先初始化的依赖函数
from api.dependencies import (
    initialize_global_chroma_client, # 新增：引入 ChromaDB 全局初始化函数
    get_query_service,
    get_chat_history_service,
    get_indexer_service,
    get_ai_extraction_service,
    get_document_oss_service,
    # 根据你的需求，确保所有在第一个请求到来前需要就绪的服务都在这里被调用
)

# 在应用启动前设置日志
setup_app_logging(level=logging.INFO, log_file="logs/app.log")
app_logger = logging.getLogger(__name__) 
app_logger.info("Starting FastAPI application...")

# --- FastAPI 应用生命周期管理器 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理器。
    用于在应用启动时初始化核心服务，并在关闭时进行清理。
    """
    app_logger.info("Application startup event: Starting core service initialization...")
    try:
        # 步骤 1: 首先初始化全局的 ChromaDB 客户端。
        # 这是为了确保它在所有其他依赖项之前被安全地创建。
        initialize_global_chroma_client() 
        
        # 步骤 2: 强制初始化所有核心服务。
        # 通过调用这些函数的依赖注入链，它们会按需获取已初始化的 ChromaDB 客户端和LLM/Embedding模型。
        get_query_service()
        get_chat_history_service()
        get_indexer_service()
        get_ai_extraction_service()
        get_document_oss_service()
        # 请根据你的应用逻辑，确保所有在第一个请求到来前需要就绪的服务都在这里被调用一次。
        # 否则，第一次请求仍可能触发初始化。

        app_logger.info("All core services initialized during startup.")
        yield # 应用启动完成，开始处理请求

    except Exception as e:
        app_logger.critical(f"FATAL: Failed to initialize application services during startup: {e}", exc_info=True)
        # 记录关键错误后，可以选择重新抛出，阻止应用启动
        raise

    # 应用关闭时执行的代码 (在 'yield' 之后)
    app_logger.info("Application shutdown event: Cleaning up resources (if any)...")
    # 如果 ChromaDB PersistentClient 有明确的关闭方法来释放文件锁，可以在这里调用
    # 例如：
    # from api.dependencies import _global_chroma_client_instance
    # if _global_chroma_client_instance:
    #     try:
    #         # 假设 ChromaDB 客户端有 .close() 方法 (目前PersistentClient没有，但未来可能有)
    #         # _global_chroma_client_instance.close() 
    #         app_logger.info("ChromaDB PersistentClient closed.")
    #     except Exception as e:
    #         app_logger.error(f"Error closing ChromaDB PersistentClient: {e}", exc_info=True)


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for querying exam papers and syllabus documents using LLM RAG",
    version=settings.VERSION,
    docs_url="/docs",        # Swagger UI 路径
    redoc_url="/redoc",      # ReDoc 文档路径
    lifespan=lifespan        # 关键：将生命周期管理器添加到 FastAPI 应用
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    自定义请求验证异常处理程序，用于打印详细的 Pydantic 错误日志。
    """
    print("--- Pydantic Validation Error Details ---")
    print(f"Request URL: {request.url}")
    print(f"Request Method: {request.method}")
    print(f"Request Body: {await request.body()}")  # 打印原始请求体
    
    # 打印详细的错误列表
    errors = exc.errors()
    for error in errors:
        loc = " -> ".join(map(str, error['loc']))
        msg = error['msg']
        type_ = error['type']
        print(f"  Field: {loc}")
        print(f"  Message: {msg}")
        print(f"  Error Type: {type_}")
        print("-" * 20)

    print("--- End of Validation Error ---")
    
    # 返回一个格式化的 JSON 响应给客户端
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": errors})
    )

if __name__ == '__main__':
    # =================================================================
    # 把日志配置移动到这里
    # 这将只在直接运行时生效，为 uvicorn.run 提供日志支持
    # 注意：如果 setup_app_logging 已经配置了 StreamHandler，这里可能不需要重复配置
    # 但为了确保 uvicorn 自身的日志输出，保留此段是安全的。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )

    import uvicorn
    # uvicorn.run 会使用上面刚刚配置好的日志设置
    uvicorn.run('main:app', host="0.0.0.0", port=8010, reload=False)