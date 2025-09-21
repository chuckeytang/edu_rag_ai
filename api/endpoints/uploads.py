import os
import logging
import httpx
from typing import Dict, List, Optional
 
from fastapi import APIRouter, Depends, Query, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from core.rag_config import RagConfig
from services.chat_history_service import ChatHistoryService
from services.document_service import DocumentService
from services.document_oss_service import DocumentOssService
from services.indexer_service import IndexerService
from services.task_manager_service import TaskManagerService
from services.oss_service import OssService
from services.query_service import QueryService 
from llama_index.core.schema import Document
from models.schemas import (
    DeleteByMetadataRequest,
    TaskStatus,
    UploadFromTextRequest,
    UploadFromOssRequest,
)
from api.dependencies import (
    get_indexer_service, 
    get_document_oss_service,
    get_task_manager_service,
    get_document_service
)

logger = logging.getLogger(__name__)
router = APIRouter()


# 核心业务逻辑包装函数
async def process_task_wrapper(request: UploadFromOssRequest, 
                         task_id: str,
                         document_oss_service: DocumentOssService,
                         task_manager_service: TaskManagerService):
    """
    负责调用业务服务层的方法，并把 task_id 传过去。
    所有的业务逻辑、流程控制、进度汇报和最终状态设置，都由被调用的服务自己完成。
    """
    logger.info(f"[TASK_ID: {task_id}] Background task wrapper started. Delegating all logic to DocumentOssService...")
    try:
        await document_oss_service.process_new_oss_file(request, task_id, request.rag_config)
    except Exception as e:
        logger.error(f"[TASK_ID: {task_id}] A critical unhandled exception escaped the service layer: {e}", exc_info=True)
        task_manager_service.finish_task(task_id, "error", result={"message": "A critical and unexpected error occurred in the service layer."})


# --- 文件上传：从OSS异步上传 ---
@router.post("/upload-from-oss", response_model=TaskStatus, status_code=202)
async def upload_from_oss(request: UploadFromOssRequest, 
                          background_tasks: BackgroundTasks,
                          document_oss_service: DocumentOssService = Depends(get_document_oss_service),
                          task_manager_service: TaskManagerService = Depends(get_task_manager_service)):
    """
    接收来自后端的指令，启动后台任务，将OSS文件URL提交给火山引擎进行索引。
    """
    initial_data = {
        "file_name": request.metadata.file_name,
        "file_key": request.file_key
    }
    initial_status = task_manager_service.create_task(
        task_type="document_indexing",
        initial_message=f"Task scheduled for file '{request.metadata.file_name}'.",
        initial_data=initial_data
    )
    task_id = initial_status.task_id
    logger.info(f"Received request to upload from OSS for file: '{request.metadata.file_name}'. Assigning Task ID: {task_id}.")
    
    background_tasks.add_task(process_task_wrapper, 
                              request, 
                              task_id,
                              document_oss_service,
                              task_manager_service)
    logger.info(f"Task {task_id} successfully scheduled.")
    return initial_status

# --- 元数据更新与发布（已移除） ---
# 由于元数据更新和权限发布逻辑已不再适用，相关的 API 和后台任务被移除。

@router.post("/delete-by-metadata", summary="[同步操作] 根据元数据删除文档")
def delete_by_metadata(request: DeleteByMetadataRequest,
                       indexer_service: IndexerService = Depends(get_indexer_service)):
    """
    接收来自后端的指令，根据元数据过滤器删除知识库中的文档。
    这是一个同步操作。
    """
    logger.info(f"Received request to delete documents with filters: {request.filters}")
    try:
        result = indexer_service.delete_nodes_by_metadata(
            knowledge_base_id=request.knowledge_base_id,
            filters=request.filters
        )
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Deletion request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during deletion: {str(e)}")

# --- 文本上传：异步上传（用于PaperCut） ---
JAVA_CALLBACK_BASEURL = os.environ.get("JAVA_CALLBACK_BASEURL")
JAVA_TEXTRAG_ENDPOINT = os.environ.get("JAVA_TEXTRAG_ENDPOINT")
JAVA_CALLBACK_URL = f"{JAVA_CALLBACK_BASEURL}{JAVA_TEXTRAG_ENDPOINT}"

async def process_text_indexing_task(request: UploadFromTextRequest, task_id: str):
    """
    这是一个后台任务，负责处理文本内容的索引，并在完成后发送回调。
    """
    logger.info(f"[TASK_ID: {task_id}] Starting text indexing task for paper_cut_id: {request.metadata.paper_cut_id}")
    
    final_status = "error"
    final_message = "An unexpected error occurred."
    
    try:
        metadata_payload = request.metadata.model_dump(by_alias=True, exclude_none=True)
        metadata_payload["page_label"] = f"PaperCut-{request.metadata.paper_cut_id}" 

        doc = Document(
            text=request.text_content,
            metadata=metadata_payload
        )
        
        indexer_service = get_indexer_service()
        indexer_service.add_documents_to_index(
            [doc], 
            knowledge_base_id=request.knowledge_base_id, 
            rag_config=request.rag_config
        )
        
        final_status = "success"
        final_message = "试题索引成功。"
        logger.info(f"[TASK_ID: {task_id}] Successfully indexed text content for paper_cut_id: {request.metadata.paper_cut_id}")

    except Exception as e:
        final_status = "error"
        final_message = f"索引试题时发生错误: {str(e)}"
        logger.error(f"[TASK_ID: {task_id}] Failed to index text content: {e}", exc_info=True)
        
    finally:
        callback_payload = {
            "material_type": "paper_cut",
            "material_id": request.metadata.paper_cut_id,
            "status": final_status,
            "message": final_message
        }
        
        try:
            if not JAVA_CALLBACK_URL:
                logger.error("JAVA_CALLBACK_URL is not configured. Skipping callback.")
                return

            async with httpx.AsyncClient() as client:
                response = await client.post(JAVA_CALLBACK_URL, json=callback_payload)
                if response.status_code != 200:
                    logger.error(f"Failed to send callback to Java. Status: {response.status_code}. Response: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send callback to Java: {e}", exc_info=True)

@router.post("/upload-from-text", status_code=202)
async def upload_from_text_async(
    request: UploadFromTextRequest,
    background_tasks: BackgroundTasks,
    task_manager_service: TaskManagerService = Depends(get_task_manager_service)
):
    """
    启动一个后台任务，从文本内容创建RAG索引。
    """
    initial_status = task_manager_service.create_task(
        task_type="paper_cut_indexing",
        initial_message=f"Task scheduled for paper_cut_id: {request.metadata.paper_cut_id}.",
        initial_data={"paper_cut_id": request.metadata.paper_cut_id}
    )
    task_id = initial_status.task_id
    logger.info(f"Received request to index text for paper_cut_id: {request.metadata.paper_cut_id}. Assigning Task ID: {task_id}.")

    background_tasks.add_task(process_text_indexing_task, request, task_id)
    logger.info(f"Task {task_id} successfully scheduled.")
    return initial_status

# --- Polling Endpoint ---
@router.get("/status/{task_id}", response_model=TaskStatus, summary="查询后台任务的状态")
async def get_task_status(task_id: str,
                          task_manager_service: TaskManagerService = Depends(get_task_manager_service)):
    """
    Polls for the status and result of a background processing task.
    """
    status = task_manager_service.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")
    return status