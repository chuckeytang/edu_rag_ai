import os
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from services.document_service import document_service
from services.document_oss_service import document_oss_service
from services.oss_service import oss_service
from services.query_service import query_service
from models.schemas import AddChatMessageRequest, DeleteByMetadataRequest, UploadResponse, UploadFromOssRequest, UpdateMetadataRequest, UpdateMetadataResponse
import shutil
router = APIRouter()

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

processing_task_results = {}

# ---- 单文件上传保持你刚改好的版本 ----
@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    return await _handle_single_file(file)

# ---- 批量上传：调用同一套业务逻辑 _handle_single_file ----
@router.post("/upload-multiple", response_model=List[UploadResponse])
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    responses: List[UploadResponse] = []
    seen_hashes: set[str] = set()  # 本批次内已处理的文件 Hash

    for file in files:
        try:
            # ⭐ 先预计算 Hash，避免一个请求里两份相同文件入库
            file_hash = await document_service._calculate_file_hash(file)
            await file.seek(0)  # 重要：复位指针，后续还能读取

            if file_hash in seen_hashes:
                # 同一批里重复
                responses.append(
                    UploadResponse(
                        message="Duplicate file within current batch",
                        file_name=file.filename,
                        pages_loaded=0,
                        total_pages=0,
                        file_hash=file_hash,
                        status="duplicate"
                    )
                )
                continue
            seen_hashes.add(file_hash)

            # 交给统一处理函数
            response = await _handle_single_file(file, precomputed_hash=file_hash)
            responses.append(response)

        except HTTPException as e:
            responses.append(
                UploadResponse(
                    message=f"Failed to upload {file.filename}: {e.detail}",
                    file_name=file.filename,
                    pages_loaded=0,
                    total_pages=0,
                    file_hash="unknown",
                    status="error"
                )
            )

    return responses

# ---- 把核心业务提炼成内部函数，供单/多文件复用 ----
async def _handle_single_file(
    file: UploadFile,
    precomputed_hash: str | None = None
) -> UploadResponse:
    """
    真正执行: 去重 -> 保存 -> 解析 -> 过滤 -> 向量索引
    参数 precomputed_hash: 在批量接口里已算好的 Hash 可传入；单文件上传则现场计算
    """
    # 1. 计算 Hash
    file_hash = precomputed_hash or await document_service._calculate_file_hash(file)
    await file.seek(0)  # 保证后续操作可重新读取

    # 2. 先查重
    is_duplicate = file_hash in document_service.file_hashes
    if is_duplicate:
        existing_file = document_service.file_hashes[file_hash]
        return UploadResponse(
            message="File content already exists",
            file_name=file.filename,
            pages_loaded=0,
            total_pages=0,
            file_hash=file_hash,
            status="duplicate",
            existing_file=existing_file
        )

    # 3. 保存文件
    file_path = await document_service.save_uploaded_file(file)

    # 4. 解析 + 过滤空页
    docs = document_service.load_documents([os.path.basename(file_path)])
    filtered_docs, _ = document_service.filter_documents(docs)
    if not filtered_docs:
        raise HTTPException(status_code=400, detail="No valid content found in the uploaded file")

    logger.debug(f"Type of filtered_docs: {type(filtered_docs)}")
    for i, item in enumerate(filtered_docs):
        logger.debug(f"Item {i} in filtered_docs has type: {type(item)}")
        
    # 5. 更新向量索引
    query_service.update_index(filtered_docs)

    # 6. 返回
    return UploadResponse(
        message="File uploaded and indexed successfully",
        file_name=os.path.basename(file_path),
        pages_loaded=len(filtered_docs),
        total_pages=len(docs),
        file_hash=file_hash,
        status="new"
    )

def process_and_index_task(request: UploadFromOssRequest):
    """
    一个简洁的后台任务调度器。
    它调用 DocumentOssService 来执行所有复杂的业务逻辑，
    然后用返回的结果更新全局的任务状态。
    """
    task_id = str(request.metadata.material_id) if request.metadata.material_id else "N/A"
    logger.info(f"[TASK_ID: {task_id}] Background task started. Delegating to DocumentOssService...")
    
    # 调用服务执行完整流程，并获取最终状态
    final_status = document_oss_service.process_new_oss_file(request)

    # 用服务返回的结果更新全局任务字典
    processing_task_results[task_id] = UploadResponse(
        message=final_status["message"],
        file_name=request.metadata.file_name,
        pages_loaded=final_status.get("pages_loaded", 0),
        total_pages=final_status.get("total_pages", 0), 
        file_hash=request.file_key,
        status=final_status["status"],
        # 如果是 "duplicate"，可以考虑从final_status中获取更多信息
        existing_file=request.file_key if final_status["status"] == "duplicate" else None
    )
    logger.info(f"[TASK_ID: {task_id}] Background task finished. Final status: '{final_status['status']}'")


@router.post("/upload-from-oss", response_model=UploadResponse, status_code=202)
async def upload_from_oss(request: UploadFromOssRequest, background_tasks: BackgroundTasks):
    """
    Schedules the OSS file processing and indexing as a background task.
    """
    task_id = str(request.metadata.material_id) if request.metadata.material_id else "N/A"
    logger.info(f"Received request to upload from OSS for file: '{request.metadata.file_name}'. Assigning Task ID: {task_id}.")

    background_tasks.add_task(process_and_index_task, request)
    logger.info(f"Task ID {task_id} has been successfully scheduled to run in the background.")

    return UploadResponse(
        message="Accepted: File processing and indexing has been scheduled.",
        file_name=request.metadata.file_name,
        pages_loaded=0,
        total_pages=0,
        file_hash="", 
        task_id=task_id,
        status="processing"
    )

def update_metadata_task(request: UpdateMetadataRequest):
    """
    A thin background task dispatcher for metadata updates.
    It calls DocumentOssService to perform the logic and updates the global task status.
    """
    task_id = str(request.material_id)
    logger.info(f"[TASK_ID: {task_id}] Metadata update task started. Delegating to DocumentOssService...")
    
    # Call the new service method to execute the update flow
    final_status = query_service.add_public_acl_to_material(request.material_id, request.collection_name)

    # Update the global task results dictionary with the outcome
    # We use the UpdateMetadataResponse schema for consistency
    processing_task_results[task_id] = UpdateMetadataResponse(
        message=final_status["message"],
        material_id=request.material_id,
        task_id=task_id,
        status=final_status["status"]
    )
    logger.info(f"[TASK_ID: {task_id}] Metadata update task finished. Final status: '{final_status['status']}'")

@router.post(
    "/update-metadata", 
    response_model=UpdateMetadataResponse, 
    status_code=202
)
async def update_metadata(request: UpdateMetadataRequest, background_tasks: BackgroundTasks):
    """
    Schedules the metadata update for an existing document as a background task.
    """
    task_id = str(request.material_id)
    logger.info(f"Received request to update metadata for material_id: {request.material_id}. Assigning Task ID: {task_id}.")

    # Schedule the new, thin task dispatcher
    background_tasks.add_task(update_metadata_task, request)
    
    logger.info(f"Task ID {task_id} for metadata update has been successfully scheduled.")

    # Return an immediate response indicating the task is scheduled
    return UpdateMetadataResponse(
        message="Accepted: Metadata update has been scheduled.",
        material_id=request.material_id,
        task_id=task_id,
        status="scheduled"
    )

# --- Polling Endpoint (New) ---
@router.get("/status/{task_id}", response_model=UploadResponse)
async def get_task_status(task_id: str):
    """
    Polls for the status and result of a background processing task.
    """
    logger.info(f"Received status poll request for Task ID: '{task_id}'.")
    result = processing_task_results.get(task_id)
    
    if result is None:
        logger.warning(f"Task ID '{task_id}' not found in processing results. Returning 404.")
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found. It may not have been scheduled or has expired.")
    
    logger.info(f"Found status for Task ID '{task_id}'. Current status: '{result.status}'. Returning result to client.")
    return result


@router.post("/delete-by-metadata", summary="[同步操作] 根据元数据删除文档")
def delete_by_metadata(request: DeleteByMetadataRequest):
    """
    接收来自后端的指令，根据元数据过滤器删除ChromaDB中的文档。
    这是一个同步操作，因为删除通常很快。
    """
    logger.info(f"Received request to delete documents with filters: {request.filters}")
    try:
        result = query_service.delete_nodes_by_metadata(
            collection_name=request.collection_name,
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


@router.post("/add-chat-message")
async def add_chat_message_api(request: AddChatMessageRequest):
    """
    将聊天消息同步到 ChromaDB 的聊天历史Collection。
    由 Spring 后端调用。
    """
    try:
        from services.chat_history_service import chat_history_service # 导入服务实例
        chat_history_service.add_chat_message_to_chroma(request.dict())
        return {"status": "success", "message": "Chat message added to ChromaDB."}
    except Exception as e:
        logger.error(f"Error adding chat message to ChromaDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add chat message to ChromaDB.")

