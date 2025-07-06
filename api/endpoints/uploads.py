import os
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
from services.document_service import document_service
from services.document_oss_service import document_oss_service
from services.oss_service import oss_service
from services.query_service import query_service
from models.schemas import AddChatMessageRequest, DeleteByMetadataRequest, TaskStatus, UploadResponse, UploadFromOssRequest, UpdateMetadataRequest, UpdateMetadataResponse
import shutil
from services.task_manager_service import task_manager
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

def process_task_wrapper(request: UploadFromOssRequest, task_id: str):
    """
    这个函数现在是一个纯粹的包装器/调度器。
    它的唯一职责就是调用业务服务层的方法，并把 task_id 传过去。
    所有的业务逻辑、流程控制、进度汇报和最终状态设置，都由被调用的服务自己完成。
    """
    logger.info(f"[TASK_ID: {task_id}] Background task wrapper started. Delegating all logic to DocumentOssService...")
    try:
        # 直接调用，不关心返回值。所有状态更新都在 process_new_oss_file 内部完成。
        document_oss_service.process_new_oss_file(request, task_id)
    except Exception as e:
        # 添加一个最终的保障性异常捕获。
        # 正常情况下，process_new_oss_file 内部有自己的try-except，这里不应该被触发。
        # 但为了系统健壮性，我们在这里捕获任何未被捕获的致命错误。
        logger.error(f"[TASK_ID: {task_id}] A critical unhandled exception escaped the service layer: {e}", exc_info=True)
        # 尝试将任务标记为失败
        task_manager.finish_task(task_id, "error", result={"message": "A critical and unexpected error occurred in the service layer."})

@router.post("/upload-from-oss", response_model=TaskStatus, status_code=202)
async def upload_from_oss(request: UploadFromOssRequest, background_tasks: BackgroundTasks):
        
    # 准备要存入任务初始状态的上下文数据
    initial_data = {
        "file_name": request.metadata.file_name,
        "file_key": request.file_key
    }

    # --- 使用TaskManager创建任务，并立即获取task_id ---
    initial_status = task_manager.create_task(
        task_type="document_indexing",
        initial_message=f"Task scheduled for file '{request.metadata.file_name}'.",
        initial_data=initial_data
    )
    task_id = initial_status.task_id
    logger.info(f"Received request to upload from OSS for file: '{request.metadata.file_name}'. Assigning Task ID: {task_id}.")

    background_tasks.add_task(process_task_wrapper, request, task_id)
    logger.info(f"Task {task_id} successfully scheduled.")
    return initial_status

def update_metadata_task(request: UpdateMetadataRequest):
    """
    A thin background task dispatcher for metadata updates.
    It calls DocumentOssService to perform the logic and updates the global task status.
    """
    task_id = str(request.material_id)
    material_id = request.material_id
    collection_name = request.collection_name
    logger.info(f"[TASK_ID: {task_id}] Metadata update task started. Delegating to DocumentOssService...")
    
    final_status = {"status": "error", "message": "Task failed unexpectedly."}

    try:
        # --- 步骤 A：执行通用的元数据更新 ---
        # 1. 准备用于通用更新的 payload，需要移除权限字段
        update_payload = request.metadata.model_dump(exclude_unset=True)
        update_payload.pop('accessible_to', None) # 移除权限字段，因为它将被单独处理

        # 2. 调用通用更新方法
        query_service.update_existing_nodes_metadata(
            collection_name=collection_name,
            material_id=material_id,
            metadata_update_payload=update_payload
        )
        logger.info(f"[TASK_ID: {task_id}] Step A: Generic metadata update completed.")
        
        # --- 步骤 B：检查并执行条件性发布 ---
        # 1. 检查原始请求中是否包含 "public" 权限
        if request.metadata.accessible_to and "public" in request.metadata.accessible_to:
            logger.info(f"[TASK_ID: {task_id}] Step B: Publish condition met. Triggering public ACL addition.")
            # 2. 调用发布（新增公共节点）的方法
            publish_status = query_service.add_public_acl_to_material(
                material_id=material_id,
                collection_name=collection_name
            )
            # 将发布操作的结果作为最终结果
            final_status = publish_status
        else:
            logger.info(f"[TASK_ID: {task_id}] Step B: No publish condition. Task finished after metadata update.")
            final_status = {"status": "success", "message": "Metadata updated successfully without publishing."}

    except Exception as e:
        logger.error(f"[TASK_ID: {task_id}] An error occurred in the update/publish task: {e}", exc_info=True)
        final_status = {"status": "error", "message": str(e)}

    # 更新全局任务状态字典
    processing_task_results[task_id] = UpdateMetadataResponse(
        message=final_status["message"],
        material_id=material_id,
        task_id=task_id,
        status=final_status["status"]
    )
    logger.info(f"[TASK_ID: {task_id}] Metadata update/publish task finished. Final status: '{final_status['status']}'")


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


# --- Polling Endpoint (New) ---
@router.get("/status/{task_id}", response_model=TaskStatus, summary="查询后台任务的状态")
async def get_task_status(task_id: str):
    """
    Polls for the status and result of a background processing task.
    """
    status = task_manager.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")
    return status