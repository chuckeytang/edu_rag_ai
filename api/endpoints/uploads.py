import os
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from services.document_service import document_service
from services.document_oss_service import document_oss_service
from services.oss_service import oss_service
from services.query_service import query_service
from models.schemas import UploadResponse, UploadFromOssRequest
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

async def process_and_index_task(request: UploadFromOssRequest):
    """
    Orchestrates the RAG pipeline using the OSS file_key for deduplication.
    1. Downloads file from OSS.
    2. Checks for duplicates using file_key.
    3. Processes file to get documents with metadata.
    4. Indexes the documents.
    5. Cleans up temporary files.
    6. Updates the task status/result.
    """
    local_file_path = None
    pages_loaded = 0
    total_pages = 0
    task_status = "error" # Default to error
    file_hash = request.file_key
    message = "An unexpected error occurred during processing."
    existing_file = None

    task_id = str(request.metadata.material_id) if request.metadata.material_id else str(uuid.uuid4())
    logger.info(f"[TASK_ID: {task_id}] Background task CREATED for file: '{request.metadata.file_name}'.")

    # Initial status update
    processing_task_results[task_id] = UploadResponse(
        message="Processing started.",
        file_name=request.metadata.file_name,
        pages_loaded=pages_loaded,
        total_pages=total_pages,
        file_hash=file_hash, 
        status="processing",
    )
    logger.info(f"[TASK_ID: {task_id}] Initial status set to 'processing'.")

    try:
        # Step 1: Download file from OSS
        logger.info(f"[TASK_ID: {task_id}] Step 1: Starting file download from OSS Key: '{request.file_key}'.")
        local_file_path = oss_service.download_file_to_temp(request.file_key)
        logger.info(f"[TASK_ID: {task_id}] File downloaded to temporary path: '{local_file_path}'.")

        # Step 2: Deduplication check using the OSS file_key
        logger.info(f"[TASK_ID: {task_id}] Step 2: Checking for duplicate using OSS key: '{request.file_key}'.")
        if request.file_key in document_oss_service.processed_files:
            existing_file_path = document_oss_service.processed_files[request.file_key]
            existing_file = existing_file_path
            message = "This exact OSS file has already been processed."
            task_status = "duplicate"
            logger.info(f"[TASK_ID: {task_id}] Duplicate file key detected. The file at this OSS path was already indexed. Task will now finalize.")
        else:
            logger.info(f"[TASK_ID: {task_id}] No duplicate key found. Proceeding to process file content.")

            # Step 3: Process the local file using the new service
            logger.info(f"[TASK_ID: {task_id}] Step 3: Processing file to load and filter document pages.")
            loaded_docs, filtered_docs = document_oss_service.process_temp_file(
                local_file_path=local_file_path,
                file_key=request.file_key,
                metadata=request.metadata.dict()
            )

            if not filtered_docs:
                message = "No valid content found in the file after filtering."
                task_status = "error"
                logger.warning(f"[TASK_ID: {task_id}] File processing resulted in zero valid pages for indexing.")
            else:
                total_pages = len(loaded_docs)
                pages_loaded = len(filtered_docs)
                logger.info(f"[TASK_ID: {task_id}] File processing complete. Total pages found: {total_pages}. Valid pages for indexing: {pages_loaded}.")

                # Step 4: Index the documents
                collection_to_index = request.collection_name or "public_collection"
                logger.info(f"[TASK_ID: {task_id}] Step 4: Starting to index {pages_loaded} document chunks into collection '{collection_to_index}'.")
                query_service.update_index(filtered_docs, collection_name=collection_to_index)
                logger.info(f"[TASK_ID: {task_id}] Indexing complete.")

                message = "File from OSS has been processed and indexed successfully."
                task_status = "success"

    except Exception as e:
        message = f"An unexpected error occurred: {str(e)}"
        task_status = "error"
        logger.error(f"[TASK_ID: {task_id}] Unhandled exception for file '{request.metadata.file_name}': {e}", exc_info=True)
    finally:
        # Step 5: Cleanup
        logger.info(f"[TASK_ID: {task_id}] Step 5: Entering finalization and cleanup block.")
        if local_file_path:
            temp_dir = os.path.dirname(local_file_path)
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"[TASK_ID: {task_id}] Cleaned up temporary directory: '{temp_dir}'.")
                except Exception as e:
                    logger.error(f"[TASK_ID: {task_id}] Failed to clean up temporary directory '{temp_dir}': {e}", exc_info=True)

        # Final status update
        final_result = UploadResponse(
            message=message,
            file_name=request.metadata.file_name,
            pages_loaded=pages_loaded,
            total_pages=total_pages,
            file_hash=file_hash, 
            status=task_status,
            existing_file=existing_file
        )
        processing_task_results[task_id] = final_result
        logger.info(f"[TASK_ID: {task_id}] Background task finished with status: '{task_status}'. Final message: '{message}'.")

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
        status="processing"
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