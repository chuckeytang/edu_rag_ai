import os
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from services.document_service import document_service
from services.query_service import query_service
from models.schemas import UploadResponse, UploadFromOssRequest
router = APIRouter()

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

def process_and_index_task(request: UploadFromOssRequest):
    """
    This function orchestrates the entire RAG pipeline and is executed in the background.
    1. Downloads file from OSS.
    2. Processes file to get documents with metadata.
    3. Indexes the documents.
    4. Cleans up temporary files.
    """
    local_file_path = None
    try:
        # Step 1: Download file from OSS to a temporary local path
        logger.info(f"Background task started for: {request.metadata.file_name}")
        local_file_path = oss_service.download_file(request.file_key)
        
        # Step 2: Process the local file to get document chunks with metadata
        documents_to_index = document_service.process_oss_file(
            local_file_path=local_file_path,
            metadata=request.metadata
        )
        
        # Step 3: If there are documents, pass them to the query service for indexing
        if documents_to_index:
            query_service.update_index(documents_to_index)
            logger.info(f"Indexing task for {request.metadata.file_name} completed successfully.")
        else:
            logger.info(f"No documents to index for {request.metadata.file_name} (file may be a duplicate or empty).")

    except Exception as e:
        logger.error(f"Error in background task for file {request.metadata.file_name}: {e}", exc_info=True)
    finally:
        # Step 4: CRITICAL - Always clean up the temporary download directory
        if local_file_path:
            temp_dir = os.path.dirname(local_file_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")

@router.post("/upload-from-oss", response_model=UploadResponse, status_code=202)
async def upload_from_oss(request: UploadFromOssRequest, background_tasks: BackgroundTasks):
    """
    Main RAG pipeline endpoint.
    Schedules the file download, processing, and indexing as a background task.
    """
    # Add the single orchestrator function to the background tasks
    background_tasks.add_task(process_and_index_task, request)
    
    logger.info(f"Accepted request to process file: {request.metadata.file_name}")
    return UploadResponse(
        message="Accepted: File processing and indexing has been scheduled.",
        file_name=request.metadata.file_name,
        status="processing"
    )