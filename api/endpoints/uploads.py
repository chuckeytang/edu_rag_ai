import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from services.document_service import document_service
from services.query_service import query_service
from models.schemas import UploadResponse
router = APIRouter()
@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        if await document_service.is_duplicate_file(file):
            existing_file = document_service.file_hashes.get(
                await document_service._calculate_file_hash(file)
            )
            return UploadResponse(
                message="File content already exists",
                file_name=file.filename,
                pages_loaded=0,
                total_pages=0,
                status="duplicate",
                existing_file=existing_file
            )
        file_path = await document_service.save_uploaded_file(file)
        docs = document_service.load_documents([os.path.basename(file_path)])
        filtered_docs, page_info = document_service.filter_documents(docs)
        if not filtered_docs:
            raise HTTPException(
                status_code=400,
                detail="No valid content found in the uploaded file"
            )
        if not await document_service.is_duplicate_file(file):
            query_service.update_index(filtered_docs)
        return UploadResponse(
            message="File uploaded and indexed successfully",
            file_name=os.path.basename(file_path),
            pages_loaded=len(filtered_docs),
            total_pages=len(docs),
            status="new"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded file: {str(e)}"
        )
@router.post("/upload-multiple", response_model=List[UploadResponse])
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    responses = []
    for file in files:
        try:
            response = await upload_file(file)
            responses.append(response)
        except HTTPException as e:
            responses.append(UploadResponse(
                message=f"Failed to upload {file.filename}: {e.detail}",
                file_name=file.filename,
                pages_loaded=0,
                total_pages=0
            ))
    return responses