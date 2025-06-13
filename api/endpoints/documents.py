from fastapi import APIRouter, HTTPException
from typing import List
from services.document_service import document_service
from models.schemas import Document, DocumentMetadata, DebugRequest
router = APIRouter()
@router.get("/documents", response_model=List[Document])
async def get_documents():
    try:
        docs = document_service.load_documents([
            "exam_paper.pdf",
            "0610_syllabus_2024.pdf"
        ])
        filtered_docs, _ = document_service.filter_documents(docs)
        return [
            Document(
                text=doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                metadata=DocumentMetadata(
                    file_name=doc.metadata["file_name"],
                    page_label=doc.metadata["page_label"]
                )
            )
            for doc in filtered_docs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/debug-question")
async def debug_question_search(request: DebugRequest):
    return document_service.debug_question_search(
        request.filename,
        request.question
    )