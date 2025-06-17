from pydantic import BaseModel
from typing import Optional, List
class DocumentMetadata(BaseModel):
    file_name: str
    page_label: str
class Document(BaseModel):
    text: str
    metadata: DocumentMetadata
class QueryRequest(BaseModel):
    question: str
    similarity_top_k: Optional[int] = 10
    target_file_ids: Optional[List[str]] = None  # ✅ 使用 hash 列表代替文件名
    prompt: Optional[str] = None
class QueryResponseNode(BaseModel):
    file_name: str
    page_label: str
    content: str
    score: float
class QueryResponse(BaseModel):
    response: str
    nodes: List[QueryResponseNode]
class UploadResponse(BaseModel):
    message: str
    file_name: str
    pages_loaded: int
    total_pages: int
    status: str  
    file_hash: str 
    existing_file: Optional[str] = None
class StreamingResponseWrapper:
    def __init__(self, async_generator):
        self.async_generator = async_generator
    async def __aiter__(self):
        async for chunk in self.async_generator:
            yield chunk
class StreamChunk(BaseModel):
    content: str
    is_last: bool = False
class DebugRequest(BaseModel):
    filename: str
    question: str