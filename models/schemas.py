from pydantic import BaseModel, Field
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

class RAGMetadata(BaseModel):
    clazz: Optional[str] = None
    exam: Optional[str] = None
    subject: Optional[str] = None
    level_list: Optional[List[str]] = Field(default_factory=list, alias="levelList")
    type: str
    title: str
    author_id: Optional[int] = Field(None, alias="authorId")
    author: Optional[str] = None
    file_name: str = Field(..., alias="fileName")
    file_size: str = Field(..., alias="fileSize")
    gen_year: str = Field(..., alias="genYear")
    is_vip: bool = Field(..., alias="isVip")
    material_id: int = Field(..., alias="materialId")
    
# The request from Java will contain the OSS key and the full metadata
class UploadFromOssRequest(BaseModel):
    file_key: str = Field(..., description="The object key for the file in the public OSS bucket.", alias="fileKey")
    metadata: RAGMetadata