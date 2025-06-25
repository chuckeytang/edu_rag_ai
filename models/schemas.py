from pydantic import BaseModel, Field
from typing import Optional, List, Dict
class DocumentMetadata(BaseModel):
    file_name: str
    page_label: str
class Document(BaseModel):
    text: str
    metadata: DocumentMetadata
class QueryRequest(BaseModel):
    question: str
    collection_name: str = Field(...) 
    filters: Optional[Dict] = Field(default_factory=dict)
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
    task_id: Optional[str] = 0
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
    material_id: int = Field(...)
    author_id: Optional[int] = Field(None)
    file_key: str = Field(..., description="The unique OSS key, e.g., 'material-import/...'")
    file_name: str = Field(..., description="The original display name of the file.")
    file_size: str = Field(..., description="The file size with unit, e.g., '15.39MB'")
    title: str
    author: Optional[str] = None
    is_vip: Optional[bool] = Field(None)
    gen_year: Optional[str] = Field(None)
    clazz: Optional[str] = None
    exam: Optional[str] = None
    subject: Optional[str] = None
    type: str
    accessible_to: Optional[List[str]] = Field(None)
    level_list: Optional[List[str]] = Field(None)
    class Config:
        # 允许Pydantic从额外的字段创建模型，但不会包含在模型中
        extra = 'ignore' 
    
# The request from Java will contain the OSS key and the full metadata
class UploadFromOssRequest(BaseModel):
    file_key: str = Field(..., description="The object key for the file in the public OSS bucket.")
    metadata: RAGMetadata
    collection_name: Optional[str] = Field(None)

class UpdateMetadataRequest(BaseModel):
    material_id: int = Field(..., description="The ID of the material to update.")
    collection_name: Optional[str] = Field(None)
    metadata: RAGMetadata = Field(..., description="The new metadata payload.")

class UpdateMetadataResponse(BaseModel):
    message: str
    material_id: int
    task_id: str
    status: str
    
class DocumentChunkResponse(BaseModel):
    page_label: str
    text_snippet: str
    metadata: RAGMetadata
