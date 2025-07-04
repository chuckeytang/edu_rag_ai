from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
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

# 请求体模型，用于接收文本内容或 OSS file_key
class ExtractionRequest(BaseModel):
    file_key: Optional[str] = None # OSS 文件键
    content: Optional[str] = None  # 直接的文本内容
    is_public: Optional[bool] = True

    # 确保至少提供一个
    class Config:
        extra = "forbid"
        # 验证器：确保 file_key 或 content 至少有一个被提供
        @classmethod
        def validate_at_least_one(cls, values):
            if not values.get('file_key') and not values.get('content'):
                raise ValueError("必须提供 'file_key' 或 'content' 中的至少一个。")
            if values.get('file_key') and values.get('content'):
                raise ValueError("不能同时提供 'file_key' 和 'content'。")
            return values
        
        json_schema_extra = {
            "examples": [
                {
                    "file_key": "path/to/your/document.pdf",
                    "is_public": True
                },
                {
                    "file_key": "path/to/your/private_document.docx",
                    "is_public": False
                },
                {
                    "content": "This is a sample document content about Chemistry."
                }
            ]
        }
        
# 用于文档元数据提取
class ExtractedDocumentMetadata(BaseModel): # 新的、更具体的元数据模型名称
    clazz: Optional[str] = Field(None, description="课程体系名称，如：IB、IGCSE、AP...，匹配课程体系选项集合中的一个")
    exam: Optional[str] = Field(None, description="考试局名称，如：CAIE、Edexcel、AQA，匹配考试局选项集合中的一个")
    labelList: List[str] = Field([], description="标签名称列表，该字段是从一个label集合中，匹配对应的label，如：[\"Znotes\",\"Definitions\",\"Paper 5\",\"Book\"...]，当然如果没有匹配也可以填空")
    levelList: List[str] = Field([], description="等级名称，如：[\"AS\",\"A2\",...]，匹配等级集合中的0-多个")
    subject: Optional[str] = Field(None, description="学科，如：Chemistry、Business、Computer Science，匹配学科选项集合中的一个")
    type: Optional[str] = Field(None, description="资料类型，如：Handout、Paper1、IA...，匹配资料类型选项集合中的一个")
    description: Optional[str] = Field(None, max_length=1024, description="文档摘要，对文档内容的简洁描述，最大长度1024字符。") 

# 用于知识点（记忆卡）提炼
class Flashcard(BaseModel):
    term: str = Field(..., description="知识点或术语，例如：光合作用, 二氧化碳固定")
    explanation: str = Field(..., description="对术语的简洁解释，一小段话。")

class FlashcardList(BaseModel): # 用于解析LLM返回的列表，内部使用
    flashcards: List[Flashcard] = Field(..., description="提取到的记忆卡列表。")

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

class DeleteByMetadataRequest(BaseModel):
    collection_name: str = Field(..., description="The name of the collection to delete from.")
    filters: Optional[Dict] = Field(..., description="The `where` clause to filter documents for deletion.")

class DocumentChunkResponse(BaseModel):
    page_label: str
    text_snippet: str
    metadata: RAGMetadata

class Flashcard(BaseModel):
    term: str = Field(..., description="知识点或术语，例如：光合作用, 二氧化碳固定")
    explanation: str = Field(..., description="对术语的简洁解释，一小段话。")

class FlashcardList(BaseModel): # 用于解析JSON数组
    flashcards: List[Flashcard] = Field(..., description="提取到的记忆卡列表。")


class AddChatMessageRequest(BaseModel):
    id: str # Unique ID for ChromaDB, e.g., "mysql_chat_id_123"
    session_id: str
    account_id: int
    role: str # "user" or "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = {} # 这里使用 Dict[str, Any] 来兼容 JsonNode 传递过来的数据
    timestamp: str # ISO formatted string

class ChatQueryRequest(BaseModel):
    question: str
    session_id: str
    account_id: int
    context_retrieval_query: str
    collection_name: str
    target_file_ids: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    similarity_top_k: Optional[int] = 5
    prompt: Optional[str] = None