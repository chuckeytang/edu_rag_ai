from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
from datetime import datetime, timezone
class DocumentMetadata(BaseModel):
    file_name: str
    page_label: str
class Document(BaseModel):
    text: str
    metadata: DocumentMetadata
    
class QueryResponseNode(BaseModel):
    file_name: str
    page_label: str
    content: str
    score: float
class QueryResponse(BaseModel):
    response: str
    nodes: List[QueryResponseNode]
class UploadResponse(BaseModel):
    task_id: Optional[str] = 0
    message: str
    file_name: str
    status: str  
    pages_loaded: Optional[int] = None
    total_pages: Optional[int] = None
    file_hash: Optional[str] = None # 在OSS流程中，我们用它来存 file_key
    existing_file: Optional[Dict] = None
    
class TaskStatus(BaseModel):
    """一个通用的后台任务状态模型"""
    task_id: str
    task_type: str # 例如 'document_indexing', 'report_generation'
    status: str # e.g., 'pending', 'running', 'success', 'error', 'duplicate'
    progress: int = Field(0, ge=0, le=100)
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # 一个灵活的字段，用于存放任务的最终结果
    result: Optional[Dict[str, Any]] = None 

class StreamingResponseWrapper:
    def __init__(self, async_generator):
        self.async_generator = async_generator
    async def __aiter__(self):
        async for chunk in self.async_generator:
            yield chunk
class StreamChunk(BaseModel):
    content: str
    is_last: bool = False
    metadata: Optional[Dict[str, Any]] = None 

class DebugRequest(BaseModel):
    filename: str
    question: str

class WxMineCollectSubjectList(BaseModel):
    subject: str
    clazz: Optional[str] = None # Optional 因为Java端可能为null
    exam: Optional[str] = None   # Optional 因为Java端可能为null

# 请求体模型，用于接收文本内容或 OSS file_key，并新增用户上下文信息
class ExtractionRequest(BaseModel):
    file_key: Optional[str] = None # OSS 文件键
    content: Optional[str] = None  # 直接的文本内容
    is_public: Optional[bool] = True

    # --- 用户提供的元数据预设 (现在 level 是 List[str]) ---
    user_provided_clazz: Optional[str] = Field(None, description="用户提供的课程体系预设，如：IB、IGCSE、AP")
    user_provided_subject: Optional[str] = Field(None, description="用户提供的学科预设，如：Chemistry、Business")
    user_provided_exam: Optional[str] = Field(None, description="用户提供的考试局预设，如：CAIE、Edexcel")
    # !!! 修改 1: user_provided_level 变为 List[str] !!!
    user_provided_level: List[str] = Field([], description="用户提供的等级预设，如：['AS', 'HL']") 

    # !!! 修改 2: subscribed_subjects 变为 List[WxMineCollectSubjectList] !!!
    subscribed_subjects: List[WxMineCollectSubjectList] = Field([], description="用户订阅的学科列表，包含学科、体系、考试局信息")

    class Config:
        extra = "forbid"
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
                    "is_public": True,
                    "user_provided_subject": "Mathematics",
                    "user_provided_level": ["AS"], # 示例更新
                    "subscribed_subjects": [ # 示例更新
                        {"subject": "Mathematics", "clazz": "A-Level", "exam": "CAIE"},
                        {"subject": "Physics", "clazz": "IB", "exam": "Edexcel"}
                    ]
                },
                {
                    "content": "This is a sample document about IB Chemistry, focusing on redox reactions.",
                    "user_provided_clazz": "IB",
                    "subscribed_subjects": [{"subject": "Chemistry", "clazz": "IB", "exam": ""}] # 示例更新
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
class QueryRequest(BaseModel):
    question: str
    collection_name: str = Field(...) 
    filters: Optional[Dict] = Field(default_factory=dict)
    similarity_top_k: Optional[int] = 10
    target_file_ids: Optional[List[str]] = None  # ✅ 使用 hash 列表代替文件名
    prompt: Optional[str] = None
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
    is_first_query: bool = False
    use_llm_reranker: bool = True
    use_reranker: Optional[bool] = True

class UpdateChatMessageRequest(BaseModel):
    """
    更新 ChromaDB 中聊天消息的请求 DTO。
    目前ChatHistoryService不支持更新，此DTO主要为删除后重新添加或将来扩展使用。
    """
    id: str = Field(..., description="聊天消息在ChromaDB中的唯一ID (通常是mysql_chat_id_X)")
    session_id: str
    account_id: int
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str # 时间戳字符串，例如 "2025-07-16T10:30:00"

class DeleteChatMessagesRequest(BaseModel):
    """
    删除 ChromaDB 中聊天消息的请求 DTO。
    """
    session_id: str = Field(..., description="要删除的会话ID")
    account_id: int = Field(..., description="会话所属的用户ID")