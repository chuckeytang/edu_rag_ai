# models/rag_config.py 

from pydantic import BaseModel, Field, confloat, conint
from typing import Optional

class RagConfig(BaseModel):
    """
    RAG 核心参数配置模型。
    """
    # 提示词模板
    qa_prompt_template: str = Field(
        ...,
        description="主问答提示词模板。{chat_history_context}, {context_str}, {query_str} 是占位符。"
    )
    title_prompt_template: str = Field(
        ...,
        description="标题生成提示词模板。{query_str}, {context_str} 是占位符。"
    )
    general_chat_prompt_template: str = Field(
        ...,
        description="通用聊天提示词模板。{chat_history_context}, {query_str} 是占位符。"
    )
    knowledge_base_id: str = Field(
        ...,
        description="火山引擎知识库的唯一ID。"
    )

    # 召回参数
    retrieval_top_k: conint(ge=1) = Field(
        ...,
        description="最终送入LLM的文档数，也是火山引擎召回的limit值。"
    )
    history_retrieval_top_k: int = Field(..., description="聊天历史召回的文档数")
    use_reranker: bool = Field(True, description="是否启用重排")

    # 检索分数阈值
    retrieval_score_threshold: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="用于判断检索结果是否有效的最高重排分数阈值。低于此阈值将回退到通用问答。"
    )
    
    # LLM & 系统参数
    llm_max_context_tokens: conint(ge=1) = Field(
        4096,
        description="LLM模型的最大上下文令牌数。用于限制传入LLM的总文本量，以避免超出模型限制。"
    )
    llm_max_retries: conint(ge=0) = Field(
        3,
        description="LLM请求的最大重试次数。设为0则不重试。"
    )

    retry_base_delay: confloat(ge=0.0) = Field(
        1.0,
        description="LLM请求重试的基础等待时间（秒）。"
    )
    
    # 引用相似度阈值
    citation_similarity_threshold: confloat(ge=0.0, le=1.0) = Field(
        0.3,
        description="用于判断LLM生成句子是否引用了原始文档的相似度阈值。取值范围0.0-1.0。"
    )
