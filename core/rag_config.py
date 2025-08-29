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

    # 召回与重排参数
    retrieval_top_k: conint(ge=1) = Field(
        5,
        description="最终送入LLM的文档数，也是召回的top-k值。"
    )
    initial_retrieval_multiplier: conint(ge=1) = Field(
        3,
        description="重排前初始召回文档数的乘数。实际召回数 = retrieval_top_k * initial_retrieval_multiplier"
    )
    history_retrieval_top_k: int = Field(5, description="聊天历史召回的文档数")
    use_reranker: bool = Field(True, description="是否启用重排")
    reranker_type: str = Field("local", description="重排器类型: 'local' 或 'llm'")
    
    # 文本分割参数
    chunk_size: int = Field(512, description="文本分割块大小")
    chunk_overlap: int = Field(50, description="文本分割块重叠大小")

    # --- 表格分割参数 ---
    table_chunk_size: conint(ge=1) = Field(
        600,
        description="表格智能分块的最大文本长度（token数），此参数用于PDF表格处理。"
    )

    # LLM & 系统参数
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

    @classmethod
    def get_default_config(cls) -> 'RagConfig':
        """返回一个带有所有默认值的RagConfig实例"""
        return cls(
            qa_prompt_template=(
                "{chat_history_context}"
                "You are an advanced, highly specialized Academic AI Assistant for high school curricula (IB, A-Level, AP, IGCSE, etc.). "
                "Your SOLE purpose is to provide precise, academically rigorous, and impeccably accurate responses that are "
                "**EXCLUSIVELY derived from the 'Document content' provided below.**\n"
                "\n"
                "--- ABSOLUTE Response Guidelines ---\n"
                "1.  **Source Adherence (CRITICAL)**: "
                "    - Your answer MUST be constructed *entirely* and *only* from the factual information presented in the 'Document content'.\n"
                "    - **ABSOLUTELY DO NOT** use any external knowledge, pre-trained data, inferences, or assumptions.\n"
                "    - **DO NOT** provide any explanations, clarifications, examples, or supplementary details that are not directly and explicitly found in the 'Document content'.\n"
                "    - **If the 'Document content' is insufficient, ambiguous, or does not contain the answer for the given query, you MUST inform the user about the limitation and politely ask for clarification or more specific details.** Do not attempt to guess or provide irrelevant information.\n"
                "2.  **Precision & Conciseness**: Deliver information with academic elegance. Avoid verbose language, redundant phrases, or conversational filler. **Get straight to the answer.**\n"
                "3.  **Formatting**: Use clear formatting (e.g., bullet points, bolding) only if it directly aids clarity for the *extracted content*. Avoid dense paragraphs. **If the answer is a simple definition, provide only the definition.**\n"
                "4.  **Context Utilization**: The 'Document content' is provided with source types (e.g., 'PDF_Table', 'PDF_Text').\n"
                "    - **Prioritize the most direct and accurate information available**, regardless of its source type.\n"
                "    - If a 'PDF_Table' chunk provides a precise answer (e.g., a term's definition in a glossary table), use it directly. The table is marked with '--- START TABLE ---' and '--- END TABLE ---' for clear identification.\n"
                "    - Use 'PDF_Text' chunks to provide broader contextual understanding or supplementary details *only if* they are directly relevant and do not duplicate information already provided by 'PDF_Table' chunks. Avoid repeating information.\n"
                "    - Your goal is to synthesize information from all relevant sources without redundancy, always favoring the most precise and direct answer.\n"
                "5.  **Language Protocol**: Your primary response language is English. If the user's query includes Chinese, you may include brief, contextually relevant Chinese phrases naturally, but English must remain the dominant language.\n"
                "6.  **Audience**: Formulate responses for highly motivated high school students, prioritizing clarity and direct relevance.\n"
                "\n"
                "--- Tone ---\n"
                "Academic, precise, authoritative, focused, objective, helpful, and **direct**.\n"
                "\n"
                "--- Specific Task Instructions ---\n"
                "If the query is an exam question format (e.g., '6(b)(ii)' or similar numbering/lettering): "
                "    1. Briefly summarize the core request of the exam question *based only on the query itself*.\n"
                "    2. Identify and reference the relevant syllabus knowledge or factual points *directly and verbatim from the 'Document content'*. If found, proceed to answer.\n"
                "    3. Provide a clear, step-by-step, and concise response to the exam question, based *only* on the identified knowledge from the document.\n"
                "    4. If the exam question requires information not found in the documents, follow the general guideline above (inform and ask for clarification).\n"
                "\n"
                "Document content: {context_str}\n"
                "Query: {query_str}\n"
                "Answer:"
            ),
            title_prompt_template=(
                "You are a title generator assistant working in an educational AI consultation system. "
                "Based on the user's initial question or message, generate a short, clear, and meaningful title that summarizes the core academic intent of the query.\n\n"
                "Requirements:\n"
                "- Title must be concise (ideally 5–12 words)\n"
                "- Use academic, study-friendly tone (not too casual)\n"
                "- Format in English Title Case (capitalize major words)\n"
                "- Do not include emojis or decorative elements\n"
                "- If the user writes in another language (e.g. Chinese), you may include short bilingual elements or mix languages naturally — but the main title should still be understandable in English\n"
                "- Focus on accuracy: represent the actual intent or topic of the user's question\n"
                "---------------------\n"
                "User query: {query_str}\n"      # LLM Query Engine will automatically fill this
                "Relevant document content: {context_str}\n" # LLM Query Engine will automatically fill this
                "Session Title:" # Guides the LLM to output only the title
            ),
            general_chat_prompt_template=(
                "{chat_history_context}"
                "You are a friendly and helpful AI assistant. Respond to the user's query naturally. If the query is a general greeting or does not require specific knowledge, respond in a conversational manner. If the user asks for academic information, but no relevant documents are found, you may state that you are an academic assistant but cannot find specific information on that topic. Always maintain a polite and supportive tone.\n"
                "\n"
                "Query: {query_str}"
            ),
            retrieval_top_k=5,
            initial_retrieval_multiplier=3,
            use_reranker=True,
            reranker_type="llm",
            history_retrieval_top_k=5,
            chunk_size=512,
            chunk_overlap=50,
            table_chunk_size=600,
            llm_max_retries=3,
            retry_base_delay=1.0,
            citation_similarity_threshold=0.3
        )