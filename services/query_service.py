import json
import logging
import os
import shutil
import tiktoken
import asyncio

from llama_cloud import TextNode

from core.metadata_utils import prepare_metadata_for_storage
from typing import TYPE_CHECKING, Optional

from services.indexer_service import IndexerService
if TYPE_CHECKING:
    from services.chat_history_service import ChatHistoryService

logger = logging.getLogger(__name__)

from typing import Any, AsyncGenerator, List, Union
from llama_index.core.schema import Document as LlamaDocument, NodeWithScore, TextNode as LlamaTextNode, QueryBundle

from llama_index.core import VectorStoreIndex, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine # Import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank 
from llama_index.core.postprocessor import LLMRerank
import torch 

from core.config import Settings, settings
from models.schemas import ChatQueryRequest, StreamChunk
from typing import List, Dict
import chromadb

PERSIST_DIR = settings.INDEX_PATH

try:
    _tokenizer = tiktoken.get_encoding("cl100k_base") 
    logger.info("Using tiktoken 'cl100k_base' for token counting estimation.")
except Exception as e:
    logger.warning(f"Could not load tiktoken tokenizer 'cl100k_base', token counting may be inaccurate: {e}")
    _tokenizer = None

class CustomRetriever(BaseRetriever):
    """
    一个自定义的 Retriever，它不执行实际的向量搜索，
    而是直接返回初始化时传入的节点列表。
    用于在 RAG 召回为 0 时，将虚拟节点传递给 QueryEngine。
    """
    def __init__(self, nodes: List[NodeWithScore]):
        self._nodes = nodes
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        logger.debug(f"CustomRetriever: _aretrieve called for query '{query_bundle.query_str[:50]}...'. Returning {len(self._nodes)} pre-set nodes.")
        return self._nodes
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        logger.debug(f"CustomRetriever: _retrieve called for query '{query_bundle.query_str[:50]}...'. Returning {len(self._nodes)} pre-set nodes.")
        return self._nodes
    
class QueryService:
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embedding_model: BaseEmbedding,
                 llm: LLM, 
                 indexer_service: IndexerService,
                 chat_history_service: Optional['ChatHistoryService'] = None,
                 deepseek_llm_for_reranker: Optional[LLM] = None):
        """
        构造函数 - 实现“按需加载”策略
        启动时只初始化客户端和空的索引缓存，不加载任何实际的索引数据。
        """
        self.chroma_client = chroma_client       # 使用传入的客户端
        self.embedding_model = embedding_model   # 使用传入的 embedding model
        self.llm = llm                           # 使用传入的 LLM
        
        self.indices: Dict[str, VectorStoreIndex] = {}
        self._indexer_service = indexer_service
        self.qa_prompt = self._create_qa_prompt()

        # 保存 chat_history_service 实例
        self._chat_history_service = chat_history_service

        self.reranker_top_n_fixed = 5
        # 1. 本地模型 SentenceTransformer Reranker
        self.local_reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base", 
            top_n=self.reranker_top_n_fixed, 
            device="cuda" if torch.cuda.is_available() else "cpu" 
        )
        logger.info(f"Initialized Local Reranker: SentenceTransformerRerank with model '{self.local_reranker.model}' on device '{self.local_reranker.device}' and fixed top_n={self.local_reranker.top_n}.")

        # 2. 基于 LLM 的 Reranker (使用 DeepSeek)
        self.llm_reranker = None
        if deepseek_llm_for_reranker:
            self.llm_reranker = LLMRerank(
                llm=deepseek_llm_for_reranker, # 传入 DeepSeek LLM 实例
                top_n=self.reranker_top_n_fixed # 与本地重排器返回相同数量的节点
            )
            logger.info(f"Initialized LLM Reranker: LLMRerank with model '{deepseek_llm_for_reranker.model}' and fixed top_n={self.llm_reranker.top_n}.")
        else:
            logger.warning("DeepSeek LLM for Reranker not provided. LLM Reranker will not be initialized.")
        
    def _get_or_load_index(self, collection_name: str) -> VectorStoreIndex:
        """
        重定向到 IndexerService 的按需加载方法。
        """
        return self._indexer_service._get_or_load_index(collection_name)

    def retrieve_with_filters(self, question: str, collection_name: str, filters: dict, similarity_top_k: int = 5):
        """
        [DEBUG METHOD] 仅执行带过滤的召回，并返回召回的节点列表。
        """
        logger.info(f"[DEBUG] Retrieving from collection '{collection_name}' with filters: {filters}")
        
        index = self._get_or_load_index(collection_name)
        if not index:
            raise ValueError(f"Collection '{collection_name}' does not exist or could not be loaded.")

        chroma_where_clause = self._build_chroma_where_clause(filters)

        # 1. 创建一个Retriever，配置和查询时完全一样
        retriever = index.as_retriever(
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
            similarity_top_k=similarity_top_k * 3
        )
        
        retrieved_nodes = retriever.retrieve(question)
        logger.info(f"[DEBUG] Retriever found {len(retrieved_nodes)} nodes matching the criteria.")

        query_bundle_for_rerank = QueryBundle(query_str=question)
        final_retrieved_nodes = self.reranker.postprocess_nodes(
            retrieved_nodes, 
            query_bundle=query_bundle_for_rerank
        )
        logger.info(f"[DEBUG] Reranker returned {len(final_retrieved_nodes)} nodes after reranking.")

        return final_retrieved_nodes

    def _build_chroma_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        为 ChromaDB 构建原生 where 子句。
        此版本根据值的类型，自动使用 $in 或 $eq。
        它能处理复杂过滤器（如 {"field": {"$in": [val1, val2]}}）或简单过滤器（如 {"field": val}）。
        """
        if not filters:
            return {}

        chroma_filters = []

        for key, value in filters.items():
            # 情况 1: 值本身是一个字典，并且包含 ChromaDB 支持的操作符 (如 "$in")
            # 这表示调用方已经提供了复杂的过滤条件，直接使用即可。
            if isinstance(value, dict) and any(op in value for op in ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"]):
                chroma_filters.append({key: value})
            # 情况 2: 值是一个列表，我们应使用 "$in" 操作符
            elif isinstance(value, list):
                # 检查列表是否为空，避免生成 {"key": {"$in": []}} 这种可能导致问题的情况
                if value:
                    chroma_filters.append({key: {"$in": value}})
                else:
                    # 如果列表为空，则该条件不会匹配任何文档，可以忽略或根据业务逻辑处理
                    logger.warning(f"Empty list provided for filter key '{key}'. This condition will be ignored.")
            # 情况 3: 其他所有简单类型的值 (str, int, float)，使用默认的相等比较
            # ChromaDB 对于 {"key": value} 形式，默认就是相等比较，不需要显式写 "$eq"
            else:
                chroma_filters.append({key: {"$eq": value}})

        if not chroma_filters:
            return {}
        
        # 如果有多个过滤条件，用 $and 连接
        if len(chroma_filters) > 1:
            return {"$and": chroma_filters}
        
        # 只有一个条件，直接返回
        return chroma_filters[0]
            
    def _create_qa_prompt(self, prompt: str = None) -> PromptTemplate:
        if prompt:
            return PromptTemplate(prompt)
        else:
            return self.default_qa_prompt

    @property
    def title_generation_prompt(self) -> PromptTemplate:
        """
        用于从用户查询和相关文档中生成简短会话标题的提示词。
        """
        return PromptTemplate(
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
        )
    
    @property
    def default_qa_prompt(self) -> PromptTemplate:
        return PromptTemplate(
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
            "    - **If the 'Document content' is insufficient, ambiguous, or does not contain the answer for the given query, you MUST inform the user about the limitation and politely ask for clarification or more specific details.** Do not attempt to guess or provide irrelevant information.\n" # <--- 关键修改：改为询问，而不是拒绝
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
            "Academic, precise, authoritative, focused, objective, helpful, and **direct**.\n" # <--- 增加 helpful
            "\n"
            "--- Specific Task Instructions ---\n"
            "If the query is an exam question format (e.g., '6(b)(ii)' or similar numbering/lettering): "
            "    1. Briefly summarize the core request of the exam question *based only on the query itself*.\n"
            "    2. Identify and reference the relevant syllabus knowledge or factual points *directly and verbatim from the 'Document content'*. If found, proceed to answer.\n" # <--- 强调如果找到就回答
            "    3. Provide a clear, step-by-step, and concise response to the exam question, based *only* on the identified knowledge from the document.\n"
            "    4. If the exam question requires information not found in the documents, follow the general guideline above (inform and ask for clarification).\n" # <--- 考试题也遵循总原则
            "\n"
            "Document content: {context_str}\n"
            "Query: {query_str}\n"
            "Answer:"
        )
            
    @property
    def general_chat_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            "{chat_history_context}"
            "You are a friendly and helpful AI assistant. Respond to the user's query naturally. If the query is a general greeting or does not require specific knowledge, respond in a conversational manner. If the user asks for academic information, but no relevant documents are found, you may state that you are an academic assistant but cannot find specific information on that topic. Always maintain a polite and supportive tone.\n"
            "\n"
            "Query: {query_str}"
        )
    
    async def rag_query_with_context(
        self,
        request: ChatQueryRequest
    ) -> AsyncGenerator[bytes, None]:
        
        logger.info(f"Starting RAG query for session {request.session_id}, user {request.account_id} with query: '{request.question}'")
        
        # --- 语义检索历史聊天上下文 ---
        chat_history_context_string = ""
        try:
            if self._chat_history_service: 
                chat_history_context_nodes = self._chat_history_service.retrieve_chat_history_context( 
                    session_id=request.session_id,
                    account_id=request.account_id,
                    query_text=request.context_retrieval_query,
                    top_k=request.similarity_top_k or 5
                )
                if chat_history_context_nodes:
                    chat_history_context_string = "The following are excerpts from a historical conversation relevant to your current problem:\n" + \
                                                  "\n".join([f"[{node.metadata.get('role', '未知')}]: {node.text}" for node in chat_history_context_nodes]) + \
                                                  "\n---\n"
                logger.info(f"Chat history context: \n{chat_history_context_string}")
            else:
                logger.warning("ChatHistoryService instance not available in QueryService. Skipping chat history retrieval.")

        except Exception as e:
            logger.error(f"Failed to retrieve chat history context: {e}", exc_info=True)
            chat_history_context_string = ""

        rag_index = self._get_or_load_index(request.collection_name)
        if not rag_index:
            logger.error(f"RAG collection '{request.collection_name}' does not exist or could not be loaded.")
            error_chunk_json = StreamChunk(content="抱歉，知识库未准备好，请稍后再试。", is_last=True).json()
            yield f"data: {error_chunk_json}\n\n".encode("utf-8") 
            return

        combined_rag_filters = request.filters if request.filters else {}
        # Initialize the list to store the material_ids that were actually requested by the user
        final_referenced_material_ids = [] 

        if request.target_file_ids and len(request.target_file_ids) > 0:
            try:
                material_ids_int = [int(mid) for mid in request.target_file_ids]
                # Store the original string IDs for the metadata
                final_referenced_material_ids = request.target_file_ids 
                
                if "material_id" in combined_rag_filters and isinstance(combined_rag_filters["material_id"], dict) and "$in" in combined_rag_filters["material_id"]:
                    current_material_ids = combined_rag_filters["material_id"]["$in"]
                    # 确保不重复添加
                    combined_rag_filters["material_id"]["$in"] = list(set(current_material_ids + material_ids_int))
                else:
                    combined_rag_filters["material_id"] = {"$in": material_ids_int}
            except ValueError:
                logger.warning("Invalid material_id in target_file_ids. Ignoring file filter.")
        
        chroma_where_clause = self._build_chroma_where_clause(combined_rag_filters)
        logger.info(f"Main RAG ChromaDB `where` clause: {chroma_where_clause}")


        generated_title = ""
        # 直接使用 Java 侧传递的 is_first_query 字段来判断是否生成标题
        if request.is_first_query: 
            logger.info("Attempting to generate session title as requested by Java side (is_first_query is True).")
            try:
                title_query_engine = rag_index.as_query_engine(
                    llm=self.llm,
                    vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
                    similarity_top_k=2,
                    text_qa_template=self.title_generation_prompt
                )
                title_response = await title_query_engine.aquery(request.question)
                generated_title = title_response.response.strip()
                logger.info(f"Generated session title: '{generated_title}'")
                logger.debug(f"DEBUG: Raw title response from LLM: '{title_response.response}'")
            except Exception as e:
                logger.error(f"Failed to generate session title: {e}", exc_info=True)
                generated_title = ""
        else:
            logger.info("Skipping session title generation as not requested by Java side (is_first_query is False).") 

        # --- 主 RAG 查询 ---
        max_retries = 3
        
        # --- 召回文档 ---
        retriever = rag_index.as_retriever(
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
            similarity_top_k=(request.similarity_top_k or 5) * 3 # 初始召回更多节点，例如3倍，以便 Reranker 有更多选择
        )
        logger.info(f"RAG Retriever query clause: {chroma_where_clause}") 

        retrieved_nodes = await retriever.aretrieve(request.question)
        original_retrieved_nodes_count = len(retrieved_nodes) 
        logger.info(f"Retrieved {original_retrieved_nodes_count} chunks for query: '{request.question[:50]}...'")

        current_retrieved_nodes = list(retrieved_nodes) 
        
        if original_retrieved_nodes_count == 0:
            logger.info("No documents retrieved. Adding a virtual 'NO_RELEVANT_DOCUMENTS_FOUND' node for LLM context.")
            virtual_node_content = "NO_RELEVANT_DOCUMENTS_FOUND"
            virtual_node = LlamaTextNode(text=virtual_node_content, metadata={"source": "virtual_fallback", "type": "no_document_found"})
            virtual_node_with_score = NodeWithScore(node=virtual_node, score=0.0)
            current_retrieved_nodes.append(virtual_node_with_score) 
            
        final_retrieved_nodes = []
        # 确保 reranker_top_n 是最终需要返回的数量
        effective_reranker_top_n = request.similarity_top_k or 5
        query_bundle_for_rerank = QueryBundle(query_str=request.question)

        if request.use_reranker:
            if request.use_llm_reranker and self.llm_reranker:
                logger.info(f"Applying LLM Reranker to {len(current_retrieved_nodes)} nodes (top_n={effective_reranker_top_n})...")
                # LLMRerank 的 aretrieve 方法接收 QueryBundle 和 NodeWithScore 列表
                # LLMRerank 内部会自动处理 top_n 过滤
                self.llm_reranker.top_n = effective_reranker_top_n # 动态设置 LLM Reranker 的 top_n
                final_retrieved_nodes = self.llm_reranker._postprocess_nodes( # <--- 修正：调用 _postprocess_nodes
                    nodes=current_retrieved_nodes, 
                    query_bundle=query_bundle_for_rerank
                )
                logger.info(f"LLM Reranker returned {len(final_retrieved_nodes)} nodes.")
            elif self.local_reranker:
                logger.info(f"Applying Local Reranker (SentenceTransformerRerank) to {len(current_retrieved_nodes)} nodes (top_n={effective_reranker_top_n})...")
                # SentenceTransformerRerank 的 postprocess_nodes 方法接收 QueryBundle 和 NodeWithScore 列表
                # SentenceTransformerRerank 内部会自动处理 top_n 过滤
                self.local_reranker.top_n = effective_reranker_top_n # 动态设置本地 Reranker 的 top_n
                final_retrieved_nodes = self.local_reranker.postprocess_nodes(
                    current_retrieved_nodes, 
                    query_bundle=query_bundle_for_rerank
                )
                logger.info(f"Local Reranker returned {len(final_retrieved_nodes)} nodes.")
            else:
                logger.warning("Reranker is enabled but no Reranker (LLM or Local) is initialized. Skipping reranking.")
                final_retrieved_nodes = current_retrieved_nodes[:effective_reranker_top_n] # 截断到 top_n
        else:
            logger.info("Reranker is disabled by request. Skipping reranking.")
            final_retrieved_nodes = current_retrieved_nodes[:effective_reranker_top_n] # 确保即使不 Rerank 也只取 top_n

         # --- 构建发送给 LLM 的 context_str，现在包含类型信息 ---
        context_parts = []
        for n in final_retrieved_nodes: # 使用重排后的节点列表
            doc_type = n.node.metadata.get("document_type", "Unknown")
            page_label = n.node.metadata.get("page_label", "N/A")
            file_name = n.node.metadata.get("file_name", "N/A")
            context_parts.append(
                f"--- Document Content (Type: {doc_type}, Page: {page_label}, File: {file_name}) ---\n"
                f"{n.get_content()}\n"
                f"--------------------------------------------------------------------------------"
            )
        context_str_for_llm = "\n\n".join(context_parts)

        # --- 创建一个节点ID到完整节点对象的映射 ---
        retrieved_node_map: Dict[str, LlamaTextNode] = {node_with_score.node.id_: node_with_score.node for node_with_score in final_retrieved_nodes}
        logger.debug(f"Created map of {len(retrieved_node_map)} retrieved nodes for lookup.")

        # --- 主 RAG 查询 LLM ---
        llm_response_obj = None 
        for attempt in range(max_retries):
            try:
                final_prompt_for_llm = self.qa_prompt.format( # 使用 self.qa_prompt，因为它会调用 _create_qa_prompt
                    chat_history_context=chat_history_context_string,
                    context_str=context_str_for_llm, 
                    query_str=request.question
                )
                logger.info(f"final_prompt_for_llm {final_prompt_for_llm}.")

                if _tokenizer:
                    token_count = len(_tokenizer.encode(final_prompt_for_llm))
                    logger.info(f"Attempt {attempt + 1}: Total input tokens for LLM query: {token_count}. Query: '{request.question[:50]}...'")
                else:
                    logger.warning("Tokenizer not available, skipping token count for LLM query.")

                query_engine = RetrieverQueryEngine.from_args(
                    retriever=CustomRetriever(final_retrieved_nodes), 
                    llm=self.llm,
                    streaming=True,
                    text_qa_template=PromptTemplate(final_prompt_for_llm) 
                )
                
                llm_response_obj = await query_engine.aquery(request.question) 

                is_response_empty = False
                if llm_response_obj and hasattr(llm_response_obj, 'response_gen') and llm_response_obj.response_gen is not None:
                    pass 
                elif llm_response_obj and hasattr(llm_response_obj, 'response') and (not llm_response_obj.response or llm_response_obj.response.strip() == ""):
                    is_response_empty = True
                elif not llm_response_obj: 
                     is_response_empty = True

                if is_response_empty:
                    logger.warning(f"Attempt {attempt + 1}: LLM returned an empty or nearly empty response for query: '{request.question}'. Retrying...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt) 
                        continue 
                    else:
                        logger.error(f"All {max_retries} attempts failed for query: '{request.question}', returning error to client.")
                        logger.error(f"Failed query input details: \n"
                                     f"Chat History: {chat_history_context_string}\n"
                                     f"Document Content: {context_str_for_llm}\n" 
                                     f"User Query: {request.question}\n"
                                     f"Full Prompt (first 500 chars): {final_prompt_for_llm[:500]}...")
                        error_chunk_json = StreamChunk(content="Sorry, AI did not generate a valid response, please try a different way to ask questions.", is_last=True).json()
                        yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                        return

                break 

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error during LLM query for '{request.question}': {e}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt) 
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed for query: '{request.question}', returning error to client.")
                    logger.error(f"Failed query input details: \n"
                                 f"Chat History: {chat_history_context_string}\n"
                                 f"Document Content: {context_str_for_llm}\n" 
                                 f"User Query: {request.question}\n"
                                 f"Full Prompt (first 500 chars): {final_prompt_for_llm[:500]}...")
                    error_chunk_json = StreamChunk(content="Sorry, an error occurred during the query. Please try again later.", is_last=True).json()
                    yield f"data: {error_chunk_json}\n\n".encode("utf-8")
                    return

        rag_sources_info = []
        # === 基于 final_retrieved_nodes 来构建来源信息 ===
        # 确保只包含实际的文档节点，排除虚拟节点
        for node_with_score in final_retrieved_nodes:
            node = node_with_score.node # 获取 LlamaIndex 节点对象
            
            # 过滤掉虚拟节点
            if node.metadata.get("source") == "virtual_fallback" or \
               node.metadata.get("document_type") == "no_document_found": # 确保捕获所有虚拟节点标识
                continue
                
            # 确保 file_name 和 page_label 存在于 metadata 中，并处理可能的嵌套
            file_name = node.metadata.get("file_name", "未知文件")
            page_label = node.metadata.get("page_label", "未知页")
            material_id = node.metadata.get("material_id")

            # 如果 file_name 或 page_label 可能在 _node_content 的 metadata 中，需要解析
            # 这与之前在 debug_index 中的逻辑类似
            if '_node_content' in node.metadata and isinstance(node.metadata['_node_content'], str):
                try:
                    node_content_data = json.loads(node.metadata['_node_content'])
                    node_content_meta = node_content_data.get('metadata', {})
                    file_name = node_content_meta.get('file_name', file_name)
                    page_label = node_content_meta.get('page_label', page_label)
                except json.JSONDecodeError:
                    pass

            rag_sources_info.append({
                "file_name": file_name,
                "page_number": page_label,
                "material_id": material_id # 确保 material_id 也被正确传递
            })
        
        full_response_content = ""
        if llm_response_obj and hasattr(llm_response_obj, 'response_gen'):
            async for chunk_text in llm_response_obj.response_gen:
                full_response_content += chunk_text
                sse_event_json = StreamChunk(content=chunk_text, is_last=False).json()
                yield f"data: {sse_event_json}\n\n".encode("utf-8") 
            logger.debug("Successfully streamed response chunks.")
        elif llm_response_obj and hasattr(llm_response_obj, 'response') and llm_response_obj.response: 
            full_response_content = llm_response_obj.response
            sse_event_json = StreamChunk(content=full_response_content, is_last=False).json()
            yield f"data: {sse_event_json}\n\n".encode("utf-8")
            logger.warning(f"Non-streaming QueryEngine response yielded once: '{full_response_content[:50]}...'. Expected streaming.")
        else:
            logger.warning(f"No valid LLM response content (stream or direct response) for query: '{request.question}'. This should have been caught by retry logic.")
            error_chunk_json = StreamChunk(content="AI未能生成有效回复，请尝试换种方式提问。", is_last=True).json()
            yield f"data: {error_chunk_json}\n\n".encode("utf-8")
            return 
        

        if not full_response_content or full_response_content.strip() == "NO_RELEVANT_DOCUMENTS_FOUND":
             # 处理LLM找不到答案的情况
             # ... (你的 fallback 逻辑) ...
             return
    
        # 获取所有被引用的 chunk 的文本列表和 ID 列表
        referenced_chunk_texts = [node.text for node in retrieved_node_map.values()]
        referenced_chunk_ids = list(retrieved_node_map.keys())

        # 步骤 3.1: 将回答分割成句子
        import re
        # 使用正则表达式进行句子分割
        # 注意：这可能不完美，但对于大多数情况足够
        response_sentences = re.split(r'(?<=[.!?。！？])\s+', full_response_content)
        
        sentence_citations = []
        
        # 步骤 3.2: 遍历每个句子，找到它最相似的引用 chunk
        # LlamaIndex 的 `embedding_model` 实例可以用来生成句子向量
        # 你需要在 QueryService 中注入 embedding_model
        
        if self.embedding_model:
            # 批量获取句子和 chunk 的 embeddings，效率更高
            all_text_to_embed = response_sentences + referenced_chunk_texts
            all_embeddings = await self.embedding_model.aget_text_embedding_batch(all_text_to_embed, show_progress=False)
            
            response_sentence_embeddings = all_embeddings[:len(response_sentences)]
            referenced_chunk_embeddings = all_embeddings[len(response_sentences):]

            import numpy as np
            # 计算余弦相似度
            def cosine_similarity(v1, v2):
                v1 = np.array(v1)
                v2 = np.array(v2)
                dot_product = np.dot(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    return 0
                return dot_product / (norm_v1 * norm_v2)

            for i, sentence in enumerate(response_sentences):
                sentence_embedding = response_sentence_embeddings[i]
                best_match_id = None
                max_similarity = -1.0
                
                for j, chunk_embedding in enumerate(referenced_chunk_embeddings):
                    similarity = cosine_similarity(sentence_embedding, chunk_embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_id = referenced_chunk_ids[j]

                # 阈值判断：如果最高相似度低于某个阈值，可能就不认为是引用
                if max_similarity > 0.3:  # 可以根据实际情况调整阈值
                    source_node = retrieved_node_map.get(best_match_id)
                    if source_node:
                        # 确保元数据存在
                        source_meta = source_node.metadata
                        citation_info = {
                            "sentence": sentence,
                            "referenced_chunk_id": best_match_id,
                            "referenced_chunk_text": source_node.text,
                            "document_id": source_meta.get('document_id'), # 或其他你希望引用的 ID
                            "material_id": source_meta.get('material_id'),
                            "file_name": source_meta.get('file_name'),
                            "page_label": source_meta.get('page_label')
                        }
                        sentence_citations.append(citation_info)
                        
        else:
            logger.warning("Embedding model not available for sentence-level citation.")
            
        # --- 构建最终的 metadata ---
        final_metadata = {}
        
        final_metadata["rag_sources"] = rag_sources_info

        if final_referenced_material_ids: 
            final_metadata["referenced_docs"] = final_referenced_material_ids

        if generated_title:
            final_metadata["session_title"] = generated_title
        
        if sentence_citations:
            final_metadata["sentence_citations"] = sentence_citations
        
        logger.debug(f"DEBUG: Final metadata before StreamChunk creation: {final_metadata}")

        final_sse_event_json = StreamChunk(
            content="",
            is_last=True,
            metadata=final_metadata 
        ).json()
        temp_stream_chunk = StreamChunk(content="", is_last=True, metadata=final_metadata)
        logger.debug(f"DEBUG: Final StreamChunk object's dict (pre-json): {temp_stream_chunk.__dict__}")

        logger.debug(f"Yielding final raw JSON chunk: {final_sse_event_json}")
        yield f"data: {final_sse_event_json}\n\n".encode("utf-8")
