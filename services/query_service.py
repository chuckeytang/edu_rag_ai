from typing import AsyncGenerator, List, Union
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from core.config import settings
from models.schemas import StreamChunk
from services.document_service import document_service
class QueryService:
    def __init__(self):
        self.embedding_model = DashScopeEmbedding(
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
            text_type="document"
        )
        self.llm = DashScope(
            model_name=DashScopeGenerationModels.QWEN_PLUS,
            api_key=settings.DASHSCOPE_API_KEY,
            max_tokens=4096,
            temperature=0.5,  
            similarity_cutoff=0.5  
        )
        self.index = None
        self.qa_prompt = self._create_qa_prompt()
        self._initialize_index_on_startup()
    def _initialize_index_on_startup(self):
        try:
            docs = document_service.load_documents()
            filtered_docs, _ = document_service.filter_documents(docs)
            self.initialize_index(filtered_docs)
            print("Index initialized successfully")
        except Exception as e:
            print(f"Failed to initialize index on startup: {e}")
    def initialize_index(self, documents):
        self.index = VectorStoreIndex(documents, embed_model=self.embedding_model)
    def update_index(self, new_documents):
        if self.index is None:
            self.initialize_index(new_documents)
        else:
            for doc in new_documents:
                self.index.insert(doc)
    def _create_qa_prompt(self, prompt: str = None) -> PromptTemplate:
        if prompt:
            return PromptTemplate(prompt)
        else:
            return self.default_qa_prompt
    @property
    def default_qa_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            "You have documents including exam papers and syllabuses. "
            "The user asks about a specific question from the exam paper, formatted as 'X(Y)(Z)' (e.g., '6(b)(ii)'), "
            "where X is the main question number, Y is the subsection, and Z is the sub-subsection. "
            "Follow these steps:\n"
            "1. Locate and describe the content of the specified question.\n"
            "2. Identify the most relevant knowledge point from the syllabus.\n"
            "3. Provide a clear and concise response.\n"
            "Document content: {context_str}\n"
            "Query: {query_str}"
        )
    def query(self, question: str, similarity_top_k: int = 10, prompt: str = None):
        if not self.index:
            self._initialize_index_on_startup()
            if not self.index:
                raise ValueError("Index not initialized. Failed to load documents.")
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            streaming=False,
            similarity_top_k=similarity_top_k,
            verbose=True,
            text_qa_template=self._create_qa_prompt(prompt)
        )
        return query_engine.query(question)
    async def stream_query(self, question: str, similarity_top_k: int = 10, prompt: str = None) -> AsyncGenerator[str, None]:
        if not self.index:
            self._initialize_index_on_startup()
            if not self.index:
                raise ValueError("Index not initialized. Failed to load documents.")
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            streaming=True,  
            similarity_top_k=similarity_top_k,
            text_qa_template=self._create_qa_prompt(prompt)
        )
        response = await query_engine.aquery(question)
        if hasattr(response, 'response_gen'):
            async for chunk in response.response_gen:
                yield StreamChunk(content=chunk).json() + "\n"
            yield StreamChunk(content="", is_last=True).json() + "\n"
        else:
            yield StreamChunk(content=response.response).json() + "\n"
            yield StreamChunk(content="", is_last=True).json() + "\n"
    async def query_with_files(
            self,
            question: str,
            file_names: List[str],
            similarity_top_k: int = 10,
            prompt: str = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        target_docs = document_service.get_documents_by_filenames(file_names)
        if not target_docs:
            raise ValueError("No valid documents found with the given filenames")
        temp_index = VectorStoreIndex(target_docs, embed_model=self.embedding_model)
        query_engine = temp_index.as_query_engine(
            llm=self.llm,
            streaming=True,
            similarity_top_k=similarity_top_k,
            text_qa_template=self._create_qa_prompt(prompt)
        )
        response = await query_engine.aquery(question)
        if hasattr(response, 'response_gen'):
            async for chunk in response.response_gen:
                yield chunk
        else:
            yield response.response
query_service = QueryService()