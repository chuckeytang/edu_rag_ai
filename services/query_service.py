import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from typing import AsyncGenerator, List, Union
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core import StorageContext, load_index_from_storage

from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings
import chromadb

# from llama_index.cura import CURARetriever
# print(CURARetriever)
from core.config import settings
from models.schemas import StreamChunk
from services.document_service import document_service
class QueryService:
    def __init__(self):
        self.chroma_path = settings.CHROMA_PATH
        self.collection_name = "documents"
        
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

    def _load_index_from_disk(self):
        try:
            chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            collection = chroma_client.get_or_create_collection(name=self.collection_name)
            chroma_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(
                vector_store=chroma_store
            )
            self.index = load_index_from_storage(storage_context, embed_model=self.embedding_model)
            logger.info("Index loaded from disk.")
        except Exception as e:
            logger.warning(f"Failed to load index from disk: {e}")

    def _initialize_index_on_startup(self):
        try:
            self._load_index_from_disk()
            if self.index:
                return
                
            # 如果加载失败就重新构建
            docs = document_service.load_documents()
            filtered_docs, _ = document_service.filter_documents(docs)
            self.initialize_index(filtered_docs)
            logger.info("Index initialized from fresh documents.")
        except Exception as e:
            logger.error(f"Failed to initialize index on startup: {e}")
        
    def initialize_index(self, documents):
                    
        def clean_metadata(documents):
            cleaned_docs = []
            for doc in documents:
                metadata = {}
                for k, v in doc.metadata.items():
                    if v is None:
                        metadata[k] = "unknown"
                    else:
                        metadata[k] = str(v) 
                doc.metadata = metadata
                cleaned_docs.append(doc)
            return cleaned_docs

        try:
            texts = [doc.text for doc in documents]
            embs = self.embedding_model.get_text_embedding_batch(texts)
            valid_docs = []
            valid_embs = []

            for i, (doc, emb) in enumerate(zip(documents, embs)):
                if emb is None or any(v is None for v in emb):
                    logger.warning(f"[Doc {i}] 无效 embedding，跳过此文档。")
                    continue
                valid_docs.append(doc)
                valid_embs.append(emb)
            
            if not valid_docs:
                raise ValueError("所有文档的 embedding 都无效，无法构建索引。")

            valid_docs = clean_metadata(valid_docs)

            # === 回退为 chroma_client 模式 ===
            chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            collection = chroma_client.get_or_create_collection(name=self.collection_name)
            chroma_store = ChromaVectorStore(chroma_collection=collection)

            storage_context = StorageContext.from_defaults(vector_store=chroma_store)

            # 构建索引
            self.index = VectorStoreIndex.from_documents(
                documents=valid_docs,
                storage_context=storage_context,
                embed_model=self.embedding_model
            )
            
            logger.info(f"Vector index initialized and persisted with {len(documents)} documents.")
            for i, doc in enumerate(documents):
                logger.debug(f"Chunk {i+1}: {doc.text[:100]}... (metadata: {doc.metadata})")
        except Exception as e:
            logger.error(f"Failed to initialize vector index: {e}")
            raise

    def update_index(self, new_documents):
        logger.info(f"Updating index with {len(new_documents)} documents...")
        
        if self.index is None:
            logger.info("Index is not initialized. Initializing now.")
            self.initialize_index(new_documents)
            logger.info("Index initialized successfully.")
        else:
            logger.info("Index already exists. Inserting new documents.")
            for idx, doc in enumerate(new_documents):
                try:
                    self.index.insert(doc)
                    logger.debug(f"✅ Inserted document {idx+1}: {doc.metadata.get('file_name', 'unknown')} - {doc.text[:80]}...")
                except Exception as e:
                    logger.error(f"Failed to insert document {idx+1}: {e}")

        logger.info("Index update completed.")

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