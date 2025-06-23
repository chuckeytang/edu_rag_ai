import logging
import os
import shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from typing import AsyncGenerator, List, Union
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, MetadataFilter
from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.core import VectorStoreIndex, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings

# from llama_index.cura import CURARetriever
# print(CURARetriever)
from core.config import settings
from models.schemas import StreamChunk
from services.document_service import document_service
from services.oss_service import oss_service

PERSIST_DIR = settings.INDEX_PATH
class QueryService:
    def __init__(self):
        """
        构造函数 - 实现“按需加载”策略
        启动时只初始化客户端和空的索引缓存，不加载任何实际的索引数据。
        """
        self.chroma_path = settings.CHROMA_PATH
        
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_path, # This path is now a local cache directory
            settings=Settings(
                is_persistent=True, # This MUST be True to enable persistence
            )
        )

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
        
        self.indices: Dict[str, VectorStoreIndex] = {}
        # 一个内存字典，用于缓存“按需加载”的索引对象
        # 键是 collection_name，值是 LlamaIndex 的 index 对象
        self.qa_prompt = self._create_qa_prompt()
        logger.info("QueryService initialized for on-demand index loading. No indices loaded at startup.")
        
    def _get_or_load_index(self, collection_name: str) -> VectorStoreIndex:
        """
        按需加载的核心方法。
        1. 检查内存缓存中是否存在指定collection的索引。
        2. 如果存在，直接返回。
        3. 如果不存在，从ChromaDB/OSS加载，存入缓存后再返回。
        """
        # 1. 检查内存缓存
        if collection_name in self.indices:
            logger.info(f"Returning cached index for collection: '{collection_name}'.")
            return self.indices[collection_name]

        # 2. 从持久化存储（ChromaDB/OSS）加载
        try:
            # 首先，确认ChromaDB中是否存在该collection
            logger.info(f"Checking for collection '{collection_name}' in ChromaDB...")
            collection = self.chroma_client.get_collection(name=collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
            logger.info(f"Collection '{collection_name}' found in ChromaDB.")

        except ValueError:
            # get_collection 找不到会抛出 ValueError，这意味着collection确实不存在
            logger.warning(f"Collection '{collection_name}' does not exist in ChromaDB.")
            return None

        # 3. 尝试从存储中加载完整的LlamaIndex索引（包括其元数据）
        try:
            logger.info(f"Attempting to load full index with metadata for '{collection_name}' from '{PERSIST_DIR}'...")
            index = load_index_from_storage(storage_context, embed_model=self.embedding_model)
            logger.info(f"Successfully loaded index with metadata for collection '{collection_name}'.")

        except Exception as e:
            # 4. 如果加载失败（很可能是本地元数据丢失），则从已有的VectorStore重建索引对象
            logger.warning(f"Could not load LlamaIndex metadata from '{PERSIST_DIR}' for collection '{collection_name}'. Error: {e}")
            logger.info(f"Reconstructing index object from existing VectorStore for '{collection_name}'...")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )
            logger.info(f"Successfully reconstructed index from VectorStore for '{collection_name}'.")
        
        # 5. 将加载或重建的索引存入内存缓存
        self.indices[collection_name] = index
        return index
        
    def initialize_index(self, documents: list, collection_name: str):
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
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            chroma_store = ChromaVectorStore(chroma_collection=collection)

            storage_context = StorageContext.from_defaults(vector_store=chroma_store)

            # 构建索引
            new_index = VectorStoreIndex.from_documents(
                documents=valid_docs,
                storage_context=storage_context,
                embed_model=self.embedding_model
            )

            # initialize_index() 内部 —— 关键断点
            logger.info(f"[DEBUG] 原始 documents 数 = {len(documents)}")

            # 过滤无 embedding
            logger.info(f"[DEBUG] 过滤后 valid_docs 数 = {len(valid_docs)}")

            # 如果 valid_docs 仍有内容，再看一下示例
            if valid_docs:
                logger.info(f"[DEBUG] 样例 meta = {valid_docs[0].metadata}")
                logger.info(f"[DEBUG] 样例 text = {valid_docs[0].text[:100]}...")

            logger.info(f"[DEBUG] 内存 docstore 节点数 = {len(new_index.storage_context.docstore.docs)}")
            new_index.storage_context.persist(persist_dir=PERSIST_DIR)
            logger.info("[DEBUG] persist() 已执行")
            
            self.indices[collection_name] = new_index
            logger.info(f"New index for collection '{collection_name}' has been cached in memory.")
            for i, doc in enumerate(documents):
                logger.debug(f"Chunk {i+1}: {doc.text[:100]}... (metadata: {doc.metadata})")

            # 检查 Chroma
            # client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
            # col = client.get_collection("documents")
            # print("Chroma 向量条数 =", col.count())
            
            # query_emb = self.embedding_model.get_text_embedding("photosynthesis")
            # res = col.query(query_embeddings=[query_emb],
            #                 n_results=2,
            #                 include=["metadatas"])
            # print("Chroma 查询结果(含 metadata) =", res)

            # 检查 llama-index
            # try:
            #     res = query_service.query("What is photosynthesis?")
            #     print("llama-index answer =", res)
            # except KeyError as e:
            #     print("llama-index 报错:", e)
            return new_index
        except Exception as e:
            logger.error(f"Failed to initialize vector index: {e}")
            raise

    def update_index(self, new_documents: list, collection_name: str):
        """
        为指定的collection更新索引。
        """
        logger.info(f"Updating index with {len(new_documents)} documents for collection: '{collection_name}'...")
        
        # 按需获取或加载索引
        index = self._get_or_load_index(collection_name)

        if index is None:
            logger.info(f"Index for '{collection_name}' not found. Initializing a new one.")
            self.initialize_index(new_documents, collection_name)
            logger.info(f"Index for '{collection_name}' initialized successfully.")
        else:
            logger.info(f"Index for '{collection_name}' already exists. Inserting new documents.")
            for idx, doc in enumerate(new_documents):
                try:
                    index.insert(doc)
                    logger.debug(f"✅ Inserted document {idx+1}: {doc.metadata.get('file_name', 'unknown')} - {doc.text[:80]}...")
                except Exception as e:
                    logger.error(f"Failed to insert document {idx+1}: {e}")
            logger.info(f"[DEBUG] 内存 docstore 节点数 = {len(index.storage_context.docstore.docs)}")
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            logger.info("[DEBUG] persist() 已执行")

        logger.info("Index update completed.")

    def query_with_filters(self, question: str, collection_name: str, filters: dict, similarity_top_k: int = 5, prompt: str = None):
        """
        对指定的collection执行带元数据过滤的查询。
        """
        logger.info(f"Querying collection '{collection_name}' with text: '{question}' and filters: {filters}")
        index = self._get_or_load_index(collection_name)

        if not index:
            logger.error(f"Query failed because collection '{collection_name}' does not exist.")
            raise ValueError(f"Collection '{collection_name}' does not exist or could not be loaded.")

        metadata_filters = self._build_metadata_filters(filters)
        logger.info(f"Constructed LlamaIndex MetadataFilters: {metadata_filters}")
        
        query_engine = index.as_query_engine(
            llm=self.llm,
            filters=metadata_filters,
            similarity_top_k=similarity_top_k,
            text_qa_template=self._create_qa_prompt(prompt)
        )

        logger.info(f"Querying with text: '{question}' and filters: {filters}")
        response = query_engine.query(question)
        return response


    def retrieve_with_filters(self, question: str, collection_name: str, filters: dict, similarity_top_k: int = 5):
        """
        [DEBUG METHOD] 仅执行带过滤的召回，并返回召回的节点列表。
        """
        logger.info(f"[DEBUG] Retrieving from collection '{collection_name}' with filters: {filters}")
        
        index = self._get_or_load_index(collection_name)
        if not index:
            raise ValueError(f"Collection '{collection_name}' does not exist or could not be loaded.")

        metadata_filters = self._build_metadata_filters(filters)

        # 1. 创建一个Retriever，配置和查询时完全一样
        retriever = index.as_retriever(
            filters=metadata_filters,
            similarity_top_k=similarity_top_k
        )
        
        # 2. 执行召回
        retrieved_nodes = retriever.retrieve(question)
        
        logger.info(f"[DEBUG] Retriever found {len(retrieved_nodes)} nodes matching the criteria.")
        
        # 3. 返回召回的原始节点
        return retrieved_nodes

    # Helper function to convert a simple dict to LlamaIndex's filter objects
    def _build_metadata_filters(self, filter_dict: dict) -> MetadataFilters:
        """
        Converts a dictionary of filters into a MetadataFilters object.
        Example input: {"subject": "Physics", "gen_year": "2023"}
        """
        filter_list = []
        for key, value in filter_dict.items():
            if value is None:
                continue
            
            # For lists like 'levelList', we might need special handling if needed,
            # but ChromaDB supports the 'in' operator. For now, we'll use simple equality.
            # LlamaIndex uses '==' as the default operator.
            new_filter = MetadataFilter(key=key, value=str(value)) # Ensure value is a string
            filter_list.append(new_filter)
        
        return MetadataFilters(filters=filter_list)

    # You can also create an async version for streaming if needed
    async def stream_query_with_filters(
        self, question: str, collection_name: str, filters: dict, similarity_top_k: int = 5, prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """
        对指定的collection执行带元数据过滤的流式查询。
        """
        logger.info(f"Streaming query on collection '{collection_name}' with text: '{question}' and filters: {filters}")

        index = self._get_or_load_index(collection_name)
        if not index:
            logger.error(f"Stream query failed because collection '{collection_name}' does not exist.")
            raise ValueError(f"Collection '{collection_name}' does not exist or could not be loaded.")

        metadata_filters = self._build_metadata_filters(filters)

        query_engine = index.as_query_engine(
            llm=self.llm,
            filters=metadata_filters,
            similarity_top_k=similarity_top_k,
            streaming=True,
            text_qa_template=self._create_qa_prompt(prompt)
        )
        
        response = await query_engine.aquery(question)
        if hasattr(response, 'response_gen'):
            async for chunk in response.response_gen:
                yield StreamChunk(content=chunk).json() + "\n"
        yield StreamChunk(content="", is_last=True).json() + "\n"
            
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
        
    def query(self, question: str, collection_name: str, similarity_top_k: int = 10, prompt: str = None):
        """
        对指定collection执行普通查询（无元数据过滤）。
        此方法现在调用功能更全的 query_with_filters 方法。
        """
        # 核心修改：调用带有filter功能的方法，并传递一个空的filters字典
        return self.query_with_filters(
            question=question,
            collection_name=collection_name,
            filters={}, # 传递空字典表示不过滤
            similarity_top_k=similarity_top_k,
            prompt=prompt
        )

    async def stream_query(self, question: str, collection_name: str, similarity_top_k: int = 10, prompt: str = None) -> AsyncGenerator[str, None]:
        """
        对指定collection执行流式查询（无元数据过滤）。
        此方法现在调用功能更全的 stream_query_with_filters 方法。
        """
        # 核心修改：调用带有filter功能的异步方法，并传递一个空的filters字典
        async for chunk in self.stream_query_with_filters(
            question=question,
            filters={}, # 传递空字典表示不过滤
            collection_name=collection_name,
            similarity_top_k=similarity_top_k,
            prompt=prompt
        ):
            yield chunk
            
    async def query_with_files(
            self,
            question: str,
            file_identifiers: List[str],
            similarity_top_k: int = 10,
            prompt: str = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        
        logger.info(f"Starting query_with_files with {len(file_identifiers)} identifiers.")
        
        permanent_paths = [] # 存放已经永久保存在本地的文件的路径
        temp_dirs_to_clean = [] # 存放为本次查询临时下载的文件的目录
        
        try:
            # --- 核心修改：双重查找和临时文件管理 ---
            for identifier in file_identifiers:
                path = None
                is_temporary = False

                # 1. 首先在 document_service (内容哈希) 中查找
                path = document_service.file_hashes.get(identifier)
                if path:
                    logger.info(f"  ✅ Found permanent path '{path}' for content_hash '{identifier}'.")

                # 2. 如果找不到，则尝试从OSS下载
                if not path:
                    try:
                        logger.info(f"  Identifier '{identifier}' not found locally, attempting to download from OSS...")
                        # oss_service.download_file_to_temp 会创建一个临时目录并返回完整路径
                        path = oss_service.download_file_to_temp(identifier)
                        is_temporary = True
                        # 将临时文件所在的目录记录下来，以便事后清理
                        temp_dirs_to_clean.append(os.path.dirname(path))
                        logger.info(f"  ✅ Successfully downloaded to temporary path '{path}' for oss_key '{identifier}'.")
                    except Exception as e:
                        logger.warning(f"  ❌ Failed to download or find identifier '{identifier}': {e}")
                        continue # 跳过这个无效的ID

                if path and os.path.exists(path):
                    permanent_paths.append(path)
                else:
                    logger.warning(f"  ❌ Path '{path}' for identifier '{identifier}' does not exist on disk.")

            if not permanent_paths:
                raise ValueError("No valid local documents could be found or downloaded for the given identifiers.")
            
            unique_paths = list(set(permanent_paths))
            logger.info(f"Found {len(unique_paths)} unique local files to build temporary index.")

            # --- 后续的临时索引和查询逻辑保持不变 ---
            target_docs = document_service.load_documents(unique_paths)
            if not target_docs:
                raise ValueError("Located files could not be loaded or are empty.")

            temp_index = VectorStoreIndex.from_documents(target_docs, embed_model=self.embedding_model)
            query_engine = temp_index.as_query_engine(
                llm=self.llm,
                streaming=True,
                similarity_top_k=min(similarity_top_k, len(target_docs)),
                text_qa_template=self._create_qa_prompt(prompt)
            )
            
            response = await query_engine.aquery(question)
            
            # 流式返回结果
            if hasattr(response, 'response_gen'):
                async for chunk in response.response_gen:
                    yield chunk
            else:
                yield response.response
        
        finally:
            # --- 无论成功还是失败，都清理本次查询下载的临时文件 ---
            if temp_dirs_to_clean:
                logger.info(f"Cleaning up {len(temp_dirs_to_clean)} temporary director(y/ies)...")
                for temp_dir in temp_dirs_to_clean:
                    try:
                        shutil.rmtree(temp_dir)
                        logger.info(f"  ✅ Successfully removed temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.error(f"  ❌ Failed to remove temporary directory {temp_dir}: {e}", exc_info=True)
            
query_service = QueryService()