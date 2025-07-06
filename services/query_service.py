import logging
import os
import shutil

from llama_cloud import TextNode

from core.metadata_utils import prepare_metadata_for_storage
from services import chat_history_service

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from typing import Any, AsyncGenerator, List, Union
from llama_index.core.schema import Document as LlamaDocument

from llama_index.core import VectorStoreIndex, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings

# from llama_index.cura import CURARetriever
# print(CURARetriever)
from core.config import settings
from models.schemas import ChatQueryRequest, StreamChunk
from services.document_service import document_service
from services.oss_service import oss_service
from typing import List, Dict

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
        3. 如果不存在，从ChromaDB获取或创建collection，构建索引对象，存入缓存后再返回。
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

        except (Exception, ValueError) as e:
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
                logger.info(f"[DEBUG] 样例 meta = {valid_docs[0].extra_info}")
                logger.info(f"[DEBUG] 样例 text = {valid_docs[0].text[:100]}...")

            logger.info(f"[DEBUG] 内存 docstore 节点数 = {len(new_index.storage_context.docstore.docs)}")
            new_index.storage_context.persist(persist_dir=PERSIST_DIR)
            logger.info("[DEBUG] persist() 已执行")
            
            self.indices[collection_name] = new_index
            logger.info(f"New index for collection '{collection_name}' has been cached in memory.")
            for i, doc in enumerate(documents):
                logger.debug(f"Chunk {i+1}: {doc.text[:100]}... (metadata: {doc.extra_info})")

            # 检查 Chroma
            # client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
            # col = client.get_collection("documents")
            # print("Chroma 向量条数 =", col.count())
            
            # query_emb = self.embedding_model.get_text_embedding("photosynthesis")
            # res = col.query(query_embeddings=[query_emb],
            #                 n_results=2,
            #                 include=["metadatas"])
            # print("Chroma 查询结果(含 metadata) =", res)
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
                    logger.debug(f"✅ Inserted document {idx+1}: {doc.extra_info.get('file_name', 'unknown')} - {doc.text[:80]}...")
                except Exception as e:
                    logger.error(f"Failed to insert document {idx+1}: {e}")
            logger.info(f"[DEBUG] 内存 docstore 节点数 = {len(index.storage_context.docstore.docs)}")
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            logger.info("[DEBUG] persist() 已执行")

        logger.info("Index update completed.")

    def get_nodes_by_metadata_filter(self, collection_name: str, filters: dict) -> List[TextNode]:
        """
        根据元数据过滤器，直接从ChromaDB中获取所有匹配的节点。
        这不是相似度搜索，而是精确的数据拉取。
        """
        logger.info(f"Directly fetching nodes from '{collection_name}' with filters: {filters}")
        collection = self.chroma_client.get_collection(name=collection_name)
        
        # 使用 collection.get() 方法进行元数据过滤
        # include 列表确保我们取回所有需要的信息
        results = collection.get(
            where=filters,
            include=["metadatas", "documents"]
        )
        
        if not results or not results['ids']:
            logger.info(f"No nodes found for filters: {filters}")
            return []
            
        # 将ChromaDB返回的结果重新构建为LlamaIndex的TextNode对象列表
        nodes = []
        for i in range(len(results['ids'])):
            node = TextNode(
                id_=results['ids'][i],
                text=results['documents'][i],
                extra_info=results['metadatas'][i] or {}
            )
            nodes.append(node)
            
        logger.info(f"Found {len(nodes)} nodes matching the filters.")
        return nodes

    # --- 通用的元数据更新方法 ---
    def update_existing_nodes_metadata(self, collection_name: str, material_id: int, metadata_update_payload: dict):
        """
        根据 material_id 找到所有相关节点，并更新它们的元数据。
        这是一个“合并更新”，而不是完全替换。
        """
        logger.info(f"Performing generic metadata update for material_id: {material_id} with payload: {metadata_update_payload}")
        
        # payload 不应该包含任何需要“抢救”的字段，但为安全起见我们仍以它为基础
        if metadata_update_payload is None:
            logger.warning("Update payload is None, skipping update.")
            return

        collection = self.chroma_client.get_collection(name=collection_name)
        
        # 1. 找到所有需要更新的节点
        nodes_to_update = collection.get(where={"material_id": material_id}, include=["metadatas"])
        if not nodes_to_update or not nodes_to_update['ids']:
            logger.warning(f"No nodes found for material_id {material_id} during generic update. Nothing to do.")
            return

        # 2. 准备更新数据：以新数据为基础，嫁接回不可变的核心数据
        updated_metadatas = []
        # 定义一个不应被本次更新覆盖的核心字段集合
        preserved_keys = {"material_id", "author_id", "file_key", "file_name", "file_size"}

        for old_meta in nodes_to_update['metadatas']:
            # a. 以传入的新数据为基础，这是一个干净的起点
            new_meta = metadata_update_payload.copy()
            
            # b. 从旧元数据中“抢救”回那些必须保留的核心字段
            for key in preserved_keys:
                if key in old_meta:
                    new_meta[key] = old_meta[key]
            
            # c. 这样，最终的 new_meta 就实现了“替换”效果：
            #    - 包含了所有新传入的字段。
            #    - 不包含新数据中没有、但旧数据中有的字段（如 exam）。
            #    - 确保了核心字段（如 material_id, accessible_to）被保留。
            updated_metadatas.append(new_meta)
        
        # 3. 执行批量更新
        collection.update(
            ids=nodes_to_update['ids'],
            metadatas=updated_metadatas
        )
        logger.info(f"Successfully updated metadata for {len(nodes_to_update['ids'])} nodes of material_id {material_id}.")

    def add_public_acl_to_material(self, material_id: int, collection_name: str) -> dict:
        """
        为已存在的私有文档添加公共权限。
        增加了幂等性检查，如果已存在公共节点，则直接返回成功。
        """
        task_status = "error"
        message = f"An unexpected error occurred while publishing material {material_id}."
        
        try:
            collection = self.chroma_client.get_collection(name=collection_name)

            # --- 增加幂等性检查 ---
            logger.info(f"Checking for existing public nodes for material_id: {material_id}")
            existing_public_nodes = collection.get(
                where={"$and": [{"material_id": material_id}, {"accessible_to": "public"}]},
                limit=1 # 我们只需要知道是否存在，所以 limit=1 效率最高
            )
            if existing_public_nodes and existing_public_nodes['ids']:
                message = f"Material {material_id} has already been published. Operation is idempotent."
                task_status = "duplicate" # 使用 "duplicate" 或 "success" 均可
                logger.warning(message)
                return {"message": message, "status": task_status}

            # 1. 找到该 material_id 对应的所有节点（它们应该是私有的）
            logger.info(f"Finding existing nodes for material_id: {material_id} to publish.")
            results = collection.get(
                where={"material_id": material_id},
                include=["metadatas", "documents"] # 我们需要完整信息来创建副本
            )
            
            if not results or not results['ids']:
                message = f"No existing nodes found for material_id {material_id} to publish."
                task_status = "not_found"
                logger.warning(message)
                return {"message": message, "status": task_status}

            # 2. 为每个找到的节点创建一个新的“公共”节点对象
            public_nodes_to_add = []
            for i in range(len(results['ids'])):
                metadata_copy = results['metadatas'][i].copy() if results['metadatas'][i] else {}
                # 检查是否已经存在公共副本，避免重复添加
                if metadata_copy.get('accessible_to') == 'public':
                    logger.warning(f"Node {results['ids'][i]} for material_id {material_id} is already public. Skipping.")
                    continue
                
                # 设置新节点的权限为 'public'
                metadata_copy['accessible_to'] = 'public'
                
                # 创建新的 LlamaDocument 对象
                # 注意：ID必须是唯一的，我们可以在旧ID上加一个前缀
                new_public_node = LlamaDocument(
                    text=results['documents'][i],
                    metadata=metadata_copy,
                )
                public_nodes_to_add.append(new_public_node)
            
            # 3. 将这些新的公共节点插入到索引中
            if public_nodes_to_add:
                logger.info(f"Adding {len(public_nodes_to_add)} new public nodes for material_id {material_id}.")
                # 调用自身的 update_index 方法来执行插入
                self.update_index(public_nodes_to_add, collection_name)
            else:
                logger.info(f"No new public nodes needed for material_id {material_id}, it might have been published already.")

            message = f"Successfully published material {material_id} by adding/verifying public nodes."
            task_status = "success"
            logger.info(message)

        except Exception as e:
            logger.error(f"Error during publishing of material_id '{material_id}': {e}", exc_info=True)
            message = str(e)
            task_status = "error"
        
        return {
            "message": message,
            "status": task_status,
        }
    
    def query_with_filters(self, question: str, collection_name: str, filters: dict, similarity_top_k: int = 5, prompt: str = None):
        """
        对指定的collection执行带元数据过滤的查询。
        """
        logger.info(f"Querying collection '{collection_name}' with text: '{question}' and filters: {filters}")
        index = self._get_or_load_index(collection_name)

        if not index:
            logger.error(f"Query failed because collection '{collection_name}' does not exist.")
            raise ValueError(f"Collection '{collection_name}' does not exist or could not be loaded.")

        chroma_where_clause = self._build_chroma_where_clause(filters)
        logger.info(f"Constructed ChromaDB `where` clause: {chroma_where_clause}")
        
        query_engine = index.as_query_engine(
            llm=self.llm,
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
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

        chroma_where_clause = self._build_chroma_where_clause(filters)

        # 1. 创建一个Retriever，配置和查询时完全一样
        retriever = index.as_retriever(
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
            similarity_top_k=similarity_top_k
        )
        
        # 2. 执行召回
        retrieved_nodes = retriever.retrieve(question)
        
        logger.info(f"[DEBUG] Retriever found {len(retrieved_nodes)} nodes matching the criteria.")
        
        # 3. 返回召回的原始节点
        return retrieved_nodes

    def _build_chroma_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        为 ChromaDB 构建原生 where 子句。
        此版本只使用 ChromaDB 明确支持的操作符，如 $in 和 $eq。
        """
        if not filters:
            return {}

        chroma_filters = []
        
        for key, value in filters.items():
            # 对 accessible_to 字段，直接使用 $in 操作符
            if key == "accessible_to" and isinstance(value, list) and value:
                chroma_filters.append({"accessible_to": {"$in": value}})
            # 其他所有字段，使用 $eq (等于) 操作符
            elif key != "accessible_to":
                chroma_filters.append({key: {"$eq": value}})

        if not chroma_filters:
            return {}
        
        # 如果有多个过滤条件，用 $and 连接
        if len(chroma_filters) > 1:
            return {"$and": chroma_filters}
        
        # 只有一个条件，直接返回
        return chroma_filters[0]

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

        chroma_where_clause = self._build_chroma_where_clause(filters)
        logger.info(f"Constructed ChromaDB `where` clause: {chroma_where_clause}")
        
        query_engine = index.as_query_engine(
            llm=self.llm,
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
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
        
        logger.info(f"Executing query within specific files (material_ids): {file_identifiers}")

        # --- 1. 输入校验与准备 ---
        if not file_identifiers:
            raise ValueError("file_identifiers list cannot be empty for a file-specific query.")

        try:
            material_ids_as_int = [int(mid) for mid in file_identifiers]
        except ValueError:
            raise TypeError("All file_identifiers must be valid integers (material_id).")

        # --- 2. 直接构建最终、正确的 ChromaDB `where` 子句 ---
        # 这是整个方案的核心，确保我们只使用被明确支持的 $in 操作符
        chroma_where_clause = {
            "material_id": {
                "$in": material_ids_as_int
            }
        }
        logger.info(f"Constructed a direct and final ChromaDB `where` clause: {chroma_where_clause}")

        # --- 3. 获取索引对象 (复用现有逻辑) ---
        # 假设所有文档都在 'public_collection'，如果不是，这里需要动态传入
        collection_name = "public_collection" 
        index = self._get_or_load_index(collection_name)
        if not index:
            error_message = f"Query failed because collection '{collection_name}' does not exist."
            logger.error(error_message)
            raise ValueError(error_message)

        # --- 4. 直接构建查询引擎，不再调用其他方法 ---
        # 将我们手动构建的、100%正确的 where 子句通过 vector_store_kwargs 传递
        query_engine = index.as_query_engine(
            llm=self.llm,
            vector_store_kwargs={"where": chroma_where_clause},
            similarity_top_k=similarity_top_k,
            streaming=True,
            text_qa_template=self._create_qa_prompt(prompt)
        )
        
        # --- 5. 执行查询并流式返回结果 (逻辑不变) ---
        response = await query_engine.aquery(question)
        
        if hasattr(response, 'response_gen'):
            async for chunk in response.response_gen:
                yield chunk
        # 在 LlamaIndex 的某些版本中，非流式结果也可能需要这样处理
        elif hasattr(response, 'response'):
             yield response.response
        else:
             logger.warning("Response object has no 'response_gen' or 'response' attribute.")
             yield ""
            
    def delete_nodes_by_metadata(self, collection_name: str, filters: dict) -> dict:
        """
        根据元数据过滤器，直接从ChromaDB中删除所有匹配的节点。
        """
        logger.info(f"Attempting to delete nodes from '{collection_name}' with filters: {filters}")
        
        if not filters:
            message = "Deletion filters cannot be empty."
            logger.error(message)
            return {"status": "error", "message": message}
            
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            
            # 先查询一下有多少条匹配，方便打日志
            results = collection.get(where=filters, include=[])
            count = len(results.get('ids', []))

            if count == 0:
                message = f"No documents found matching the filters. Nothing to delete."
                logger.warning(message)
                return {"status": "success", "message": message}

            # 调用 ChromaDB 的核心删除方法
            collection.delete(where=filters)
            
            # 持久化变更（如果需要，取决于您的ChromaDB配置）
            # self.chroma_client.persist() 
            
            message = f"Successfully deleted {count} document nodes matching filters: {filters}"
            logger.info(message)
            return {"status": "success", "message": message}

        except Exception as e:
            message = f"An error occurred during deletion: {e}"
            logger.error(message, exc_info=True)
            return {"status": "error", "message": message}
        
    async def rag_query_with_context(
        self,
        request: ChatQueryRequest # 直接接收 ChatQueryRequest 对象
    ) -> AsyncGenerator[str, None]:
        
        logger.info(f"Starting RAG query for session {request.session_id}, user {request.account_id} with query: '{request.question}'")
        
        # --- 1. 语义检索历史聊天上下文 ---
        chat_history_context_string = ""
        try:
            chat_history_context_nodes = chat_history_service.retrieve_chat_history_context(
                session_id=request.session_id,
                account_id=request.account_id,
                query_text=request.context_retrieval_query,
                top_k=request.similarity_top_k or 5 # 使用请求中的top_k或默认值
            )
            if chat_history_context_nodes:
                chat_history_context_string = "以下是与您当前问题相关的历史对话片段：\n" + \
                                              "\n".join([f"[{node.metadata.get('role', '未知')}]: {node.text}" for node in chat_history_context_nodes]) + \
                                              "\n---\n"
            logger.info(f"Chat history context: \n{chat_history_context_string}")
        except Exception as e:
            logger.error(f"Failed to retrieve chat history context: {e}", exc_info=True)
            chat_history_context_string = ""

        # --- 2. 主 RAG 查询 ---
        rag_index = self._get_or_load_index(request.collection_name)
        if not rag_index:
            logger.error(f"RAG collection '{request.collection_name}' does not exist or could not be loaded.")
            yield StreamChunk(content="抱歉，知识库未准备好，请稍后再试。", is_last=True).json() + "\n"
            return

        combined_rag_filters = request.filters if request.filters else {}
        if request.target_file_ids and len(request.target_file_ids) > 0:
            try:
                material_ids = [int(mid) for mid in request.target_file_ids]
                if "material_id" in combined_rag_filters and isinstance(combined_rag_filters["material_id"], dict) and "$in" in combined_rag_filters["material_id"]:
                    current_material_ids = combined_rag_filters["material_id"]["$in"]
                    # 确保不重复添加
                    combined_rag_filters["material_id"]["$in"] = list(set(current_material_ids + material_ids))
                else:
                    combined_rag_filters["material_id"] = {"$in": material_ids}
            except ValueError:
                logger.warning("Invalid material_id in target_file_ids. Ignoring file filter.")
        
        chroma_where_clause = self._build_chroma_where_clause(combined_rag_filters)
        logger.info(f"Main RAG ChromaDB `where` clause: {chroma_where_clause}")

        # 组合最终的LLM提示词
        # 确保 default_qa_prompt 有 {chat_history_context} 占位符
        final_qa_prompt_template = self._create_qa_prompt(request.prompt) # 获取或创建 PromptTemplate
        final_qa_prompt = final_qa_prompt_template.partial_format(chat_history_context=chat_history_context_string)

        query_engine = rag_index.as_query_engine(
            llm=self.llm,
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
            similarity_top_k=request.similarity_top_k or 5,
            streaming=True,
            text_qa_template=final_qa_prompt
        )
        
        response = await query_engine.aquery(request.question)
        
        rag_sources_info = []
        if response.source_nodes:
            for node in response.source_nodes:
                rag_sources_info.append({
                    "text_excerpt": node.get_content().strip()[:200] + "...",
                    "file_name": node.metadata.get("file_name", "未知文件"),
                    "page_number": node.metadata.get("page_label", "未知页"),
                    "material_id": node.metadata.get("material_id")
                })
        
        full_response_content = ""
        if hasattr(response, 'response_gen'):
            async for chunk_text in response.response_gen:
                full_response_content += chunk_text
                yield StreamChunk(content=chunk_text, is_last=False).json() + "\n"
        
        yield StreamChunk(
            content="",
            is_last=True,
            metadata={"rag_sources": rag_sources_info}
        ).json() + "\n"
query_service = QueryService()