# api/services/indexer_service.py
import time
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_cloud import TextNode
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.kvstore import SimpleKVStore

import chromadb
from chromadb.config import Settings

from core.config import settings
from core.rag_config import RagConfig

logger = logging.getLogger(__name__)

PERSIST_DIR = settings.INDEX_PATH

class IndexerService:
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embedding_model: BaseEmbedding,
                 rag_config: RagConfig):
        
        self.chroma_client = chroma_client
        self.embedding_model = embedding_model
        self._current_rag_config = rag_config
        # 内存缓存，用于按需加载索引对象
        self.indices: Dict[str, VectorStoreIndex] = {} 
        
        self.node_parser = SentenceSplitter( 
            chunk_size=self._current_rag_config.chunk_size,  
            chunk_overlap=self._current_rag_config.chunk_overlap, 
        )
        os.makedirs(PERSIST_DIR, exist_ok=True)

        logger.info("IndexerService initialized.")

    # 用于按需更新 SentenceSplitter
    def _update_node_parser_if_needed(self, new_rag_config: RagConfig):
        # 检查关键参数是否改变
        if (new_rag_config.chunk_size != self._current_rag_config.chunk_size or
            new_rag_config.chunk_overlap != self._current_rag_config.chunk_overlap):
            
            logger.info("RAG config for SentenceSplitter has changed. Re-initializing node_parser.")
            # 只有当参数发生变化时才重新创建
            self.node_parser = SentenceSplitter( 
                chunk_size=new_rag_config.chunk_size,  
                chunk_overlap=new_rag_config.chunk_overlap, 
            )
            self._current_rag_config = new_rag_config
        else:
            logger.debug("RAG config for SentenceSplitter is unchanged. Skipping re-initialization.")
            
    def _get_or_load_index(self, collection_name: str) -> Optional[VectorStoreIndex]:
        if collection_name in self.indices:
            logger.info(f"Returning cached index for collection: '{collection_name}'.")
            return self.indices[collection_name]

        collection = None
        vector_store = None
        
        # --- 步骤 1: 确保 ChromaDB Collection 可用 ---
        try:
            logger.info(f"Checking for collection '{collection_name}' in ChromaDB...")
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            logger.info(f"Collection '{collection_name}' found/created in ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to get or create collection '{collection_name}' in ChromaDB: {e}", exc_info=True)
            return None # 如果 ChromaDB 本身就无法连接或创建 Collection，则无法继续

        index = None
        
        # --- 步骤 2: 尝试从本地持久化存储加载 StorageContext 和 Index ---
        docstore_path = os.path.join(PERSIST_DIR, "docstore.json")
        index_store_path = os.path.join(PERSIST_DIR, "index_store.json")

        load_from_disk_success = False
        storage_context_from_disk = None # 声明在这里，以便在重建时使用

        if os.path.exists(docstore_path) and os.path.exists(index_store_path):
            try:
                # 额外检查文件是否为空
                if os.path.getsize(docstore_path) > 0 and os.path.getsize(index_store_path) > 0:
                    # 只有当文件存在且不为空时，才尝试从本地路径创建 storage_context
                    # 这行不会引发 FileNotFoundError，因为它前面已经检查了文件是否存在
                    storage_context_from_disk = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
                    index = load_index_from_storage(storage_context=storage_context_from_disk, embed_model=self.embedding_model)
                    logger.info(f"Successfully loaded index metadata for collection '{collection_name}' from '{PERSIST_DIR}'.")
                    load_from_disk_success = True
                else:
                    logger.warning(f"Local LlamaIndex metadata files found but are empty. Will reconstruct index for '{collection_name}'.")
            except Exception as e: # 捕获加载过程中的其他错误，例如文件损坏
                logger.warning(f"Could not load LlamaIndex metadata from '{PERSIST_DIR}' for collection '{collection_name}' (Error during load: {e}).", exc_info=True)
                logger.warning(f"Will reconstruct index object from existing VectorStore for '{collection_name}'.")
        else:
            logger.info(f"Local LlamaIndex metadata files (docstore.json, index_store.json) not found. Will reconstruct index for '{collection_name}'.")

        # --- 步骤 3: 如果未能从磁盘加载，则从 VectorStore 重建 Index ---
        if not load_from_disk_success:
            logger.info(f"Creating a new StorageContext and reconstructing index object from existing VectorStore for '{collection_name}'.")
            
            # 手动实例化空的存储组件，而不是从文件加载
            # 这确保了即使文件不存在，也不会抛出 FileNotFoundError
            docstore = SimpleDocumentStore()
            index_store = SimpleIndexStore()
            
            # 使用手动创建的存储组件来创建 StorageContext
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, 
                docstore=docstore, 
                index_store=index_store, 
            )
            
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embedding_model,
                storage_context=storage_context # 传入 StorageContext，以便它能保存新的元数据
            )
            
            # 重建后，立即持久化，确保本地文件被创建
            try:
                index.storage_context.persist(persist_dir=PERSIST_DIR)
                logger.info(f"Successfully reconstructed and persisted index metadata for '{collection_name}'.")
            except Exception as e:
                logger.error(f"Failed to persist reconstructed index for '{collection_name}': {e}", exc_info=True)
                logger.warning(f"Returning index in memory, but persistence may be an issue.")
        
        # 确保 index 不为 None
        if index is None:
            logger.error(f"Failed to load or reconstruct index for collection '{collection_name}'. Returning None.")
            return None

        self.indices[collection_name] = index
        return index

    def add_documents_to_index(self, 
                               documents: List[LlamaDocument], 
                               collection_name: str,
                               rag_config: RagConfig) -> VectorStoreIndex:
        logger.info(f"Attempting to add {len(documents)} documents to index for collection: '{collection_name}'...")
        if not documents:
            logger.warning("No LlamaDocuments provided for indexing. Skipping.")
            return self._get_or_load_index(collection_name) 
        
        self._update_node_parser_if_needed(rag_config)

        logger.debug(f"Received {len(documents)} raw LlamaDocuments for collection: '{collection_name}'.")
        all_nodes_to_add: List[TextNode] = [] 

        # --- 增加节点处理前的日志 ---
        logger.debug(f"Starting document processing into nodes for collection: '{collection_name}'.")
        processing_start_time = time.time() # 导入 time 模块
        for doc in documents:
            doc_id = doc.id_ if doc.id_ else "no_id"
            doc_type = doc.metadata.get("document_type", "Unknown")
            logger.debug(f"Processing document ID: '{doc_id}', Type: '{doc_type}', Text Length: {len(doc.text)}.")
            
            if doc_type == "PDF_Table_Chunk" or doc_type == "PDF_Table" or doc_type == "PDF_Table_Row":
                node = TextNode(
                    id_=doc.id_, 
                    text=doc.text,
                    metadata=doc.metadata 
                )
                all_nodes_to_add.append(node)
                logger.debug(f" [Indexer] Directly adding {doc_type} node (ID: {node.id_}): '{node.text[:50]}...'")
            else:
                # 对于其他文档类型（包括 PDF_Text），使用 node_parser 进行细粒度 chunking
                # LlamaIndex 的 node_parser 接受 Documents 列表，返回 TextNode 列表
                try:
                    split_nodes = self.node_parser.get_nodes_from_documents([doc]) 
                    for node in split_nodes:
                        # 确保 extra_info 是可修改的字典
                        if not isinstance(node.extra_info, dict):
                            node.extra_info = {}

                        node.extra_info.update(doc.metadata)
                        
                        # 合并原始文档的元数据到 TextNode 的 extra_info
                        for key, value in node.extra_info.items():
                            if isinstance(value, list):
                                try:
                                    node.extra_info[key] = json.dumps(value) 
                                except TypeError:
                                    logger.error(f" [Indexer] Failed to JSON serialize metadata key '{key}' with value '{value}'. Keeping original.")
                        
                        all_nodes_to_add.append(node)
                        logger.debug(f" [Indexer] Adding chunked node (ID: {node.id_}, Type: {doc_type}): '{node.text[:50]}...'")
                except Exception as e:
                    logger.error(f" [Indexer] Error chunking document ID '{doc_id}': {e}", exc_info=True)
        
        processing_end_time = time.time()
        logger.info(f" [Indexer] Document processing into nodes completed. Took {processing_end_time - processing_start_time:.2f} seconds. Original {len(documents)} documents processed into {len(all_nodes_to_add)} final nodes for indexing.")

        # 尝试获取现有索引
        index = self._get_or_load_index(collection_name)

        if index is None:
            logger.info(f" [Indexer] Index for '{collection_name}' not found/loadable. Initializing a new one.")
            # --- 增加初始化新索引前的日志 ---
            init_start_time = time.time()
            try:
                collection = self.chroma_client.get_or_create_collection(name=collection_name)
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)

                new_index = VectorStoreIndex.from_documents(
                    documents=all_nodes_to_add,
                    storage_context=storage_context,
                    embed_model=self.embedding_model,
                )
                logger.info(f" [Indexer] New index for '{collection_name}' initialized successfully.")
                index = new_index
            except Exception as e:
                logger.error(f" [Indexer] Failed to initialize new index for '{collection_name}': {e}", exc_info=True)
                raise # 重新抛出异常，让上层捕获
            finally:
                init_end_time = time.time()
                logger.info(f" [Indexer] Index initialization for '{collection_name}' took {init_end_time - init_start_time:.2f} seconds.")
        else:
            logger.info(f" [Indexer] Index for '{collection_name}' already exists. Inserting new nodes.")
            # --- 增加插入新节点前的日志 ---
            insert_start_time = time.time()
            for idx, node in enumerate(all_nodes_to_add):
                try:
                    # 记录每个节点插入前的状态
                    logger.debug(f" [Indexer] Attempting to insert node {idx+1}/{len(all_nodes_to_add)}: {node.id_} - '{node.text[:50]}...'")
                    node_insert_start_time = time.time()
                    index.insert_nodes([node]) 
                    node_insert_end_time = time.time()
                    logger.debug(f" [Indexer] ✅ Inserted node {idx+1}: {node.id_} - '{node.text[:50]}...' in {node_insert_end_time - node_insert_start_time:.4f} seconds.")
                except Exception as e:
                    logger.error(f" [Indexer] Failed to insert node {idx+1} ({node.id_}) into index: {e}", exc_info=True)
            insert_end_time = time.time()
            logger.info(f" [Indexer] All {len(all_nodes_to_add)} nodes insertion for '{collection_name}' took {insert_end_time - insert_start_time:.2f} seconds.")
        
        # --- 增加持久化操作的日志 ---
        persist_start_time = time.time()
        try:
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            logger.info(f" [Indexer] Index for '{collection_name}' updated and persisted successfully.")
        except Exception as e:
            logger.error(f" [Indexer] Failed to persist index for '{collection_name}': {e}", exc_info=True)
            # 这里可以选择 re-raise 异常或者进行其他错误处理
            raise
        finally:
            persist_end_time = time.time()
            logger.info(f" [Indexer] Index persistence for '{collection_name}' took {persist_end_time - persist_start_time:.2f} seconds.")
            
        self.indices[collection_name] = index
        return index

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
            
            def _prepare_delete_where_clause(input_filters: Dict[str, Any]) -> Dict[str, Any]:
                if not input_filters:
                    return {}

                internal_filters = []
                for key, value in input_filters.items():
                    # 假设这里只处理简单键值对，或者已经预格式化的 {key: {"$op": value}}
                    # 如果 value 是一个字典（表示已包含操作符），直接使用
                    if isinstance(value, dict) and any(op in value for op in ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$and", "$or"]):
                        internal_filters.append({key: value})
                    elif isinstance(value, list): # 确保列表转换为 $in
                         if value:
                             internal_filters.append({key: {"$in": value}})
                         else: # 空列表条件不匹配任何内容
                             logger.warning(f"Empty list provided for filter key '{key}'. This condition will be ignored in delete.")
                    else: # 简单值，转换为 $eq
                        internal_filters.append({key: {"$eq": value}})

                if not internal_filters:
                    return {}
                
                # 只有当条件多于一个时才使用 $and
                if len(internal_filters) > 1:
                    return {"$and": internal_filters}
                
                # 只有一个条件，直接返回这个条件字典
                return internal_filters[0]

            # 调用辅助方法来构建最终的 where 子句
            chroma_filters_for_delete = _prepare_delete_where_clause(filters)

            logger.debug(f"ChromaDB delete where clause prepared: {chroma_filters_for_delete}")
            
            # 使用修正后的 filters 来调用 delete 方法
            deleted_ids = collection.delete(where=chroma_filters_for_delete)
            
            count = len(deleted_ids) if deleted_ids else 0

            if count == 0:
                message = f"No documents found matching the filters. Nothing to delete."
                logger.warning(message)
                return {"status": "success", "message": message}

            # 这一行是重复的，且使用了未修正的 filters，请删除！
            # collection.delete(where=filters) 
            
            message = f"Successfully deleted {count} document nodes matching filters: {filters}"
            logger.info(message)
            return {"status": "success", "message": message}

        except Exception as e:
            # 捕获 ChromaDB 抛出的 ValueError
            if isinstance(e, ValueError) and "Expected where value for $and or $or to be a list with at least two where expressions" in str(e):
                message = f"Deletion filter error: {e}. Please check filter format."
                logger.error(message)
                return {"status": "error", "message": message}
            else:
                message = f"An error occurred during deletion: {e}"
                logger.error(message, exc_info=True)
                return {"status": "error", "message": message}

    def get_nodes_by_metadata_filter(self, collection_name: str, filters: dict) -> List[TextNode]:
        """
        根据元数据过滤器，直接从ChromaDB中获取所有匹配的节点。
        这不是相似度搜索，而是精确的数据拉取。
        """
        logger.info(f"Directly fetching nodes from '{collection_name}' with filters: {filters}")
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            
            results = collection.get(
                where=filters,
                include=["metadatas", "documents"]
            )
            
            if not results or not results['ids']:
                logger.info(f"No nodes found for filters: {filters}")
                return []
                
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
        except Exception as e:
            logger.error(f"Error fetching nodes from ChromaDB for collection '{collection_name}' with filters {filters}: {e}", exc_info=True)
            raise

    def update_existing_nodes_metadata(self, collection_name: str, material_id: int, metadata_update_payload: dict):
        """
        根据 material_id 找到所有相关节点，并更新它们的元数据。
        这是一个“合并更新”，而不是完全替换。
        更新策略：将 payload 中的所有列表字段转换为 JSON 字符串。
        """
        logger.info(f"Performing generic metadata update for material_id: {material_id} with payload: {metadata_update_payload}")
        
        if metadata_update_payload is None:
            logger.warning("Update payload is None, skipping update.")
            return

        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            
            nodes_to_update = collection.get(where={"material_id": material_id}, include=["metadatas"])
            if not nodes_to_update or not nodes_to_update['ids']:
                logger.warning(f"No nodes found for material_id {material_id} during generic update. Nothing to do.")
                return

            updated_metadatas = []
            preserved_keys = {"material_id", "author_id", "file_key", "file_name", "file_size"} # 保持这些关键字段
            
            for old_meta in nodes_to_update['metadatas']:
                processed_update_payload = {}
                for key, value in metadata_update_payload.items():
                    if isinstance(value, list):
                        processed_update_payload[key] = json.dumps(value) 
                    elif value is None:
                        processed_update_payload[key] = None 
                    else:
                        processed_update_payload[key] = value

                # 现在，将处理后的 payload 合并到最终的 new_meta 中
                # 以 old_meta 为基础，合并 processed_update_payload
                new_meta_for_db = old_meta.copy() # 以旧元数据为起点
                new_meta_for_db.update(processed_update_payload) # 用处理后的新数据更新旧数据

                # 从旧元数据中“抢救”回那些必须保留的核心字段，确保它们不会被错误覆盖
                for key in preserved_keys:
                    if key in old_meta and key not in processed_update_payload: # 只有当新payload没有明确提供时才保留
                        new_meta_for_db[key] = old_meta[key]
                
                updated_metadatas.append(new_meta_for_db)
            
            collection.update(
                ids=nodes_to_update['ids'],
                metadatas=updated_metadatas
            )
            logger.info(f"Successfully updated metadata for {len(nodes_to_update['ids'])} nodes of material_id {material_id}.")
        except Exception as e:
            logger.error(f"Error during generic metadata update for material_id '{material_id}': {e}", exc_info=True)
            raise # 重新抛出异常

    def add_public_acl_to_material(self, material_id: int, collection_name: str) -> dict:
        """
        为已存在的私有文档添加公共权限（通过创建副本）。
        """
        task_status = "error"
        message = f"An unexpected error occurred while publishing material {material_id}."
        
        try:
            collection = self.chroma_client.get_collection(name=collection_name)

            # --- 增加幂等性检查 ---
            logger.info(f"Checking for existing public nodes for material_id: {material_id}")
            existing_public_nodes = collection.get(
                where={"$and": [{"material_id": material_id}, {"accessible_to": "public"}]},
                limit=1
            )
            if existing_public_nodes and existing_public_nodes['ids']:
                message = f"Material {material_id} has already been published. Operation is idempotent."
                task_status = "duplicate"
                logger.warning(message)
                return {"message": message, "status": task_status}

            logger.info(f"Finding existing nodes for material_id: {material_id} to publish.")
            results = collection.get(
                where={"material_id": material_id},
                include=["metadatas", "documents"]
            )
            
            if not results or not results['ids']:
                message = f"No existing nodes found for material_id {material_id} to publish."
                task_status = "not_found"
                logger.warning(message)
                return {"message": message, "status": task_status}

            public_nodes_to_add = []
            for i in range(len(results['ids'])):
                metadata_copy = results['metadatas'][i].copy() if results['metadatas'][i] else {}
                if metadata_copy.get('accessible_to') == 'public':
                    logger.warning(f"Node {results['ids'][i]} for material_id {material_id} is already public. Skipping.")
                    continue
                
                metadata_copy['accessible_to'] = 'public'
                # 创建新的 LlamaDocument 对象 (LlamaDocument 内部会自动生成 ID，但为了和旧ID区分，可以考虑前缀)
                new_public_node = LlamaDocument(
                    text=results['documents'][i],
                    metadata=metadata_copy,
                    # id_=f"public_{results['ids'][i]}" # 可以考虑给公共副本一个不同的ID
                )
                public_nodes_to_add.append(new_public_node)
            
            if public_nodes_to_add:
                logger.info(f"Adding {len(public_nodes_to_add)} new public nodes for material_id {material_id}.")
                # 调用自身的 add_documents_to_index 方法来执行插入
                self.add_documents_to_index(public_nodes_to_add, collection_name) # <-- 使用通用的 add_documents_to_index
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
    