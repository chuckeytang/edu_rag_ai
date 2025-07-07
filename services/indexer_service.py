# api/services/indexer_service.py
import logging
import os
import shutil
from typing import Dict, List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_cloud import TextNode
from llama_index.core.llms import LLM

import chromadb
from chromadb.config import Settings

from core.config import settings

logger = logging.getLogger(__name__)

PERSIST_DIR = settings.INDEX_PATH

class IndexerService:
    def __init__(self, 
                 chroma_client: chromadb.PersistentClient,
                 embedding_model: BaseEmbedding):
        
        self.chroma_client = chroma_client
        self.embedding_model = embedding_model
        # 内存缓存，用于按需加载索引对象
        self.indices: Dict[str, VectorStoreIndex] = {} 
        
        logger.info("IndexerService initialized.")

    def _get_or_load_index(self, collection_name: str) -> Optional[VectorStoreIndex]:
        """
        按需加载或获取指定 collection 的 LlamaIndex 索引对象。
        1. 检查内存缓存。
        2. 如果不在缓存，尝试从 ChromaDB 和本地存储加载。
        3. 如果加载失败，则从 ChromaDB VectorStore 重建 Index 对象。
        """
        if collection_name in self.indices:
            logger.info(f"Returning cached index for collection: '{collection_name}'.")
            return self.indices[collection_name]

        try:
            logger.info(f"Checking for collection '{collection_name}' in ChromaDB...")
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
            logger.info(f"Collection '{collection_name}' found/created in ChromaDB.")

        except Exception as e:
            logger.warning(f"Failed to get or create collection '{collection_name}' in ChromaDB: {e}", exc_info=True)
            return None # 无法获取 Collection 则无法创建 Index

        try:
            index = load_index_from_storage(storage_context=storage_context, embed_model=self.embedding_model)
            logger.info(f"Successfully loaded index metadata for collection '{collection_name}' from '{PERSIST_DIR}'.")
        except Exception as e:
            logger.warning(f"Could not load LlamaIndex metadata from '{PERSIST_DIR}' for collection '{collection_name}'. Error: {e}")
            logger.info(f"Reconstructing index object from existing VectorStore for '{collection_name}' (metadata might be missing)...")
            # 如果本地元数据丢失，从现有 VectorStore 重建索引对象
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embedding_model,
                storage_context=storage_context # 即使重建也传入，以便它能保存新的元数据
            )
            logger.info(f"Successfully reconstructed index from VectorStore for '{collection_name}'.")
        
        self.indices[collection_name] = index
        return index

    def add_documents_to_index(self, documents: List[LlamaDocument], collection_name: str) -> VectorStoreIndex:
        """
        将文档添加到指定 collection 的索引中。
        如果索引不存在，则初始化；如果存在，则插入新文档。
        此方法会确保所有文档通过 embedding_model 向量化并入库。
        """
        logger.info(f"Adding {len(documents)} documents to index for collection: '{collection_name}'...")
        
        # 尝试获取现有索引
        index = self._get_or_load_index(collection_name)

        if index is None:
            logger.info(f"Index for '{collection_name}' not found/loadable. Initializing a new one.")
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)

            new_index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                embed_model=self.embedding_model # 确保使用正确的 embedding model
            )
            logger.info(f"New index for '{collection_name}' initialized successfully.")
            index = new_index
        else:
            logger.info(f"Index for '{collection_name}' already exists. Inserting new documents.")
            for idx, doc in enumerate(documents):
                try:
                    index.insert(doc) # LlamaIndex insert 会使用其关联的 embed_model
                    logger.debug(f"✅ Inserted document {idx+1}: {doc.id_} - {doc.text[:80]}...")
                except Exception as e:
                    logger.error(f"Failed to insert document {idx+1} into index: {e}", exc_info=True)
        
        # 持久化索引元数据（如 docstore）
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        logger.info(f"Index for '{collection_name}' updated and persisted.")
        self.indices[collection_name] = index # 更新内存缓存
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
            
            results = collection.get(where=filters, include=[])
            count = len(results.get('ids', []))

            if count == 0:
                message = f"No documents found matching the filters. Nothing to delete."
                logger.warning(message)
                return {"status": "success", "message": message}

            collection.delete(where=filters)
            
            message = f"Successfully deleted {count} document nodes matching filters: {filters}"
            logger.info(message)
            return {"status": "success", "message": message}

        except Exception as e:
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
                new_meta = metadata_update_payload.copy()
                for key in preserved_keys:
                    if key in old_meta:
                        new_meta[key] = old_meta[key]
                updated_metadatas.append(new_meta)
            
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
    