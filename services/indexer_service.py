# api/services/indexer_service.py
import logging
import os
import shutil
from typing import Dict, List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import BaseEmbedding # 导入 LlamaIndex 的 BaseEmbedding 类型

import chromadb
from chromadb.config import Settings # 导入 ChromaDB 的 Settings，用于 PersistentClient

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
            # 注意: get_or_create_collection 可以在这里使用，因为它不会改变已存在 collection 的维度
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            
            # 在这里检查 collection 的维度是否与 embedding_model 匹配会更稳健
            # 但 LlamaIndex 和 ChromaDB 的内部校验通常足够

            vector_store = ChromaVectorStore(chroma_collection=collection)
            # persist_dir 用于 LlamaIndex 自身的文档存储（docstore, graph store等）
            # 它与 ChromaDB 的向量存储是分开的
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
            logger.info(f"Collection '{collection_name}' found/created in ChromaDB.")

        except Exception as e:
            logger.warning(f"Failed to get or create collection '{collection_name}' in ChromaDB: {e}", exc_info=True)
            return None # 无法获取 Collection 则无法创建 Index

        # 尝试从本地持久化文件加载 LlamaIndex 的元数据
        try:
            # load_index_from_storage 会尝试加载 docstore, graph store 等
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
            # 如果是新索引，LlamaIndex 会在 from_documents 中创建 collection
            # 并在内部进行 embedding
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

# 不在这里实例化，由 dependencies.py 统一管理