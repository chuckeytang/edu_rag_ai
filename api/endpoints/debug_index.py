import logging
import json

from pydantic import BaseModel

from api.dependencies import get_indexer_service, get_query_service
from services.indexer_service import IndexerService

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional
from models.schemas import QueryRequest
from fastapi import APIRouter, Depends, HTTPException, Path, Query
import numpy as np
from services.query_service import QueryService  

router = APIRouter()

# 定义一个用于返回调试信息的Schema
class DebugQueryResponse(BaseModel):
    query_text: str
    final_llm_prompt: str
    original_retrieved_nodes: List[Dict[str, Any]]
    final_retrieved_nodes: List[Dict[str, Any]]
    final_response_content: str
    citations: List[Dict[str, Any]]

router = APIRouter()
@router.get("/indexed", summary="列出索引中的节点（chunk）")
def list_indexed(
    collection_name: str = Query(..., description="要查询的Collection名称"),
    limit: int = Query(20, ge=1, le=1000, description="最多返回多少个节点"),
    file_name: Optional[str] = Query(
        None, description="按 metadata.file_name 关键字模糊过滤（不区分大小写）"
    ),
    chroma_id: Optional[str] = Query(
        None, description="要查询的特定节点在 ChromaDB 中的内部ID"
    ),
    title: Optional[str] = Query(
        None, description="按 metadata.title 关键字模糊过滤（不区分大小写）"
    ),
    query_service: Any = Depends(get_query_service),
    indexer_service: IndexerService = Depends(get_indexer_service)
) -> List[Dict[str, Any]]:
    """
    从指定的ChromaDB Collection中直接“窥视”数据，用于调试。
    支持按Collection名称、返回数量、文件名/标题模糊过滤，以及特定节点ID精确查询。
    """
    try:
        col = query_service.chroma_client.get_collection(name=collection_name)
    except ValueError:
        logger.warning(f"Debug endpoint: Collection '{collection_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found in ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get collection '{collection_name}': {e}")

    items = []
    
    if title or file_name:
        filters = {}
        
        # 1. 打印调用前的状态
        logger.info(f"Debug: Starting filtered search for collection '{collection_name}' with title='{title}' and file_name='{file_name}'.")

        nodes = indexer_service.get_nodes_by_metadata_filter(collection_name, filters)
        
        # 2. 打印 get_nodes_by_metadata_filter 返回的节点数量和第一个节点的详细信息
        logger.info(f"Debug: get_nodes_by_metadata_filter returned {len(nodes)} nodes.")
        if nodes:
            first_node = nodes[0]
            # 注意：使用vars()打印对象的__dict__，能清楚地看到所有属性
            logger.info(f"Debug: First node details: {vars(first_node)}")
            # 专门检查 extra_info
            logger.info(f"Debug: First node's extra_info: {first_node.extra_info}")
            if 'title' in first_node.extra_info:
                logger.info(f"Debug: 'title' found in extra_info: {first_node.extra_info['title']}")
            else:
                logger.warning("Debug: 'title' not found in first node's extra_info!")

        filtered_nodes = []
        if title:
            # 3. 打印过滤前的状态
            logger.info("Debug: Applying title filter...")
            filtered_by_title = [
                node for node in nodes 
                if node.extra_info is not None and title.lower() in (node.extra_info.get("title", "") or "").lower()
            ]
            logger.info(f"Debug: Found {len(filtered_by_title)} nodes matching title '{title}'.")
            filtered_nodes.extend(filtered_by_title)
        
        if file_name:
            # 4. 打印过滤前的状态
            logger.info("Debug: Applying file_name filter...")
            filtered_by_filename = [
                node for node in nodes 
                if node.extra_info is not None and file_name.lower() in (node.extra_info.get("file_name", "") or "").lower()
            ]
            logger.info(f"Debug: Found {len(filtered_by_filename)} nodes matching file_name '{file_name}'.")
            filtered_nodes.extend(filtered_by_filename)

        unique_nodes = {node.id: node for node in filtered_nodes}.values()
        
        for node in list(unique_nodes)[:limit]:
            effective_node_id = node.extra_info.get("doc_id") or node.extra_info.get("ref_doc_id") or node.id
            
            items.append({
                # 修复点：将 node.id_ 改为 node.id
                "node_id": effective_node_id,
                "chroma_id": node.id,
                "metadata": node.extra_info,
                "file_name": node.extra_info.get("file_name"),
                "page_label": node.extra_info.get("page_label"),
                "text": node.text
            })
        
        return items

    # else 分支保持不变
    else:
        logger.info(f"Peeking into collection '{collection_name}' with limit {limit}...")
        peek_result = col.peek(limit=limit * 5)
        
        ids = peek_result.get("ids") or []
        docs = peek_result.get("documents") or []
        metas = peek_result.get("metadatas") or []

        for _id, doc, meta in zip(ids, docs, metas):
            node_content_meta = {}
            if '_node_content' in meta and isinstance(meta['_node_content'], str):
                try:
                    node_content_data = json.loads(meta['_node_content'])
                    node_content_meta = node_content_data.get('metadata', {})
                except json.JSONDecodeError:
                    pass
            
            current_file_name = (meta.get("file_name") or node_content_meta.get("file_name") or "").lower()
            current_title = (meta.get("title") or node_content_meta.get("title") or "").lower()

            if file_name and file_name.lower() not in current_file_name:
                continue
            
            if title and title.lower() not in current_title:
                continue
                
            effective_node_id = meta.get("doc_id") or meta.get("ref_doc_id") or _id

            items.append({
                "node_id": effective_node_id,
                "chroma_id": _id,
                "metadata": meta,
                "file_name": current_file_name,
                "page_label": meta.get("page_label") or node_content_meta.get("page_label"),
                "text": doc
            })
            
            if len(items) >= limit:
                break
                
    return items

@router.get("/indexed/{chroma_id}", summary="查看单个节点的完整内容与元数据")
def get_node(chroma_id: str,
            collection_name: str = Query(..., description="该节点所在的Collection名称"),
            query_service: QueryService = Depends(get_query_service)):
    """
    从指定的ChromaDB Collection中，按ID获取单个节点的详细信息。
    """
    try:
        col = query_service.chroma_client.get_collection(name=collection_name)
    except ValueError:
        logger.warning(f"Debug endpoint: Collection '{collection_name}' not found for node id '{chroma_id}'.")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found in ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get collection '{collection_name}': {e}")

    res = col.get(ids=[chroma_id], include=["documents", "metadatas"])

    if not res.get("ids"):
        raise HTTPException(status_code=404, detail=f"Node with chroma_id '{chroma_id}' not found in collection '{collection_name}'.")

    return {
        "chroma_id": res["ids"][0],
        "metadata":  res["metadatas"][0],
        "text":      res["documents"][0],
    }

@router.post("/retrieve-with-filters", summary="[DEBUG] 测试带过滤的节点召回")
def debug_retrieve_with_filters(request: QueryRequest, 
                            query_service: QueryService = Depends(get_query_service),
                            indexer_service: IndexerService = Depends(get_indexer_service)):
    """
    一个用于调试的端点。
    它只执行召回步骤，并返回召回的节点列表及其元数据，
    帮助您确认Filter是否按预期工作。
    """
    try:
        # 创建一个可变的 filters 字典，以便合并
        combined_filters = request.filters.copy() if request.filters else {}

        if request.target_file_ids:
            try:
                material_ids_int = [int(mid) for mid in request.target_file_ids]
                # 将 target_file_ids 合并到 material_id 过滤器中
                if "material_id" in combined_filters and isinstance(combined_filters["material_id"], dict) and "$in" in combined_filters["material_id"]:
                    current_material_ids = combined_filters["material_id"]["$in"]
                    combined_filters["material_id"]["$in"] = list(set(current_material_ids + material_ids_int))
                else:
                    combined_filters["material_id"] = {"$in": material_ids_int}
            except ValueError:
                logger.warning("Debug endpoint: Invalid material_id in target_file_ids. Ignoring file filter.")

        retrieved_nodes = query_service.retrieve_with_filters(
            question=request.question,
            collection_name=request.collection_name,
            filters=combined_filters, 
            similarity_top_k=request.similarity_top_k
        )

        results = []
        for node_with_score in retrieved_nodes:
            current_node_metadata = node_with_score.node.metadata 

            try:
                col = indexer_service.chroma_client.get_collection(name=request.collection_name)
                latest_meta_from_db = col.get(ids=[node_with_score.node.node_id], include=["metadatas"])['metadatas'][0]
                current_node_metadata = latest_meta_from_db
            except Exception as debug_e:
                logger.error(f"DEBUG: Failed to get latest metadata from DB for {node_with_score.node.node_id}: {debug_e}")

            score_to_append = float(node_with_score.score) # <--- 将 score 强制转换为 Python float
            
            # 对 metadata 进行深度拷贝，并转换其中的 NumPy 类型
            cleaned_metadata = {}
            for key, value in current_node_metadata.items():
                if isinstance(value, np.ndarray): # 如果是 NumPy 数组
                    cleaned_metadata[key] = value.tolist() # 转换为 Python 列表
                elif isinstance(value, (np.float32, np.float64)): # <--- 移除 np.float
                    cleaned_metadata[key] = float(value) 
                elif isinstance(value, (np.int32, np.int64)): 
                    cleaned_metadata[key] = int(value) 
                else:
                    cleaned_metadata[key] = value 
                    
            results.append({
                "score": score_to_append,
                "node_id": node_with_score.node.node_id,
                "text_snippet": node_with_score.get_text()[:300] + "...",
                "metadata": cleaned_metadata
            })

        return results

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Debug retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/debug-rag-flow", summary="[DEBUG] 执行并展示完整的RAG流程（召回、重排、回复）")
async def debug_rag_flow(
    request: QueryRequest, 
    query_service: QueryService = Depends(get_query_service),
    indexer_service: IndexerService = Depends(get_indexer_service)
) -> DebugQueryResponse:
    """
    一个用于调试的端点，它会执行完整的 RAG 流程，并返回每个关键步骤的详细结果，包括：
    - 初始召回的节点列表
    - 重排后的最终节点列表
    - 最终生成给 LLM 的完整 Prompt
    - LLM 的最终回复内容
    - 句子级的引用信息
    """
    logger.info(f"Starting debug RAG flow for collection '{request.collection_name}' with query: '{request.question}'")
    
    rag_config = query_service.rag_config
    
    try:
        # 1. 加载或获取索引
        rag_index = indexer_service._get_or_load_index(request.collection_name)
        if not rag_index:
            raise HTTPException(status_code=404, detail=f"RAG Collection '{request.collection_name}' does not exist or could not be loaded.")
            
        # 2. 构建 ChromaDB 的 where 子句
        combined_rag_filters = request.filters if request.filters else {}
        if request.target_file_ids and len(request.target_file_ids) > 0:
            try:
                material_ids_int = [int(mid) for mid in request.target_file_ids]
                if "material_id" in combined_rag_filters and isinstance(combined_rag_filters["material_id"], dict) and "$in" in combined_rag_filters["material_id"]:
                    current_material_ids = combined_rag_filters["material_id"]["$in"]
                    combined_rag_filters["material_id"]["$in"] = list(set(current_material_ids + material_ids_int))
                else:
                    combined_rag_filters["material_id"] = {"$in": material_ids_int}
            except ValueError:
                logger.warning("Debug endpoint: Invalid material_id in target_file_ids. Ignoring file filter.")

        chroma_where_clause = query_service._build_chroma_where_clause(combined_rag_filters)
        
        # 3. 执行召回（Retrieval）
        retriever = rag_index.as_retriever(
            vector_store_kwargs={"where": chroma_where_clause} if chroma_where_clause else {},
            similarity_top_k=rag_config.retrieval_top_k * rag_config.initial_retrieval_multiplier 
        )
        
        original_retrieved_nodes_with_score = await retriever.aretrieve(request.question)
        original_retrieved_nodes = [
            {
                "score": float(n.score), 
                "node_id": n.node.node_id, 
                "text_snippet": n.get_text()[:300] + "...",
                "metadata": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in n.node.metadata.items()}
            } 
            for n in original_retrieved_nodes_with_score
        ]
        logger.info(f"Initial retrieval found {len(original_retrieved_nodes)} nodes.")
        
        # 4. 执行重排（Reranking）
        query_bundle_for_rerank = QueryBundle(query_str=request.question)
        final_retrieved_nodes_with_score = []

        if rag_config.use_reranker and query_service.local_reranker:
            final_retrieved_nodes_with_score = query_service.local_reranker.postprocess_nodes(
                original_retrieved_nodes_with_score,
                query_bundle=query_bundle_for_rerank
            )
        else:
            final_retrieved_nodes_with_score = original_retrieved_nodes_with_score[:rag_config.post_rerank_top_n]

        final_retrieved_nodes = [
            {
                "score": float(n.score), 
                "node_id": n.node.node_id, 
                "text_snippet": n.get_text()[:300] + "...",
                "metadata": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in n.node.metadata.items()}
            } 
            for n in final_retrieved_nodes_with_score
        ]
        logger.info(f"After reranking, {len(final_retrieved_nodes)} nodes were selected.")

        # 5. 构建 LLM 的 Prompt
        context_parts = []
        for n in final_retrieved_nodes_with_score:
            doc_type = n.node.metadata.get("document_type", "Unknown")
            page_label = n.node.metadata.get("page_label", "N/A")
            file_name = n.node.metadata.get("file_name", "N/A")
            context_parts.append(
                f"--- Document Content (Type: {doc_type}, Page: {page_label}, File: {file_name}) ---\n"
                f"{n.get_content()}\n"
                f"--------------------------------------------------------------------------------"
            )
        context_str_for_llm = "\n\n".join(context_parts)
        
        final_prompt_for_llm = query_service.qa_prompt.format(
            chat_history_context="", # 调试接口暂不考虑聊天历史
            context_str=context_str_for_llm, 
            query_str=request.question
        )
        logger.debug(f"Final Prompt for LLM: \n{final_prompt_for_llm[:500]}...")
        
        # 6. 调用 LLM 获取最终回复
        # 构造一个非流式的 QueryEngine
        query_engine = rag_index.as_query_engine(
            llm=query_service.llm,
            retriever=query_service._indexer_service.get_query_engine(
                collection_name=request.collection_name, 
                llm=query_service.llm, 
                rag_config=rag_config
            ).retriever # 复用 QueryService 中的 Retriever
        )

        response_obj = await query_engine.aquery(final_prompt_for_llm)
        final_response_content = response_obj.response if hasattr(response_obj, 'response') else ""
        logger.info("Successfully got non-streaming LLM response.")

        # 7. 提取引用信息 (与 query_service 中的逻辑类似)
        citations = []
        if query_service.embedding_model and final_response_content:
            from llama_index.core.schema import TextNode as LlamaTextNode
            from services.query_service import _tokenizer
            from core.rag_config import RagConfig # 确保导入了 RagConfig
            from services.query_service import SentenceTransformerRerank

            # 这里我们重用query_service.rag_query_with_context中的引用逻辑，但需要一些调整
            # 简化为直接在 debug-flow 中实现
            import re
            response_sentences = re.split(r'(?<=[.!?。！？])\s+', final_response_content)
            
            referenced_chunk_texts = [n.node.text for n in final_retrieved_nodes_with_score]
            referenced_chunk_ids = [n.node.node_id for n in final_retrieved_nodes_with_score]
            
            if response_sentences and referenced_chunk_texts:
                all_text_to_embed = response_sentences + referenced_chunk_texts
                all_embeddings = await query_service.embedding_model.aget_text_embedding_batch(all_text_to_embed, show_progress=False)
                
                response_sentence_embeddings = all_embeddings[:len(response_sentences)]
                referenced_chunk_embeddings = all_embeddings[len(response_sentences):]

                import numpy as np
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

                    if max_similarity > rag_config.citation_similarity_threshold:
                        source_node_with_score = next((n for n in final_retrieved_nodes_with_score if n.node.node_id == best_match_id), None)
                        if source_node_with_score:
                            source_meta = source_node_with_score.node.metadata
                            citation_info = {
                                "sentence": sentence,
                                "referenced_chunk_id": best_match_id,
                                "referenced_chunk_text": source_node_with_score.node.text,
                                "document_id": source_meta.get('document_id'),
                                "material_id": source_meta.get('material_id'),
                                "file_name": source_meta.get('file_name'),
                                "page_label": source_meta.get('page_label')
                            }
                            citations.append(citation_info)
        
        # 8. 封装并返回所有调试信息
        return DebugQueryResponse(
            query_text=request.question,
            final_llm_prompt=final_prompt_for_llm,
            original_retrieved_nodes=original_retrieved_nodes,
            final_retrieved_nodes=final_retrieved_nodes,
            final_response_content=final_response_content,
            citations=citations
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run debug RAG flow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during debug RAG flow: {str(e)}")

@router.post("/delete-material/{material_id}", summary="根据 material_id 删除所有相关文档块（chunk）")
def delete_chunks_by_material_id(
    material_id: int = Path(..., description="要删除的文档的 material_id"),
    collection_name: str = Query(..., description="目标Collection的名称"),
    indexer_service: IndexerService = Depends(get_indexer_service) # 依赖注入 IndexerService
) -> Dict[str, Any]:
    """
    根据给定的 material_id，从指定 Collection 中删除所有相关的文档块（chunk）。
    """
    logger.info(f"Received request to delete chunks for material_id: {material_id} in collection: '{collection_name}'")
    
    try:
        # IndexerService 中的 delete_nodes_by_metadata 方法已经处理了过滤逻辑
        filters = {"material_id": material_id} # material_id 存储为整数，直接传递整数
        
        result = indexer_service.delete_nodes_by_metadata(collection_name=collection_name, filters=filters)
        
        # delete_nodes_by_metadata 已经返回了清晰的状态和信息
        if result.get("status") == "success":
            logger.info(f"Successfully processed delete request for material_id {material_id}: {result.get('message')}")
            return {"status": "success", "message": result.get("message")}
        else:
            logger.error(f"Failed to delete chunks for material_id {material_id}: {result.get('message')}")
            raise HTTPException(status_code=500, detail=result.get("message"))
            
    except HTTPException: # 如果 delete_nodes_by_metadata 内部抛出 HTTPException，直接向上抛
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during deletion for material_id {material_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
