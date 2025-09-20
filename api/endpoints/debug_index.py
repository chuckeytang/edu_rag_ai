import logging
import json
import time

from pydantic import BaseModel
from llama_index.core.schema import QueryBundle
from llama_index.core import PromptTemplate
from api.dependencies import get_indexer_service, get_query_service, get_retrieval_service
from services.indexer_service import IndexerService
from llama_index.core.query_engine import RetrieverQueryEngine

from services.retrieval_service import RetrievalService 
logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional
from models.schemas import ChatQueryRequest, QueryRequest
from fastapi import APIRouter, Depends, HTTPException, Path, Query
import numpy as np
from services.query_service import CustomRetriever, QueryService  

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
    # 在函数入口处直接打印原始参数值
    logger.info(f"Received parameters: collection_name='{collection_name}', title='{title}', file_name='{file_name}', limit='{limit}'")

    try:
        col = query_service.chroma_client.get_collection(name=collection_name)
    except ValueError:
        logger.warning(f"Debug endpoint: Collection '{collection_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found in ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get collection '{collection_name}': {e}")

    items = []
    
    # 在条件判断前，对 title 和 file_name 进行显式检查
    # 如果参数为 None 或空字符串，则将其重置为 None
    # 这样可以确保 if title or file_name: 的判断在所有环境中都一致
    effective_title = title.strip() if title else None
    effective_file_name = file_name.strip() if file_name else None

    # 然后使用修正后的变量进行判断
    if effective_title or effective_file_name:
        filters = {}
        
        logger.info(f"Debug: Starting filtered search for collection '{collection_name}' with effective_title='{effective_title}' and effective_file_name='{effective_file_name}'.")

        nodes = indexer_service.get_nodes_by_metadata_filter(collection_name, filters)
        
        logger.info(f"Debug: get_nodes_by_metadata_filter returned {len(nodes)} nodes.")
        if nodes:
            first_node = nodes[0]
            logger.info(f"Debug: First node details: {vars(first_node)}")
            logger.info(f"Debug: First node's extra_info: {first_node.extra_info}")
            if 'title' in first_node.extra_info:
                logger.info(f"Debug: 'title' found in extra_info: {first_node.extra_info['title']}")
            else:
                logger.warning("Debug: 'title' not found in first node's extra_info!")

        filtered_nodes = []
        if effective_title:
            logger.info("Debug: Applying title filter...")
            filtered_by_title = [
                node for node in nodes 
                if node.extra_info is not None and effective_title.lower() in (node.extra_info.get("title", "") or "").lower()
            ]
            logger.info(f"Debug: Found {len(filtered_by_title)} nodes matching title '{effective_title}'.")
            filtered_nodes.extend(filtered_by_title)
        
        if effective_file_name:
            logger.info("Debug: Applying file_name filter...")
            filtered_by_filename = [
                node for node in nodes 
                if node.extra_info is not None and effective_file_name.lower() in (node.extra_info.get("file_name", "") or "").lower()
            ]
            logger.info(f"Debug: Found {len(filtered_by_filename)} nodes matching file_name '{effective_file_name}'.")
            filtered_nodes.extend(filtered_by_filename)

        unique_nodes = {node.id: node for node in filtered_nodes}.values()
        
        for node in list(unique_nodes)[:limit]:
            effective_node_id = node.extra_info.get("doc_id") or node.extra_info.get("ref_doc_id") or node.id
            
            items.append({
                "node_id": effective_node_id,
                "chroma_id": node.id,
                "metadata": node.extra_info,
                "file_name": node.extra_info.get("file_name"),
                "page_label": node.extra_info.get("page_label"),
                "text": node.text
            })
        
        return items

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

            if effective_file_name and effective_file_name.lower() not in current_file_name:
                continue
            
            if effective_title and effective_title.lower() not in current_title:
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
async def debug_retrieve_with_filters(request: QueryRequest, 
                            retrieval_service: RetrievalService = Depends(get_retrieval_service)):
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

        retrieved_nodes = await retrieval_service.retrieve_documents(
            query_text=request.question,
            collection_name=request.collection_name,
            filters=combined_filters, 
            top_k=request.similarity_top_k,
            use_reranker=True # 调试时通常需要看重排效果
        )

        results = []
        for node_with_score in retrieved_nodes:
            current_node_metadata = node_with_score.node.metadata 
            score_to_append = float(node_with_score.score) 
            
            cleaned_metadata = {}
            for key, value in current_node_metadata.items():
                if isinstance(value, np.ndarray):
                    cleaned_metadata[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
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

@router.delete("/delete-collection", summary="删除指定的整个Collection")
def delete_entire_collection(
    collection_name: str = Query(..., description="要删除的Collection名称"),
    query_service: Any = Depends(get_query_service)
) -> Dict[str, Any]:
    """
    永久删除 ChromaDB 中指定的整个 Collection 及其所有数据。
    这个操作是不可逆的，请谨慎使用！
    """
    try:
        # 获取 ChromaDB 客户端
        client = query_service.chroma_client
        
        # 检查 Collection 是否存在
        # ChromaDB 的 get_collection 如果不存在会抛出 ValueError
        try:
            client.get_collection(name=collection_name)
        except ValueError:
            logger.warning(f"Collection '{collection_name}' not found. No action taken.")
            return {"status": "success", "message": f"Collection '{collection_name}' not found, no deletion necessary."}
        
        # 删除 Collection
        logger.info(f"Attempting to delete collection: '{collection_name}'")
        client.delete_collection(name=collection_name)
        
        message = f"Successfully deleted collection '{collection_name}'."
        logger.info(message)
        return {"status": "success", "message": message}
    
    except Exception as e:
        logger.error(f"Failed to delete collection '{collection_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete collection '{collection_name}': {str(e)}")
    
@router.post("/debug-rag-flow", summary="[DEBUG] 执行并展示完整的RAG流程（召回、重排、回复）")
async def debug_rag_flow(
    request: ChatQueryRequest, 
    query_service: QueryService = Depends(get_query_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
) -> DebugQueryResponse:
    import numpy as np 
    """
    一个用于调试的端点，它会执行完整的 RAG 流程，并返回每个关键步骤的详细结果，包括：
    - 初始召回的节点列表
    - 重排后的最终节点列表
    - 最终生成给 LLM 的完整 Prompt
    - LLM 的最终回复内容
    - 句子级的引用信息
    """
    logger.info(f"Starting debug RAG flow for collection '{request.collection_name}' with query: '{request.question}'")
    
    rag_config = request.rag_config
    if not rag_config:
        logger.warning("No rag_config found in request, using default from query_service.")
        rag_config = query_service.rag_config
        
    try:
        logger.info(f"rag_config.use_reranker: {rag_config.use_reranker}.")
        filter_expressions = []
        
        # 1. 处理现有的 filters
        if request.filters:
            filter_expressions.append(request.filters)
        
        # 2. 处理 target_file_ids
        if request.target_file_ids:
            logger.info(f"Target file IDs specified: {request.target_file_ids}. Only retrieving from these files.")

            # 将所有 ID 转换为整数类型
            try:
                integer_material_ids = [int(id_str) for id_str in request.target_file_ids]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid target_file_ids. All IDs must be integers.")

            if len(integer_material_ids) == 1:
                # 只有一个文件ID，直接用 $eq 表达式
                file_filter = {"material_id": {"$eq": integer_material_ids[0]}}
                filter_expressions.append(file_filter)
            else:
                # 多个文件ID，使用 $in 表达式 (更简洁)
                # 注: ChromaDB 的 $in 表达式接受列表，而不需要 $or 
                file_filter = {"material_id": {"$in": integer_material_ids}}
                filter_expressions.append(file_filter)

        # 3. 根据表达式数量构建最终的过滤器
        final_filters = None
        if len(filter_expressions) == 1:
            # 只有一个表达式，直接使用它
            final_filters = filter_expressions[0]
        elif len(filter_expressions) > 1:
            # 多个表达式，用 $and 组合
            final_filters = {"$and": filter_expressions}
        
        logger.info(f"Final filters to be used: {final_filters}")
        # --- 结束新增逻辑 ---

        # 直接调用 RetrievalService 的通用方法，获取最终重排后的节点
        final_retrieved_nodes_with_score = await retrieval_service.retrieve_documents(
            query_text=request.question,
            collection_name=request.collection_name,
            filters=final_filters,
            top_k=request.similarity_top_k,
            use_reranker=rag_config.use_reranker
        )
        
        # 为了获取初始召回的节点列表（用于调试），我们需要额外调用 RetrievalService 的底层方法
        # 这有点冗余，但对于调试端点是合理的
        initial_retrieval_top_k = request.similarity_top_k * rag_config.initial_retrieval_multiplier
        original_retrieved_nodes_with_score = await retrieval_service.retrieve_documents(
            query_text=request.question,
            collection_name=request.collection_name,
            filters=final_filters,
            top_k=initial_retrieval_top_k,
            rag_config=rag_config,
            use_reranker=False
        )

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
            chat_history_context="",
            context_str=context_str_for_llm, 
            query_str=request.question
        )
        logger.info(f"Final Prompt for LLM: \n{final_prompt_for_llm[:500]}...")

        # 6. 调用 LLM 获取最终回复
        llm_start_time = time.time()
        query_engine = RetrieverQueryEngine.from_args(
            llm=query_service.llm,
            retriever=CustomRetriever(final_retrieved_nodes_with_score),
            streaming=False,
            text_qa_template=PromptTemplate(final_prompt_for_llm) 
        )
        response_obj = await query_engine.aquery(request.question)
        llm_end_time = time.time()
        final_response_content = response_obj.response if hasattr(response_obj, 'response') else ""
        logger.info(f"Successfully got non-streaming LLM response. LLM inference took {llm_end_time - llm_start_time:.2f} seconds.")

        # 7. 提取引用信息
        citations = []
        if query_service.embedding_model and final_response_content:
            import re
            response_sentences = re.split(r'(?<=[.!?。！？])\s+', final_response_content)
            logger.info(f"LLM response split into {len(response_sentences)} sentences.")
            
            referenced_chunk_texts = [n.node.text for n in final_retrieved_nodes_with_score]
            referenced_chunk_ids = [n.node.node_id for n in final_retrieved_nodes_with_score]
            
            if response_sentences and referenced_chunk_texts:
                logger.info("Embedding LLM sentences and referenced chunks.")
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
                        logger.debug(f"Sentence '{sentence[:30]}...' matched with chunk ID {best_match_id} (Similarity: {max_similarity:.2f}).")
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
            else:
                logger.warning("No sentences or referenced chunks available for citation matching.")
        else:
            logger.warning("Skipping citation extraction due to missing embedding model or empty response content.")
        
        logger.info(f"Citation extraction complete. Found {len(citations)} citations.")
        
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

@router.post("/delete-by-filter", summary="根据 material_id 或文档标题删除相关文档块")
def delete_chunks_by_filter(
    collection_name: str = Query(..., description="目标Collection的名称"),
    material_id: Optional[int] = Query(None, description="要删除的文档的 material_id"),
    keyword: Optional[str] = Query(None, description="按文档标题进行模糊匹配的关键字"),
    doc_type: Optional[str] = Query(None, description="按文档类型进行精确匹配的关键字，例如 'PaperCut'"),
    indexer_service: IndexerService = Depends(get_indexer_service) # 依赖注入 IndexerService
) -> Dict[str, Any]:
    """
    根据给定的过滤器（material_id, keyword, 或 doc_type），从指定 Collection 中删除所有相关的文档块。
    如果提供了多个过滤器，它们会以 AND 逻辑组合。
    """
    if material_id is None and keyword is None and doc_type is None:
        raise HTTPException(status_code=400, detail="必须提供 material_id 或 keyword 中的一个。")

    logger.info(f"Received request to delete chunks for material_id: {material_id} in collection: '{collection_name}'")
    
    filters = {}
    if material_id:
        # 如果提供了 material_id，则进行精确删除
        filters = {"material_id": material_id}
        logger.info(f"Deleting by material_id: {material_id}")
    if doc_type:
        filters["type"] = doc_type
        logger.info(f"Deleting by doc_type: {doc_type}")
    elif keyword:
        
        # 获取所有节点，然后在内存中过滤出匹配关键字的 material_id
        all_nodes = indexer_service.get_nodes_by_metadata_filter(collection_name, {})
        
        matching_material_ids = set()
        for node in all_nodes:
            # 确保 extra_info 和 title 存在
            if node.extra_info and 'title' in node.extra_info:
                node_title = node.extra_info.get('title', '').lower()
                if keyword.lower() in node_title:
                    matching_material_ids.add(node.extra_info.get('material_id'))
        
        matching_material_ids.discard(None) # 移除可能存在的 None 值
        
        if not matching_material_ids:
            message = f"No documents found matching keyword '{keyword}'. Nothing to delete."
            logger.warning(message)
            return {"status": "success", "message": message}

        # 构建 $in 过滤器
        filters = {"material_id": {"$in": list(matching_material_ids)}}
        logger.info(f"Found {len(matching_material_ids)} material_ids matching keyword '{keyword}': {list(matching_material_ids)}")
    
    try:
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

@router.get("/count-chunks", summary="查询指定Collection中的文档块总数")
def count_chunks_in_collection(
    collection_name: str = Query(..., description="要查询的Collection名称"),
    query_service: Any = Depends(get_query_service)
) -> Dict[str, Any]:
    """
    获取指定 ChromaDB Collection 中索引的文档块（chunk）总数。
    """
    try:
        col = query_service.chroma_client.get_collection(name=collection_name)
        count = col.count()
        logger.info(f"Collection '{collection_name}' has {count} documents.")
        return {"collection_name": collection_name, "count": count, "status": "success"}
    except ValueError:
        logger.warning(f"Collection '{collection_name}' not found. Returning count 0.")
        return {"collection_name": collection_name, "count": 0, "status": "not_found"}
    except Exception as e:
        logger.error(f"Failed to count chunks in collection '{collection_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")