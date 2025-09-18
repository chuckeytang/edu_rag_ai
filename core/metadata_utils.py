# core/metadata_utils.py
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def prepare_metadata_for_storage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备元数据以便存入存储。
    因为火山引擎知识库的 user_data 字段是一个 JSON 字符串，
    我们只需确保元数据是可序列化的，而不需要特殊的字符串转换。
    """
    # 在这里，我们不再进行任何特殊的字符串转换，因为 json.dumps() 会处理列表。
    # 为了保持函数的完整性，我们只返回一个副本，并打印日志。
    storage_ready_metadata = metadata.copy()
    logger.debug("Metadata is prepared for storage. No special conversion needed for JSON format.")
    return storage_ready_metadata

# reconstruct_metadata_from_storage 方法已被移除，因为它仅用于 ChromaDB