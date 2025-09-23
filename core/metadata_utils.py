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
    storage_ready_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (list, dict)):
            # 序列化列表或字典为JSON字符串
            try:
                storage_ready_metadata[key] = json.dumps(value, ensure_ascii=False)
            except TypeError as e:
                logger.error(f"Failed to serialize metadata key '{key}': {e}")
                storage_ready_metadata[key] = str(value) # 降级为字符串
        else:
            storage_ready_metadata[key] = value

    logger.debug("Metadata is prepared for storage.")
    return storage_ready_metadata