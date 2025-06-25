import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def prepare_metadata_for_storage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备元数据以便存入 ChromaDB。
    遍历字典，将所有列表类型的值转换为 ChromaDB 兼容的分隔符字符串格式。
    例如: {"accessible_to": ["public", "123"]} -> {"accessible_to": ",public,123,"}
    """
    # 创建一个副本以避免修改原始字典
    storage_ready_metadata = metadata.copy()
    
    for key, value in storage_ready_metadata.items():
        if isinstance(value, list):
            # 将列表转换为带前后逗号的分隔符字符串
            transformed_value = f",{','.join(map(str, value))},"
            storage_ready_metadata[key] = transformed_value
            logger.debug(f"Transformed list in metadata key '{key}' to string: '{transformed_value}'")
            
    return storage_ready_metadata


def reconstruct_metadata_from_storage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    从存储中重建元数据，供 Pydantic 模型或 API 使用。
    遍历字典，将所有可能是分隔符字符串的字段转换回列表。
    """
    # 创建一个副本
    reconstructed_metadata = metadata.copy()
    
    # 定义哪些键可能是需要从字符串还原为列表的
    keys_to_reconstruct = ['accessible_to', 'level_list', 'label_list'] # 您可以根据需要增删

    for key in keys_to_reconstruct:
        value = reconstructed_metadata.get(key)
        # 检查值是否存在且为我们约定的分隔符字符串格式
        if isinstance(value, str) and value.startswith(',') and value.endswith(','):
            # 去掉首尾的逗号
            stripped_value = value.strip(',')
            # 如果中间有内容则切割，否则返回空列表
            reconstructed_metadata[key] = stripped_value.split(',') if stripped_value else []
            logger.debug(f"Reconstructed string in metadata key '{key}' back to list: {reconstructed_metadata[key]}")

    return reconstructed_metadata