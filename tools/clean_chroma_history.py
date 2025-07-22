# clean_chroma_history.py
import chromadb
import logging
import os
from chromadb.config import Settings

# 导入你的 settings，以便获取 CHROMA_PATH
# 假设你的 settings 是从 core.config 导入的
from core.config import settings 

logger = logging.getLogger(__name__)

CHROMA_PATH = settings.CHROMA_PATH # 确保这个路径正确

# 要清理的 Collection 名称
CHAT_HISTORY_COLLECTION_NAME = "chat_history_collection"

def clean_chat_history_collection():
    logger.info(f"Connecting to ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(is_persistent=True)
    )

    try:
        # 尝试获取 Collection
        collection = client.get_collection(name=CHAT_HISTORY_COLLECTION_NAME)
        logger.info(f"Found collection '{CHAT_HISTORY_COLLECTION_NAME}'. Counting items...")

        count_before_delete = collection.count()
        logger.info(f"Collection '{CHAT_HISTORY_COLLECTION_NAME}' has {count_before_delete} items.")

        if count_before_delete > 0:
            # 删除 Collection 中的所有数据
            # collection.delete(ids=collection.get()['ids']) # 也可以通过获取所有ID来删除

            # 更简单粗暴但有效的方式：直接删除整个 Collection
            logger.info(f"Deleting collection '{CHAT_HISTORY_COLLECTION_NAME}'...")
            client.delete_collection(name=CHAT_HISTORY_COLLECTION_NAME)
            logger.info(f"Collection '{CHAT_HISTORY_COLLECTION_NAME}' deleted successfully.")
        else:
            logger.info(f"Collection '{CHAT_HISTORY_COLLECTION_NAME}' is empty. No deletion needed.")

    except Exception as e:
        logger.warning(f"Collection '{CHAT_HISTORY_COLLECTION_NAME}' not found or error during deletion: {e}. Assuming it doesn't exist or already cleaned.")
        logger.info(f"Proceeding to ensure the collection can be re-created with correct dimension.")
        # 如果 Collection 不存在，get_collection 会抛异常，这是正常的，可以忽略

    # 重新创建 Collection (只是为了确保它能以正确的维度被创建，如果之前不存在)
    # 实际创建维度是在你通过 LlamaIndex 的 `add_documents_to_index` 第一次插入数据时。
    # 这里的 get_or_create 只是一个占位符，并不强制维度。
    try:
        collection = client.get_or_create_collection(name=CHAT_HISTORY_COLLECTION_NAME)
        logger.info(f"Collection '{CHAT_HISTORY_COLLECTION_NAME}' is ready for new data.")
        logger.info("Remember to restart your main FastAPI application to ensure it uses the newly created/reset collection.")
    except Exception as e:
        logger.error(f"Failed to get_or_create_collection '{CHAT_HISTORY_COLLECTION_NAME}' after deletion attempt: {e}", exc_info=True)


if __name__ == "__main__":
    # 为了避免导入你整个 FastAPI 应用，只导入必要的配置
    # 如果 core.config 依赖 FastAPI 应用的其他部分，你需要调整这里
    # 确保 core.config.settings 能够独立地被加载
    try:
        settings.load_settings() # 如果你的settings有加载方法
    except Exception:
        pass # 否则假设它会自动加载

    clean_chat_history_collection()