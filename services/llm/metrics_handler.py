import logging
import time
from typing import Any, Dict, Optional
from llama_index.core.callbacks import CBEventType, EventPayload, BaseCallbackHandler

logger = logging.getLogger(__name__)

class TimingCallbackHandler(BaseCallbackHandler):
    """一个简单的回调处理器，用于记录特定事件的耗时。"""
    
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.start_times: Dict[str, float] = {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        """事件开始时记录时间。"""
        self.start_times[event_id] = time.time()
        logger.debug(f"Event '{event_type.value}' started with ID '{event_id}'.")
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """事件结束时计算并记录耗时。"""
        if event_id in self.start_times:
            duration = time.time() - self.start_times[event_id]
            logger.info(f"Callback Event '{event_type.value}' (ID: '{event_id}') finished in {duration:.4f} seconds.")
            
            if event_type == CBEventType.RETRIEVE:
                query_str = payload.get(EventPayload.QUERY_STR)
                if query_str:
                    logger.info(f"    -> Retrieve event for query: '{query_str[:50]}...'")
            elif event_type == CBEventType.EMBEDDING:
                # 检查是否存在 DOCUMENTS 或 CHUNKS 来获取文本列表
                text_list = []
                documents = payload.get(EventPayload.DOCUMENTS)
                if documents:
                    text_list = [doc.text for doc in documents]
                else:
                    chunks = payload.get(EventPayload.CHUNKS)
                    if chunks:
                        text_list = chunks

                if text_list:
                    logger.info(f"    -> Embedding event for {len(text_list)} texts.")
                else:
                    logger.warning(f"    -> Embedding event payload did not contain text content.")