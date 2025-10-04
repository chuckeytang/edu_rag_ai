# In services/task_manager_service.py
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from models.schemas import TaskStatus as SchemasTaskStatus # 导入时重命名以避免与内部类名冲突

logger = logging.getLogger(__name__)

class TaskManagerService:
    def __init__(self):
        # 状态存储。在生产环境中，这里可以轻易地替换为Redis或数据库
        self._tasks: Dict[str, SchemasTaskStatus] = {}

    def create_task(self, task_type: str, 
                    initial_message: str = "Task scheduled.", 
                    initial_data: Optional[Dict[str, Any]] = None) -> SchemasTaskStatus:
        """创建一个新任务，并返回其初始状态"""
        task_id = str(uuid.uuid4())
        new_task = SchemasTaskStatus(
            task_id=task_id,
            kb_doc_id="",
            task_type=task_type,
            status="pending",
            message=initial_message,
            result=initial_data
        )
        self._tasks[task_id] = new_task
        logger.info(f"Created new task. ID: {task_id}, Type: {task_type}")
        return new_task

    def get_status(self, task_id: str) -> Optional[SchemasTaskStatus]:
        """获取指定任务的状态"""
        return self._tasks.get(task_id)

    def update_progress(self, task_id: str, progress: int, message: str):
        """更新任务的进度和消息"""
        task = self.get_status(task_id)
        if task and task.status not in ["success", "error", "duplicate"]:
            task.progress = min(max(progress, 0), 100) # 保证进度在0-100之间
            task.message = message
            task.kb_doc_id = ""
            task.status = "running"
            task.updated_at = datetime.now(timezone.utc)
            logger.info(f"[TASK_ID: {task_id}] Progress: {progress}%, Message: {message}")
        elif not task:
            logger.warning(f"Attempted to update progress for non-existent task_id: {task_id}")

    def finish_task(self, task_id: str, final_status: str, result: Optional[Dict] = None):
        """ 标记任务为完成（成功或失败）"""
        task = self.get_status(task_id)
        if task:
            task.progress = 100
            task.kb_doc_id = result.get('kb_doc_id')
            task.status = final_status
            task.message = result.get("message", f"Task finished with status: {final_status}") if result else f"Task finished with status: {final_status}"
            task.result = result
            task.updated_at = datetime.now(timezone.utc)
            logger.info(f"[TASK_ID: {task_id}] Task finished. Final Status: {final_status}")
        elif not task:
            logger.warning(f"Attempted to finish non-existent task_id: {task_id}")
