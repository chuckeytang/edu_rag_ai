# core/logging_config.py

import logging
import sys
from typing import Optional
import os

def setup_app_logging(level=logging.INFO, log_file: Optional[str] = None):
    """
    配置应用程序的全局日志。
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger = logging.getLogger(__name__)
        logger.info(f"Logging also configured to file: {log_file}")

    # 1. 设置常见第三方库的日志级别，避免它们输出过多
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    logging.getLogger('dashscope').setLevel(logging.INFO)

    # 2. 针对你自己的核心模块，单独设置日志级别
    # 现在主要关注与火山引擎、OSS和AI抽取相关的服务
    logging.getLogger('services.volcano_rag_service').setLevel(logging.DEBUG)
    logging.getLogger('services.document_oss_service').setLevel(logging.DEBUG)
    logging.getLogger('services.ai_extraction_service').setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    logger.info("Application logging configured successfully.")