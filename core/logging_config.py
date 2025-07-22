# core/logging_config.py

import logging
import sys
from typing import Optional
import os

def setup_app_logging(level=logging.INFO, log_file: Optional[str] = None): # 可以通过参数设置默认级别
    """
    配置应用程序的全局日志。
    """
    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除所有现有的handlers，确保没有重复
    # 这在开发环境下非常重要，特别是在使用 uvicorn --reload 时
    # 否则每次重启服务都会添加新的handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建一个StreamHandler，将日志输出到控制台
    console_handler = logging.StreamHandler(sys.stdout) # 明确指定输出到标准输出
    
    # 定义日志格式
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True) # 确保日志目录存在
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger = logging.getLogger(__name__)
        logger.info(f"Logging also configured to file: {log_file}")

    # 1. 设置常见第三方库的日志级别，避免它们输出过多
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    logging.getLogger('chromadb').setLevel(logging.WARNING) # ChromaDB 通常不需要太多 INFO 日志
    logging.getLogger('llama_index').setLevel(logging.INFO) # LlamaIndex 核心，通常 INFO 即可
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING) # 避免大量连接日志
    logging.getLogger('dashscope').setLevel(logging.INFO) # DashScope API 客户端，INFO 即可

    # 2. **针对您需要调试的特定模块，单独设置为 DEBUG 级别**
    # 例如，您当前主要调试的是 CamelotPDFReader 和 IndexerService 的 chunking
    logging.getLogger('services.readers.camelot_pdf_reader').setLevel(logging.DEBUG)
    logging.getLogger('services.document_oss_service').setLevel(logging.DEBUG) # 处理文件下载和 Reader 调用
    logging.getLogger('services.indexer_service').setLevel(logging.DEBUG) # 处理 Document 到 Node 的转换和入库

    # 3. 其他您自己的模块，如果没有特别设置，将继承根 logger 的级别 (INFO)
    # 例如：
    # logging.getLogger('services.oss_service').setLevel(logging.INFO)
    # logging.getLogger('services.ai_extraction_service').setLevel(logging.INFO)
    # logging.getLogger('services.document_service').setLevel(logging.INFO)
    # logging.getLogger('services.task_manager_service').setLevel(logging.INFO)
    # logging.getLogger('api.endpoints.uploads').setLevel(logging.INFO)
    # logging.getLogger('api.endpoints.debug_index').setLevel(logging.INFO)

    logger = logging.getLogger(__name__) # 这里的 __name__ 是 core.logging_config
    logger.info("Application logging configured successfully.")