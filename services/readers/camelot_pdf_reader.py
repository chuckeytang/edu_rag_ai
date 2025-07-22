# services/readers/camelot_pdf_reader.py
import camelot
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import List, Dict, Any, Optional, Union
import os
import logging
from pypdf import PdfReader # 确保这是camelot-py安装的那个pypdf版本

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class CamelotPDFReader(BaseReader):
    """
    一个自定义的LlamaIndex文档加载器，使用camelot-py提取PDF中的表格和文本。
    并可以生成更细粒度的表格行级别或单元格级别的chunk。
    """
    def __init__(
        self,
        flavor: str = 'stream', # 'stream' 或 'lattice'
        table_settings: Optional[Dict[str, Any]] = None, # 传递给camelot.read_pdf的额外参数
        extract_text_also: bool = True, # 是否也提取非表格文本
        chunk_tables_by_row: bool = True, # 是否将表格按行切分
        **kwargs # 允许传入其他初始化参数，虽然BaseReader目前可能用不上
    ):
        super().__init__(**kwargs)
        self.flavor = flavor
        self.table_settings = table_settings or {}
        self.extract_text_also = extract_text_also
        self.chunk_tables_by_row = chunk_tables_by_row

    def load_data(
        self, file: Union[str, os.PathLike], extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """
        加载PDF文件，提取表格和文本，并返回LlamaIndex Document列表。
        """
        documents = []
        file_path = str(file)
        
        # 提取表格
        try:
            # 尝试使用 camelot 提取表格
            tables = camelot.read_pdf(
                file_path, 
                flavor=self.flavor, 
                pages='all', # 总是提取所有页面的表格
                **self.table_settings
            )
            
            for table_idx, table in enumerate(tables):
                # 提取表格内容
                table_text_content = table.df.to_csv(index=False) # 将表格转换为CSV格式字符串，更适合chunk
                
                # 构建元数据
                metadata = {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "page_label": str(table.page), # Camelot 返回的是页码
                    "table_index": table_idx,
                    "table_bbox": str(table.bbox), # 表格在页面上的位置
                    "document_type": "PDF_Table",
                    **({} if extra_info is None else extra_info) # 合并额外信息
                }

                if self.chunk_tables_by_row:
                    # 按行切分表格
                    for row_idx, row in table.df.iterrows():
                        row_text = row.to_string(header=False) # 将行转换为字符串
                        row_metadata = metadata.copy()
                        row_metadata["row_index"] = row_idx
                        row_metadata["document_type"] = "PDF_Table_Row" # 更细粒度的类型
                        documents.append(Document(text=row_text, metadata=row_metadata))
                else:
                    # 将整个表格作为一个chunk
                    documents.append(Document(text=table_text_content, metadata=metadata))

        except Exception as e:
            logger.warning(f"Error extracting tables from {file_path} using camelot-py: {e}")
            # 如果表格提取失败，不中断，尝试提取纯文本
            pass # 继续进行文本提取，或者直接返回空列表

        # 提取非表格文本
        if self.extract_text_also:
            try:
                # 使用 pypdf 提取所有文本（包括表格和非表格）
                # 注意：pypdf 提取的表格文本可能不如 camelot-py 精确
                pdf_reader = PdfReader(file_path)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_metadata = {
                            "file_path": file_path,
                            "file_name": os.path.basename(file_path),
                            "page_label": str(page_num + 1),
                            "document_type": "PDF_Text",
                            **({} if extra_info is None else extra_info)
                        }
                        documents.append(Document(text=page_text, metadata=text_metadata))
            except Exception as e:
                logger.error(f"Error extracting text from {file_path} using pypdf: {e}")
                # 文本提取也失败，记录错误并继续
        
        # 如果文件是一个图片PDF（无可选择文本），并且没有成功提取到内容
        if not documents:
             logger.warning(f"No content extracted from {file_path}. It might be a scanned PDF or empty.")
             # 可以选择返回一个空文档，或者抛出错误，取决于你的业务逻辑
             # documents.append(Document(text="", metadata={"file_path": file_path, "file_name": os.path.basename(file_path), "document_type": "Empty_PDF"}))

        return documents