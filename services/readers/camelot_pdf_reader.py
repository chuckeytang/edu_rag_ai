# services/readers/camelot_pdf_reader.py
import camelot
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import List, Dict, Any, Optional, Union
import os
import logging
from pypdf import PdfReader 

logger = logging.getLogger(__name__)

class CamelotPDFReader(BaseReader):
    """
    一个自定义的LlamaIndex文档加载器，使用camelot-py提取PDF中的表格和文本。
    并可以生成更细粒度的表格行级别或单元格级别的chunk。
    """
    
    def __init__(
        self,
        flavor: str = 'lattice', # 默认值
        table_settings: Optional[Dict[str, Any]] = None, # 传递给camelot.read_pdf的额外参数
        extract_text_also: bool = True,
        chunk_tables_by_row: bool = True,
        **kwargs # 允许传入其他初始化参数
    ):
        super().__init__(**kwargs) # 首先调用父类的构造函数

        self.flavor = flavor
        self.extract_text_also = extract_text_also
        self.chunk_tables_by_row = chunk_tables_by_row

        # 处理 table_settings，确保兼容性
        # 从提供的 table_settings 开始，如果为 None，则为空字典
        _effective_table_settings = table_settings.copy() if table_settings is not None else {}

        if self.flavor == 'lattice':
            # 'edge_tol' 不兼容 'lattice' 模式，如果存在则移除
            if 'edge_tol' in _effective_table_settings:
                logger.warning("[CamelotPDFReader] 'edge_tol' is not used with flavor='lattice'. Removing from table_settings.")
                _effective_table_settings.pop('edge_tol')
            # 为 lattice 模式添加默认的 line_scale，如果未提供
            if 'line_scale' not in _effective_table_settings:
                 _effective_table_settings['line_scale'] = 40 # 常见默认值

        elif self.flavor == 'stream': # 仅为 stream 模式处理特定参数
            # 'line_scale' 不兼容 'stream' 模式，如果存在则移除
            if 'line_scale' in _effective_table_settings:
                logger.warning("[CamelotPDFReader] 'line_scale' is not used with flavor='stream'. Removing from table_settings.")
                _effective_table_settings.pop('line_scale')
            # 为 stream 模式添加默认的 edge_tol，如果未提供
            if 'edge_tol' not in _effective_table_settings:
                 _effective_table_settings['edge_tol'] = 50 # 常见默认值
        
        # 将处理后的设置赋值给实例属性，只赋值一次
        self.table_settings = _effective_table_settings # <--- 只在这里赋值一次 self.table_settings

        logger.debug(f"[CamelotPDFReader] Initialized with flavor='{self.flavor}', extract_text_also={self.extract_text_also}, chunk_tables_by_row={self.chunk_tables_by_row}, table_settings={self.table_settings}")

    def load_data(
        self, file: Union[str, os.PathLike], extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """
        加载PDF文件，提取表格和文本，并返回LlamaIndex Document列表。
        """
        documents = []
        file_path = str(file)
        logger.info(f"[CamelotPDFReader] Loading data from file: {file_path}")
        extracted_tables_count = 0
        extracted_pages_text_count = 0

        try:
            # 尝试使用 camelot 提取表格
            logger.debug(f"[CamelotPDFReader] Attempting to extract tables with flavor='{self.flavor}' and settings={self.table_settings}")
            tables = camelot.read_pdf(
                file_path, 
                flavor=self.flavor, 
                pages='all', # 总是提取所有页面的表格
                **self.table_settings
            )
            extracted_tables_count = tables.n # 获取提取到的表格数量
            logger.info(f"[CamelotPDFReader] Found {extracted_tables_count} tables using camelot-py.")
            
            if extracted_tables_count > 0: # 只有当识别到表格时才进入处理循环
                for table_idx, table in enumerate(tables):
                    logger.debug(f"[CamelotPDFReader] Processing table {table_idx+1}/{extracted_tables_count} from page {table.page}.")
                    
                    table_bbox_str = "N/A_Error" # 默认值，如果无法获取
                    table_md_content = table.df.to_markdown(index=False) 
                    # LLM 会识别这种模式，并知道这是一段表格数据
                    table_formatted_text = (
                        f"--- START TABLE (Page {table.page}, Table {table_idx+1}) ---\n"
                        f"{table_md_content}\n"
                        f"--- END TABLE ---"
                    )

                    try:
                        x1, y1, x2, y2 = table._bbox
                        table_bbox_str = f"({x1}, {y1}, {x2}, {y2})"
                    except AttributeError:
                        logger.warning(f"[CamelotPDFReader] Table object has no 'bbox' attribute in version 1.0.0, trying alternative.")
                        if 'page_bbox' in table.parsing_report:
                            table_bbox_str = str(table.parsing_report['page_bbox'])
                        else:
                            table_bbox_str = "N/A"
                    except Exception as e:
                        logger.warning(f"[CamelotPDFReader] Error accessing table.bbox or its components: {e}", exc_info=True)
                        table_bbox_str = "Error_bbox" 

                    # 构建元数据
                    metadata = {
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                        "page_label": str(table.page),
                        "table_index": table_idx,
                        "table_bbox": table_bbox_str,
                        "document_type": "PDF_Table", # 现在是整个表格
                        **({} if extra_info is None else extra_info) # 合并额外信息
                    }

                    if table_formatted_text.strip():
                        documents.append(Document(text=table_formatted_text, metadata=metadata))
                        logger.debug(f"[CamelotPDFReader] Added full table chunk (Pg:{metadata['page_label']}, Table:{table_idx}): '{table_formatted_text.strip()[:100]}...'")
                    else:
                        logger.debug(f"[CamelotPDFReader] Skipping empty full table chunk (Pg:{metadata['page_label']}, Table:{table_idx}).")

            else:
                logger.info(f"[CamelotPDFReader] No tables found by camelot-py in {file_path}.")


        except Exception as e:
            logger.warning(f"[CamelotPDFReader] Major Error during table extraction from {file_path} using camelot-py (read_pdf failed): {e}", exc_info=True)
            # 如果 camelot.read_pdf 失败，tables_successfully_extracted 仍为 False
            pass 
        
        # 提取非表格文本
        if self.extract_text_also: 
            try:
                logger.debug(f"[CamelotPDFReader] Attempting to extract non-table text using pypdf.")
                pdf_reader = PdfReader(file_path)
                extracted_pages_text_count = len(pdf_reader.pages)
                
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
                        logger.debug(f"[CamelotPDFReader] Added PDF_Text chunk (Page:{page_num+1}): '{page_text.strip()[:100]}...'")
                    else:
                        logger.debug(f"[CamelotPDFReader] Page {page_num+1} extracted no text (might be image-based or empty).")

            except Exception as e:
                logger.error(f"[CamelotPDFReader] Error extracting text from {file_path} using pypdf: {e}", exc_info=True)
        
        # 如果文件是一个图片PDF（无可选择文本），并且没有成功提取到任何内容
        if not documents:
             logger.warning(f"[CamelotPDFReader] No content extracted at all from {file_path}. It might be a scanned PDF or empty.")
             # documents.append(Document(text="", metadata={"file_path": file_path, "file_name": os.path.basename(file_path), "document_type": "Empty_PDF"}))

        logger.info(f"[CamelotPDFReader] Finished loading. Generated {len(documents)} LlamaDocuments for {file_path}. (Tables: {extracted_tables_count}, Pages text: {extracted_pages_text_count}).")
        return documents