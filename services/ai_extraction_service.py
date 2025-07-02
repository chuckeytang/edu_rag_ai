# services/ai_extraction_service.py

import json
import logging
import os
import shutil
from typing import List, Dict, Any, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core import SimpleDirectoryReader

from core.config import settings # 假设settings包含DeepSeek配置
from models.schemas import ExtractedDocumentMetadata
from services.oss_service import oss_service # 用于从OSS下载文件

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

# --- Pydantic 模型定义 ---
class DocumentMetadata(BaseModel):
    clazz: Optional[str] = Field(None, description="课程体系名称，如：IB、IGCSE、AP")
    exam: Optional[str] = Field(None, description="考试局名称，如：CAIE、Edexcel、AQA")
    labelList: List[str] = Field([], description="标签名称列表，如：Znotes, Definitions, Paper 5, Book")
    levelList: List[str] = Field([], description="等级名称，如：AS, A2")
    subject: Optional[str] = Field(None, description="学科，如：Chemistry, Business, Computer Science")
    type: Optional[str] = Field(None, description="资料类型，如：Handout, Paper1, IA")

class Flashcard(BaseModel):
    term: str = Field(..., description="知识点或术语，例如：光合作用, 二氧化碳固定")
    explanation: str = Field(..., description="对术语的简洁解释，一小段话。")

class FlashcardList(BaseModel): # 用于解析JSON数组
    flashcards: List[Flashcard] = Field(..., description="提取到的记忆卡列表。")

class AIExtractionService:
    def __init__(self):
        # 从您的 settings 中获取 DeepSeek 配置
        self.deepseek_api_base = settings.DEEPSEEK_API_BASE
        self.deepseek_api_key = settings.DEEPSEEK_API_KEY
        
        self.llm_metadata = OpenAILike(
            model="deepseek-chat", # 或者 "deepseek-coder", 提取元数据可能更适合chat模型
            api_base=self.deepseek_api_base,
            api_key=self.deepseek_api_key,
            temperature=0.0, # 提取结构化数据时建议温度设低
        )
        self.llm_flashcard = OpenAILike(
            model="deepseek-chat", # 知识点提取也可用chat模型
            api_base=self.deepseek_api_base,
            api_key=self.deepseek_api_key,
            temperature=0.3, # 提取知识点可以稍微提高温度
        )

        # 预定义的选项集合 (这些应该从Spring后端获取，或从数据库加载)
        # 实际应用中，这些可能通过构造函数参数传入，或通过单独的服务获取
        self.CLAZZ_OPTIONS = ["A-Level", "IGCSE", "AP", "A-Level", "GCSE", "IB", "SAT", "竞赛"]
        self.EXAM_BOARD_OPTIONS = ["CAIE", "Edexcel", "AQA"]
        self.LABEL_OPTIONS = []
        self.LEVEL_OPTIONS = ["SL","HL","AS","A2"]
        self.SUBJECT_OPTIONS = ["Chinese A Lang&Lit","Economics","Physics","Biology","Business","English","Mathematics","Chemistry","Psychology","Accounting","Computer Science","Further Math","Geography","Calculus AB","Calculus BC","Computer Science A","Environmental Science","Macroeconomics","Microeconomics","Physics 1: Algebra-Based","Physics C: Electricity & Magnetism","Physics C: Mechanics","Statistics","Business Management","English A Lang&Lit","English A Literature","English B","ESS","Global Politics","History","Maths AA","Maths AI","Philosophy","SEHS","TOK","Visual Arts","Business Studies","English Literature","ESL","ICT","Additional Math","Combined Science","Maths","Reading and Writing","AMC10","BBO","BPho 中/高级","MAT","Math Kangaroo","NEC","UKChO","UKMT"]
        self.DOCUMENT_TYPE_OPTIONS = ["Handout", "Paper1", "Book", "IA"]

    async def _get_content_from_oss_or_text(self, 
                                            file_key: Optional[str] = None, 
                                            text_content: Optional[str] = None,
                                            is_public: bool = False # 新增参数
                                            ) -> str:
        """
        根据 file_key 从 OSS 下载文件并提取内容，或直接使用提供的文本内容。
        根据 is_public 参数决定从公共桶或私有桶下载。
        """
        if file_key:
            local_file_path = None
            try:
                # 根据 is_public 决定目标桶
                target_bucket = settings.OSS_PUBLIC_BUCKET_NAME if is_public else settings.OSS_PRIVATE_BUCKET_NAME
                logger.info(f"Attempting to download '{file_key}' from bucket: '{target_bucket}'")

                local_file_path = oss_service.download_file_to_temp(
                    object_key=file_key, 
                    bucket_name=target_bucket
                )
                
                # 使用 SimpleDirectoryReader 来读取文件内容
                reader = SimpleDirectoryReader(input_files=[local_file_path])
                documents = reader.load_data()
                
                full_content = "\n".join([doc.text for doc in documents])
                return full_content
            except Exception as e:
                logger.error(f"从OSS下载或读取文件 '{file_key}' 失败 (桶: {target_bucket}): {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"从OSS下载或读取文件失败: {e}. 请检查文件键和权限。")
            finally:
                if local_file_path and os.path.exists(os.path.dirname(local_file_path)):
                    shutil.rmtree(os.path.dirname(local_file_path))
                    logger.info(f"Cleaned up temporary directory for {file_key}")
        elif text_content:
            return text_content
        else:
            raise ValueError("必须提供 'file_key' 或 'text_content' 中的至少一个。")

    async def extract_document_metadata(self, file_key: Optional[str] = None, text_content: Optional[str] = None, is_public: bool = True) -> ExtractedDocumentMetadata: # 新增 is_public
        """
        使用DeepSeek模型从文档内容中提取元数据。
        支持从OSS下载文件或直接提供文本内容，并根据 is_public 决定桶。
        """
        doc_content = await self._get_content_from_oss_or_text(file_key, text_content, is_public) # 传递 is_public

        # ... (提示词构建逻辑不变) ...
        clazz_str = ", ".join(self.CLAZZ_OPTIONS)
        exam_boards_str = ", ".join(self.EXAM_BOARD_OPTIONS)
        labels_str = ", ".join(self.LABEL_OPTIONS)
        levels_str = ", ".join(self.LEVEL_OPTIONS)
        subjects_str = ", ".join(self.SUBJECT_OPTIONS)
        doc_types_str = ", ".join(self.DOCUMENT_TYPE_OPTIONS)

        prompt_template = f"""
        您是一个专业的教育内容分析助手。请从以下文档内容中提取其元数据。
        请确保您的输出严格遵守以下JSON格式，并从提供的选项中选择最匹配的值。
        如果某个字段在文档中找不到对应信息，或者无法匹配到提供的选项，请将其设为 null 或空列表。

        可选项列表：
        - 课程体系 (clazz): [{clazz_str}]
        - 考试局 (exam): [{exam_boards_str}]
        - 标签 (labelList): [{labels_str}]
        - 等级 (levelList): [{levels_str}]
        - 学科 (subject): [{subjects_str}]
        - 资料类型 (type): [{doc_types_str}]

        文档内容：
        ---
        {doc_content}
        ---

        请提供JSON格式的元数据：
        """

        parser = PydanticOutputParser(output_cls=ExtractedDocumentMetadata) # 使用更新后的模型名
        program = LLMTextCompletionProgram.from_defaults(
            output_parser=parser,
            prompt_template_str=prompt_template,
            llm=self.llm_metadata,
            verbose=True
        )

        try:
            metadata = await program.acall(document_content=doc_content)
            logger.info(f"成功提取元数据：{metadata.model_dump_json(indent=2)}") # 使用 model_dump_json，并可选地设置 indent 使输出更易读
            return metadata
        except Exception as e:
            logger.error(f"提取元数据时发生错误: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"元数据提取过程中AI服务出错: {e}") # 明确是AI服务错误

    async def extract_knowledge_flashcards(self, file_key: Optional[str] = None, text_content: Optional[str] = None, is_public: bool = True) -> List[Flashcard]: # 新增 is_public
        """
        使用DeepSeek模型从文档内容中提炼知识点，形成记忆卡。
        支持从OSS下载文件或直接提供文本内容，并根据 is_public 决定桶。
        """
        doc_content = await self._get_content_from_oss_or_text(file_key, text_content, is_public) # 传递 is_public

        # ... (提示词构建逻辑不变) ...
        prompt_template = """
        您是一个教育专家，请仔细阅读以下文档内容，并从中提炼出重要的知识点。
        每个知识点应该被表示为一张“记忆卡”，包含一个“术语”和一个“解释”。
        
        请遵循以下要求：
        1.  **术语 (term)**: 应该是一个简洁的词或词组，代表一个核心概念或专有名词。
        2.  **解释 (explanation)**: 应该是一小段话，清晰、准确地解释对应的术语。
        3.  请以JSON数组的形式返回所有记忆卡，每个元素包含 "term" 和 "explanation" 两个键。
        4.  只提取文档中明确提及的知识点，不要进行推断或引入外部信息。
        5.  如果文档内容较少或没有明显的知识点，可以返回空列表。

        文档内容：
        ---
        {doc_content}
        ---

        请提供JSON格式的记忆卡列表：
        """

        parser = PydanticOutputParser(output_cls=FlashcardList)
        program = LLMTextCompletionProgram.from_defaults(
            output_parser=parser,
            prompt_template_str=prompt_template,
            llm=self.llm_flashcard,
            verbose=True
        )

        try:
            flashcard_list_obj = await program.acall(document_content=doc_content)
            logger.info(f"成功提取记忆卡，数量：{len(flashcard_list_obj.flashcards)}")
            return flashcard_list_obj.flashcards
        except Exception as e:
            logger.error(f"提取记忆卡时发生错误: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"记忆卡提取过程中AI服务出错: {e}") # 明确是AI服务错误

# 创建一个单例实例
ai_extraction_service = AIExtractionService()