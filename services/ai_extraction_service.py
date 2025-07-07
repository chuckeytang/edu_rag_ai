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
from models.schemas import ExtractedDocumentMetadata, Flashcard, FlashcardList
from services.oss_service import OssService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

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
            kwargs={
                'response_format': {'type': 'json_object'} 
            }
        )
        self.llm_flashcard = OpenAILike(
            model="deepseek-chat", # 知识点提取也可用chat模型
            api_base=self.deepseek_api_base,
            api_key=self.deepseek_api_key,
            temperature=0.3, # 提取知识点可以稍微提高温度
            kwargs={
                'response_format': {'type': 'json_object'} 
            }
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
                                            is_public: bool = False,
                                            oss_service: OssService = None
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
        You are an expert assistant for analyzing educational content. Please extract metadata and generate a concise summary (description) from the document content provided below.
        Ensure your output strictly follows the specified JSON format.

        Available Options for Matching:
        - Curriculum System (clazz): [{clazz_str}]
        - Exam Board (exam): [{exam_boards_str}]
        - Level (levelList): [{levels_str}]
        - Subject (subject): [{subjects_str}]
        - Document Type (type): [{doc_types_str}]

        Special Instructions for 'labelList':
        - For 'labelList', identify 3 to 6 keywords or short phrases from the document that best represent its main topics or categories.
        - These labels should be single words or short phrases, acting as descriptive tags.
        - Do NOT select from a predefined list for 'labelList'; generate them based on the document's content.

        Special Instruction for 'description':
        - Generate a concise, objective summary of the document's content.
        - The description should be a maximum of 1024 characters.

        Document Content:
        ---
        {doc_content}
        ---

        Please provide the metadata and description in JSON format. Here is an example of the desired JSON structure:

        EXAMPLE JSON OUTPUT:
        {{
            "clazz": "IB",
            "exam": "CAIE",
            "labelList": ["Photosynthesis", "Carbon Cycle", "Plant Biology"],
            "levelList": ["HL"],
            "subject": "Biology",
            "type": "Handout",
            "description": "This document provides a detailed overview of the photosynthesis process, including light-dependent and light-independent reactions, and its role in the global carbon cycle, suitable for advanced biology students. The summary is concise and covers the main topics without excessive detail. It ensures all key aspects are highlighted for quick understanding, adhering to the character limit."
        }}
        """

        parser = PydanticOutputParser(output_cls=ExtractedDocumentMetadata) # 使用更新后的模型名
        program = LLMTextCompletionProgram.from_defaults(
            output_parser=parser,
            prompt_template_str=prompt_template,
            llm=self.llm_metadata,
            verbose=True
        )

        try:
            logger.info(f"Preparing to extract metadata: {doc_content[:100]}...") 
            metadata = await program.acall(document_content=doc_content)
            logger.info(f"Successfully extracted metadata: {metadata.model_dump_json(indent=2)}")
            return metadata
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"AI service error during metadata extraction: {e}") 

    async def extract_knowledge_flashcards(self, file_key: Optional[str] = None, text_content: Optional[str] = None, is_public: bool = False) -> List[Flashcard]:
        """
        Extracts knowledge points (flashcards) from the document content using the DeepSeek model.
        Supports content from OSS or direct text, and determines the bucket based on is_public.
        """
        doc_content = await self._get_content_from_oss_or_text(file_key, text_content, is_public)

        prompt_template = f"""
        You are an educational expert. Please carefully read the following document content and extract important knowledge points from it.
        Each knowledge point should be represented as a "flashcard", consisting of a "term" and an "explanation".
        
        EXAMPLE JSON OUTPUT:
        [
            {{
                "term": "Photosynthesis",
                "explanation": "The process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water."
            }},
            {{
                "term": "Calvin Cycle",
                "explanation": "The set of chemical reactions that take place in chloroplasts during photosynthesis. The cycle is light-independent."
            }}
        ]
        
        Please follow these requirements:
        1.  **Term**: Should be a concise word or phrase representing a core concept or technical term.
        2.  **Explanation**: Should be a brief paragraph, clearly and accurately explaining the corresponding term.
        3.  Extract only knowledge points explicitly mentioned in the document; do not infer or introduce external information.
        4.  If the document content is minimal or contains no clear knowledge points, return an empty list.
        
        Document Content:
        ---
        {doc_content}
        ---
        
        Please provide the list of flashcards in JSON format:
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
            logger.info(f"Successfully extracted flashcards, count: {len(flashcard_list_obj.flashcards)}")
            return flashcard_list_obj.flashcards
        except Exception as e:
            logger.error(f"Error during flashcard extraction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"AI service error during flashcard extraction: {e}")
