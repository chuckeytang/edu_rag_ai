# services/ai_extraction_service.py

import json
import logging
import os
import shutil
from typing import List, Optional, Any

from fastapi import HTTPException

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser

from core.config import settings 
from models.schemas import ExtractedDocumentMetadata, Flashcard, FlashcardList, WxMineCollectSubjectList
from tools.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

MAX_LLM_INPUT_TOKENS_FOR_EXTRACTION = 58000

class AIExtractionService:
    def __init__(self,
                 llm_metadata_model: OpenAILike,
                 llm_flashcard_model: OpenAILike
                 ):
        self.llm_metadata = llm_metadata_model
        self.llm_flashcard = llm_flashcard_model
        # 移除oss_service的依赖，因为这个服务不再负责文件下载和本地处理。

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
                                            is_public: bool = False
                                            ) -> str:
        """
        这个服务现在只处理直接提供的文本内容。
        文件下载和处理逻辑已转移到调用服务。
        """
        if text_content:
            return text_content
        else:
            # 如果没有提供文本内容，就抛出错误
            raise ValueError("AI Extraction Service now only accepts 'text_content' directly. 'file_key' is not supported for local processing.")

    async def extract_document_metadata(self, 
                                        file_key: Optional[str] = None, 
                                        text_content: Optional[str] = None, 
                                        is_public: bool = True,
                                        user_provided_clazz: Optional[str] = None,
                                        user_provided_subject: Optional[str] = None,
                                        user_provided_exam: Optional[str] = None,
                                        user_provided_level: List[str] = [], 
                                        subscribed_subjects: List[WxMineCollectSubjectList] = [] 
                                        ) -> ExtractedDocumentMetadata: 
        """
        Extracts document metadata using the DeepSeek model.
        Supports content from OSS or direct text, and determines the bucket based on is_public.
        Considers user-provided preferences and subscribed subjects for better extraction.
        """
        doc_content = await self._get_content_from_oss_or_text(file_key, text_content, is_public)

        clazz_str = ", ".join(self.CLAZZ_OPTIONS)
        exam_boards_str = ", ".join(self.EXAM_BOARD_OPTIONS)
        levels_str = ", ".join(self.LEVEL_OPTIONS)
        subjects_str = ", ".join(self.SUBJECT_OPTIONS)
        doc_types_str = ", ".join(self.DOCUMENT_TYPE_OPTIONS)

        # --- 构建用户上下文信息字符串，以供提示词使用 ---
        user_context_info = []
        if user_provided_clazz:
            user_context_info.append(f"User's Preferred Curriculum System: '{user_provided_clazz}'")
        if user_provided_exam:
            user_context_info.append(f"User's Preferred Exam Board: '{user_provided_exam}'")
        if user_provided_subject:
            user_context_info.append(f"User's Preferred Subject: '{user_provided_subject}'")
        if user_provided_level:
            user_context_info.append(f"User's Preferred Levels: '{', '.join(user_provided_level)}'") 

        if subscribed_subjects:
            subscribed_str_list = []
            for sub_item in subscribed_subjects:
                sub_info = f"Subject: '{sub_item.subject}'"
                if sub_item.clazz:
                    sub_info += f", Class: '{sub_item.clazz}'"
                if sub_item.exam:
                    sub_info += f", Exam: '{sub_item.exam}'"
                subscribed_str_list.append(sub_info)
            user_context_info.append("User is subscribed to:\n" + "\n".join([f"  - {s}" for s in subscribed_str_list]))

        user_context_str = ""
        if user_context_info:
            user_context_str = "\nUser Context Information:\n" + "\n".join(user_context_info) + "\n"
            user_context_str += "When the document content is ambiguous or needs alignment with user intent, consider these hints. Prioritize direct evidence from the document itself.\n"
        else:
            user_context_str = "\nNo specific user context information provided.\n"

        prompt_template = f"""
        You are an expert assistant for analyzing educational content. Please extract metadata from the document content provided below.
        Your response MUST be a single, complete, and valid JSON object. Do not include any text before or after the JSON.

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

        {user_context_str}

        Document Content:
        ---
        {doc_content}
        ---

        Your response MUST be ONLY the JSON object. Do NOT add any extra text, explanations, or comments.
        Example JSON Output:
        {{
            "clazz": "IB",
            "exam": "CAIE",
            "labelList": ["Photosynthesis", "Carbon Cycle", "Plant Biology"],
            "levelList": ["HL"],
            "subject": "Biology",
            "type": "Handout"
        }}

        Your final output is:
        """

        logger.info(f"Final prompt for metadata extraction: {prompt_template}")

        parser = PydanticOutputParser(output_cls=ExtractedDocumentMetadata) 
        program = LLMTextCompletionProgram.from_defaults(
            output_parser=parser,
            prompt_template_str=prompt_template,
            llm=self.llm_metadata,
            verbose=True
        )

        try:
            logger.info(f"Preparing to extract metadata: {doc_content[:100]}...") 
            raw_output = await program._llm.acomplete(
                prompt=program.prompt.format(document_content=doc_content)
            )
            raw_output_text = raw_output.text
            
            logger.info(f"Raw LLM Output:\n{raw_output_text}")

            start_index = raw_output_text.find('{')
            end_index = raw_output_text.rfind('}')
            
            if start_index != -1 and end_index != -1 and end_index > start_index:
                cleaned_json_str = raw_output_text[start_index : end_index + 1]
                logger.info(f"Cleaned JSON String:\n{cleaned_json_str}")
                metadata = ExtractedDocumentMetadata.model_validate_json(cleaned_json_str)
            else:
                # 如果没有找到合法的 JSON 结构，则抛出异常
                raise ValueError("LLM output does not contain a valid JSON structure.")

            logger.info(f"Successfully extracted metadata: {metadata.model_dump_json(indent=2)}")
            
            # --- 后处理和默认值设置 ---
            if metadata.type is None or metadata.type.strip() == "":
                logger.warning(f"Extracted metadata 'type' is null or empty. Setting to default 'Book'. Original: {metadata.type}")
                metadata.type = "Book"

            if metadata.clazz is None or metadata.clazz.strip() == "":
                logger.warning(f"Extracted metadata 'clazz' is null or empty. Setting to 'None'. Original: {metadata.clazz}")
                metadata.clazz = "None"

            if metadata.subject is None or metadata.subject.strip() == "":
                logger.warning(f"Extracted metadata 'subject' is null or empty. Setting to 'None'. Original: {metadata.subject}")
                metadata.subject = "None" 

            if metadata.exam is None or metadata.exam.strip() == "":
                logger.warning(f"Extracted metadata 'exam' is null or empty. Setting to 'None'. Original: {metadata.exam}")
                metadata.exam = "None" 

            metadata.description = ""
            
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