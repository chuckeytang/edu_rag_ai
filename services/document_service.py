import os
import re
import shutil
import hashlib

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
from typing import List, Optional, Tuple
from fastapi import UploadFile
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument
from core.config import settings
class DocumentService:
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.data_config_dir = settings.DATA_CONFIG_DIR
        self.file_hashes = self._load_existing_hashes()
    @property
    def hash_file(self):
        return os.path.join(self.data_config_dir, ".file_hashes")
        
    def _load_existing_hashes(self) -> dict:
        hashes = {}
        if os.path.exists(self.hash_file):
            with open(self.hash_file, "r") as f:
                for line in f:
                    if ":" in line:
                        file_hash, file_path = line.strip().split(":", 1)
                        hashes[file_hash] = file_path
        return hashes
    def _save_file_hashes(self):
        with open(self.hash_file, "w") as f:
            for file_hash, file_path in self.file_hashes.items():
                f.write(f"{file_hash}:{file_path}\n")
    @staticmethod
    async def _calculate_file_hash(file: UploadFile) -> str:
        file.file.seek(0)
        hash_obj = hashlib.sha256()
        while chunk := await file.read(8192):
            hash_obj.update(chunk)
        await file.seek(0)
        return hash_obj.hexdigest()

    async def is_duplicate_file(self, file: UploadFile) -> bool:
        file_hash = await self._calculate_file_hash(file)
        return file_hash in self.file_hashes

    async def save_uploaded_file(self, file: UploadFile) -> str:
        file_hash = await self._calculate_file_hash(file)
        if file_hash in self.file_hashes:
            existing_file = self.file_hashes[file_hash]
            if os.path.exists(existing_file):
                return existing_file  
        file_path = os.path.join(self.data_dir, file.filename)
        counter = 1
        while os.path.exists(file_path):
            name, ext = os.path.splitext(file.filename)
            file_path = os.path.join(self.data_dir, f"{name}_{counter}{ext}")
            counter += 1
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        self.file_hashes[file_hash] = file_path
        self._save_file_hashes()
        return file_path

    def load_documents(self, file_names: List[str] = None) -> List[LlamaDocument]:
        if file_names:
            input_files = [os.path.join(self.data_dir, fname) for fname in file_names]
        else:
            input_files = None

        try:
            # 加载原始文档
            docs = SimpleDirectoryReader(
                input_files=input_files,
                input_dir=self.data_dir if not input_files else None
            ).load_data()

            # 安全补全 metadata
            for idx, doc in enumerate(docs):
                # 初始化 metadata 字典
                if doc.metadata is None:
                    doc.metadata = {}

                # 设置默认 file_name
                if "file_name" not in doc.metadata or doc.metadata["file_name"] is None:
                    if input_files and len(input_files) == 1:
                        doc.metadata["file_name"] = os.path.basename(input_files[0])
                    else:
                        doc.metadata["file_name"] = f"unknown_file_{idx}"

                # 设置默认 page_label
                if "page_label" not in doc.metadata or doc.metadata["page_label"] is None:
                    doc.metadata["page_label"] = str(idx + 1)

                logger.debug(f"[Document {idx}] Metadata: {doc.metadata}")

            return docs

        except FileNotFoundError as e:
            raise ValueError(f"File not found: {e}")

    def filter_documents(self, documents: List[LlamaDocument]) -> Tuple[List[LlamaDocument], dict]:
        filtered_docs = []
        page_info = {}
        for doc in documents:
            if not self._is_blank_page(doc.text):
                filtered_docs.append(doc)
                fname = doc.metadata["file_name"]
                if fname not in page_info:
                    page_info[fname] = set()
                page_info[fname].add(doc.metadata["page_label"])
        return filtered_docs, page_info
        
    @staticmethod
    def _is_blank_page(text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        return all(
            line.strip() == "" or "BLANK PAGE" in line.upper()
            for line in text.splitlines()
        )
    @staticmethod
    def find_question(doc: LlamaDocument, pattern: str) -> Optional[Tuple[str, int]]:
        normalized = re.sub(r'[\s.]', '', pattern).lower()
        parts = re.match(r'^(\d+)(?:\(?([a-z])\)?)?(?:\(?([a-z]+)\)?)?', normalized)
        if not parts:
            return None
        main_num, sub1, sub2 = parts.groups()
        lines = doc.text.split('\n')
        for i, line in enumerate(lines):
            line_clean = re.sub(r'[\s.]', '', line).lower()
            pattern_parts = [main_num]
            if sub1: pattern_parts.append(f"\(?{sub1}\)?")
            if sub2: pattern_parts.append(f"\(?{sub2}\)?")
            pattern = r'^' + r'[ .]*'.join(pattern_parts) + r'\b'
            if re.search(pattern, line_clean):
                return line.strip(), i
        return None
    def get_document_by_filename(self, filename: str) -> Optional[LlamaDocument]:
        docs = self.load_documents([filename])
        filtered_docs, _ = self.filter_documents(docs)
        return filtered_docs[0] if filtered_docs else None
    def get_documents_by_filenames(self, filenames: List[str]) -> List[LlamaDocument]:
        docs = self.load_documents(filenames)
        filtered_docs, _ = self.filter_documents(docs)
        return filtered_docs
    def debug_question_search(self, filename: str, question: str):
        doc = self.get_document_by_filename(filename)
        if not doc:
            return {"error": "File not found"}
        print(f"\n=== Document Structure ===")
        print(doc.text[:500] + "...")
        formats = [
            question,  
            question.replace("(", ".").replace(")", ""),  
            question.replace("(", " ").replace(")", " "),  
            question.replace(")(", ") (")  
        ]
        results = {}
        for fmt in formats:
            result = self.find_question(doc, fmt)
            results[fmt] = "Found" if result else "Not Found"
        return {
            "file": filename,
            "tested_formats": results,
            "raw_text_sample": doc.text[:200] + "..."
        }
document_service = DocumentService()