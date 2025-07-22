import os
import re
import shutil
import hashlib
import logging

from typing import List, Optional, Tuple
from fastapi import UploadFile
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument
from core.config import settings
from models.schemas import RAGMetadata

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.data_config_dir = settings.DATA_CONFIG_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.data_config_dir, exist_ok=True)
        self.file_hashes = self._load_existing_hashes()
        logger.info("DocumentService initialized.")
    
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
        logger.info(f"DocumentService: Loaded {len(hashes)} content hashes from '{self.hash_file}'.")
        return hashes

    def _save_file_hashes(self):
        with open(self.hash_file, "w") as f:
            for file_hash, file_path in self.file_hashes.items():
                f.write(f"{file_hash}:{file_path}\n")

    @staticmethod
    def calculate_file_hash_from_path(file_path: str) -> str:
        """Calculates SHA256 hash for a file given its local path."""
        hash_obj = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def list_all_files(self) -> List[str]:
        """
        返回 data_dir 下所有文件的文件名（不包含路径）
        """
        if not os.path.exists(self.data_dir):
            return []

        return [
            f for f in os.listdir(self.data_dir)
            if os.path.isfile(os.path.join(self.data_dir, f))
        ]
    
    async def _calculate_upload_file_hash(self, file: UploadFile) -> str:
        """Calculates SHA256 hash for an UploadFile content without saving."""
        # 必须先seek(0)，因为file.file可能在前面被读取过
        await file.seek(0) 
        hash_obj = hashlib.sha256()
        while chunk := await file.read(8192):
            hash_obj.update(chunk)
        # 再次seek(0)，以便文件内容可以再次被读取（例如被保存）
        await file.seek(0) 
        return hash_obj.hexdigest()

    async def is_duplicate_file(self, file: UploadFile) -> bool:
        file_hash = await self._calculate_upload_file_hash(file)
        return file_hash in self.file_hashes

    async def save_uploaded_file(self, file: UploadFile) -> str:
        file_hash = await self._calculate_upload_file_hash(file)
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

    def load_documents(self, file_names: List[str] = None, extra_metadata: Optional[dict] = None) -> List[LlamaDocument]:
        if file_names:
            input_files = [
                fname if os.path.isabs(fname) or "/" in fname or "\\" in fname
                else os.path.join(self.data_dir, fname)
                for fname in file_names
            ]
        else:
            input_files = None

        try:
            docs = SimpleDirectoryReader(
                input_files=input_files,
                input_dir=self.data_dir if not input_files else None
            ).load_data()

            for idx, doc in enumerate(docs):
                # Always initialize metadata if it's None
                if doc.metadata is None:
                    doc.metadata = {}
                
                # --- THIS IS THE KEY INJECTION POINT ---
                # Inject the rich metadata from the Java service
                if extra_metadata:
                    doc.metadata.update(extra_metadata)

                # Fallback completion logic (mostly for old files or direct uploads)
                if "file_name" not in doc.metadata:
                     doc.metadata["file_name"] = os.path.basename(doc.id_)
                if "page_label" not in doc.metadata:
                    doc.metadata["page_label"] = str(idx + 1)
            
            if docs and extra_metadata:
                 logger.debug(f"Metadata injected into first page: {docs[0].metadata}")

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