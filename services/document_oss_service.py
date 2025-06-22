# In services/document_oss_service.py

import os
import shutil
import logging
from typing import List, Optional, Tuple

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument
from core.config import settings
from models.schemas import RAGMetadata

logger = logging.getLogger(__name__)

class DocumentOssService:
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.data_config_dir = settings.DATA_CONFIG_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.data_config_dir, exist_ok=True)
        # This dictionary tracks processed files by their unique OSS file_key
        self.processed_files = self._load_processed_keys()

    @property
    def key_file_path(self):
        """Path to the file that stores the mapping of processed OSS keys."""
        return os.path.join(self.data_config_dir, ".oss_file_keys")

    def _load_processed_keys(self) -> dict:
        """Loads the map of processed oss_file_key -> original_filename."""
        keys = {}
        if os.path.exists(self.key_file_path):
            with open(self.key_file_path, "r") as f:
                for line in f:
                    if ":" in line:
                        key, filename = line.strip().split(":", 1)
                        keys[key] = filename
        logger.info(f"DocumentOssService: Loaded {len(keys)} processed OSS file keys from '{self.key_file_path}'.")
        return keys

    def _save_processed_keys(self):
        """Saves the current map of oss_file_key -> original_filename."""
        with open(self.key_file_path, "w") as f:
            for key, filename in self.processed_files.items():
                f.write(f"{key}:{filename}\n")

    def process_temp_file(self, local_file_path: str, file_key: str, metadata: dict) -> Tuple[List[LlamaDocument], List[LlamaDocument]]:
        """
        Processes a file directly from its temporary local path without saving it permanently.
        1. Updates the tracking of processed file keys with the filename for readability.
        2. Loads the document content FROM THE TEMP PATH and injects metadata.
        3. Filters blank pages.
        4. Returns both all loaded docs and the filtered docs.
        """
        # 1. Update tracking with the new file_key and save the readable filename as its value.
        #    NOTE: We are NOT saving the file permanently anymore.
        original_filename = metadata.get('file_name', 'unknown_filename')
        self.processed_files[file_key] = original_filename
        self._save_processed_keys()
        logger.info(f"Added OSS key '{file_key}' (Filename: '{original_filename}') to processed files cache.")

        # 2. Load documents directly from the provided temporary path
        logger.info(f"Loading documents directly from temporary path: '{local_file_path}'")
        all_docs = self._load_documents_with_metadata(
            file_name=local_file_path, # <-- Use the temp path here
            extra_metadata=metadata
        )

        # 3. Filter blank pages
        filtered_docs, _ = self._filter_documents(all_docs)

        return all_docs, filtered_docs

    def _load_documents_with_metadata(self, file_name: str, extra_metadata: Optional[dict] = None) -> List[LlamaDocument]:
        """Loads a single document and injects metadata into each page."""
        try:
            docs = SimpleDirectoryReader(input_files=[file_name]).load_data()

            for doc in docs:
                if doc.metadata is None:
                    doc.metadata = {}
                if extra_metadata:
                    doc.metadata.update(extra_metadata)
                if "page_label" not in doc.metadata:
                    doc.metadata["page_label"] = doc.metadata.get('page_label', '1')

            logger.debug(f"Metadata injected into {len(docs)} pages from file '{file_name}'.")
            return docs
        except FileNotFoundError as e:
            logger.error(f"File not found during document loading: {e}")
            raise ValueError(f"File not found: {e}")

    def _filter_documents(self, documents: List[LlamaDocument]) -> Tuple[List[LlamaDocument], dict]:
        """Filters out blank or empty pages from a list of documents."""
        filtered_docs = []
        page_info = {}
        for doc in documents:
            text = doc.text.strip()
            if text and "BLANK PAGE" not in text.upper():
                filtered_docs.append(doc)
                fname = doc.metadata.get("file_name", "unknown")
                if fname not in page_info:
                    page_info[fname] = set()
                page_info[fname].add(doc.metadata.get("page_label", "unknown"))
        return filtered_docs, page_info

# Create a singleton instance for the application to use
document_oss_service = DocumentOssService()