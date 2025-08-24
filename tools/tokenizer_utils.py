# core/tokenizer_utils.py
import logging
import tiktoken
from typing import Optional

logger = logging.getLogger(__name__)

_tokenizer = None
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base") 
    logger.info("Using tiktoken 'cl100k_base' for token counting estimation.")
except Exception as e:
    logger.warning(f"Could not load tiktoken tokenizer 'cl100k_base', token counting may be inaccurate: {e}")

def get_tokenizer():
    """Returns the global tokenizer instance."""
    return _tokenizer