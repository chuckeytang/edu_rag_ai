import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.query_service import QueryService
from services.document_service import DocumentService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# 仅为 LlamaIndex 相关的模块设置 DEBUG 级别
logging.getLogger("llama_index.core").setLevel(logging.DEBUG)
logging.getLogger("services.retrieval_service").setLevel(logging.DEBUG)

def main():
    """
    A script to deliberately and completely rebuild the vector index from all
    documents found in the data directory.
    
    WARNING: This is a destructive operation and will replace the existing index in OSS.
    """
    
    confirm = input("WARNING: This will delete the existing collection and rebuild the entire index from scratch. Are you sure? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborting.")
        return

    try:
        logging.info("Initializing services...")
        # These services will connect to OSS based on your .env configuration
        query_service = QueryService()
        document_service = DocumentService()

        # Delete the existing collection
        logging.info(f"Deleting existing collection: '{query_service.collection_name}'...")
        query_service.chroma_client.delete_collection(name=query_service.collection_name)
        logging.info("Collection deleted.")
        
        # Re-create it to ensure it's fresh
        query_service.chroma_client.get_or_create_collection(name=query_service.collection_name)

        # Load all documents from the source data directory
        logging.info(f"Loading all source documents from '{document_service.data_dir}'...")
        all_docs = document_service.load_documents() # This loads ALL files

        if not all_docs:
            logging.error("No documents found in the data directory. Aborting rebuild.")
            return

        logging.info(f"Found {len(all_docs)} total document pages. Filtering blank pages...")
        filtered_docs, _ = document_service.filter_documents(all_docs)

        if not filtered_docs:
            logging.error("No content found after filtering blank pages. Aborting rebuild.")
            return
            
        logging.info(f"Starting to initialize new index with {len(filtered_docs)} document chunks...")
        # The `initialize_index` method will now upload all data to your OSS bucket
        query_service.initialize_index(filtered_docs)
        
        logging.info("✅ Index rebuild completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred during index rebuild: {e}", exc_info=True)

if __name__ == "__main__":
    main()