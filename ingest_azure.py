# ingest.py
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import json

from utils.document_parser import parse_document
from utils.azure_search_helpers import create_search_index

# --- SSL Configuration for Corporate Networks ---
# This is crucial for environments with SSL inspection.
CERTIFICATE_PATH = r"C:\Users\Administrator\Downloads\trusted_certs.crt"
if os.path.exists(CERTIFICATE_PATH):
    print(f"Found certificate bundle at: {CERTIFICATE_PATH}")
    os.environ['REQUESTS_CA_BUNDLE'] = CERTIFICATE_PATH
else:
    print("Warning: Custom certificate bundle not found. Using default system certificates.")


# --- Configuration ---
load_dotenv()
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
DOCUMENTS_DIR = r"E:\rnakka\Downloads\Fireball_AI_Training_Docs\Fireball_AI_Training_Docs" # IMPORTANT: Set this to your documents folder

# --- 1. METADATA EXTRACTION ---
def extract_metadata_from_path(file_path: Path) -> dict:
    """
    Creates hierarchical metadata from the file path.
    Example: /docs/ProjectPhoenix/auth.md -> {"project": "ProjectPhoenix", "file_name": "auth.md"}
    """
    parts = file_path.relative_to(DOCUMENTS_DIR).parts
    metadata = {"source": str(file_path.name)}
    if len(parts) > 1:
        metadata["project"] = parts[0]
    return metadata

# --- 2. CHUNKING ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len
)

# --- 3. MAIN INGESTION LOGIC ---
async def ingest_data():
    """
    Walks through the document directory, processes files, and uploads them to Azure AI Search.
    """
    print("Initializing clients and models...")
    # The AzureKeyCredential and SearchClient will automatically use the
    # REQUESTS_CA_BUNDLE environment variable for SSL verification.
    credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dimensions = encoder.get_sentence_embedding_dimension()

    # Create the search index if it doesn't exist
    # This client will also respect the environment variable.
    create_search_index(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME, embedding_dimensions)

    # Create a client to upload documents
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=credential)

    documents_to_upload = []
    
    print(f"Starting ingestion from directory: {DOCUMENTS_DIR}")
    file_paths = list(Path(DOCUMENTS_DIR).rglob("*.*"))
    
    for file_path in file_paths:
        if file_path.is_file():
            print(f"Processing: {file_path.name}")
            
            # Stage 1: Partition (Parse the document)
            text_content = parse_document(str(file_path))
            if not text_content:
                continue
            
            # Stage 2: Extract Metadata
            metadata = extract_metadata_from_path(file_path)
            
            # Stage 3: Chunk
            chunks = text_splitter.split_text(text_content)
            
            for chunk in chunks:
                # Create a document for Azure AI Search
                document = {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "vector": encoder.encode(chunk).tolist(),
                    "source": metadata.get("source"),
                    "project": metadata.get("project")
                }
                documents_to_upload.append(document)

    # Batch upload documents to Azure AI Search
    if documents_to_upload:
        print(f"\nUploading {len(documents_to_upload)} documents to Azure AI Search...")
        result = search_client.upload_documents(documents=documents_to_upload)
        
        # Check for errors
        successful_uploads = sum(1 for r in result if r.succeeded)
        print(f"Successfully uploaded {successful_uploads} documents.")
        
        for r in result:
            if not r.succeeded:
                print(f"Failed to upload document {r.key}: {r.error_message}")
    else:
        print("No new documents to upload.")

if __name__ == "__main__":
    # IMPORTANT: Before running, ensure DOCUMENTS_DIR is set correctly.
    asyncio.run(ingest_data())