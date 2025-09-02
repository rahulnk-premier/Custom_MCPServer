import os
import asyncio
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.core.exceptions import ServiceRequestError, HttpResponseError
from asyncio import sleep
from utils.document_parser import parse_document
from utils.azure_search_helpers import create_search_index
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

DOCUMENTS_DIR = r"E:\rnakka\Downloads\Fireball_AI_Training_Docs\Fireball_AI_Training_Docs"
EMBEDDING_DIMENSION = 3072

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len
)

def extract_metadata_from_path(file_path: Path) -> dict:
    parts = file_path.relative_to(DOCUMENTS_DIR).parts
    metadata = {"source": str(file_path.name)}
    if len(parts) > 1:
        metadata["project"] = parts[0]
    return metadata

async def ingest_data():
    print("Initializing clients and models...")
    # Azure OpenAI client
    openai_client = AzureOpenAI(
        api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY
    )

    create_search_index(
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_ADMIN_KEY,
        AZURE_SEARCH_INDEX_NAME,
        EMBEDDING_DIMENSION
    )

    credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=credential
    )

    documents_to_upload = []
    print(f"Starting ingestion from directory: {DOCUMENTS_DIR}")
    file_paths = list(Path(DOCUMENTS_DIR).rglob("*.*"))

    for file_path in file_paths:
        if file_path.is_file():
            print(f"Processing: {file_path.name}")
            text_content = parse_document(str(file_path))
            if not text_content:
                continue
            metadata = extract_metadata_from_path(file_path)
            chunks = text_splitter.split_text(text_content)

            for chunk in chunks:
                try:
                    embedding_response = openai_client.embeddings.create(
                        input=chunk,
                        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                    )
                    vector_embedding = embedding_response.data[0].embedding
                except Exception as e:
                    print(f"Error generating embedding for a chunk: {e}")
                    continue

                document = {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "vector": vector_embedding,
                    "source": metadata.get("source"),
                    "project": metadata.get("project")
                }
                documents_to_upload.append(document)

    if documents_to_upload:
        print(f"\nTotal chunks to upload: {len(documents_to_upload)}")
        await batch_upload_with_retry(search_client, documents_to_upload)
    else:
        print("No new documents to upload.")

async def batch_upload_with_retry(client: SearchClient, docs: list,
                                  batch_size: int = 200,
                                  max_retries: int = 3,
                                  backoff_factor: float = 2.0):
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i: i + batch_size]
        attempt = 0
        while attempt < max_retries:
            try:
                print(f"Uploading batch {i // batch_size + 1} (docs {i}–{i + len(batch) - 1}) try {attempt + 1}")
                results = client.upload_documents(documents=batch)
                success = sum(1 for r in results if getattr(r, "succeeded", False))
                print(f"Batch upload success: {success}/{len(batch)}")
                # Log failures
                for r in results:
                    if not getattr(r, "succeeded", False):
                        print(f"  - Failed document key: {r.key}, error: {r.error_message}")
                break  # exit retry loop on success
            except (ServiceRequestError, HttpResponseError) as e:
                attempt += 1
                wait_time = backoff_factor ** attempt
                print(f"Batch failed with {e.__class__.__name__}: {e}. Retrying in {wait_time:.1f}s...")
                await sleep(wait_time)
        else:
            print(f"Batch starting at index {i} failed after {max_retries} attempts.")

if __name__ == "__main__":
    asyncio.run(ingest_data())
