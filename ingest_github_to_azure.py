# ingest_github_to_azure.py

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from github import Github, GithubException
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from sentence_transformers import SentenceTransformer
import uuid

# Import the helper function to create an index
from utils.azure_search_helpers import create_search_index

# --- SSL Configuration for Corporate Networks ---
CERTIFICATE_PATH = r"C:\Users\Administrator\Downloads\trusted_certs.crt"
if os.path.exists(CERTIFICATE_PATH):
    print(f"Found certificate bundle at: {CERTIFICATE_PATH}")
    os.environ['REQUESTS_CA_BUNDLE'] = CERTIFICATE_PATH
    os.environ['SSL_CERT_FILE'] = CERTIFICATE_PATH
else:
    print("Warning: Custom certificate bundle not found. Using default system certificates.")

# --- Configuration ---
load_dotenv()
# GitHub Config
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ORGANIZATION_NAME = "PremierInc"

# Azure AI Search Config
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME_REPOS = os.getenv("AZURE_SEARCH_INDEX_NAME_REPOS") # New index name

# Embedding Model & Batching Config
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = 100 # Process and upload in batches of 100 documents

# --- GitHub Fetching Functions (from your script) ---
def get_all_repositories(g, org_name):
    """Fetches all repositories for a given GitHub organization."""
    try:
        organization = g.get_organization(org_name)
        return organization.get_repos()
    except GithubException as e:
        print(f"Error fetching organization {org_name}: {e}")
        return []

def get_repository_details(repo):
    """Extracts title, description, README, and recent commits from a repository."""
    repo_details = {"name": repo.name, "title": repo.full_name, "description": repo.description or "", "readme": "", "commits": []}
    try:
        readme_content = repo.get_contents("README.md")
        repo_details["readme"] = readme_content.decoded_content.decode("utf-8")
    except GithubException:
        pass # Silently ignore missing READMEs
    try:
        recent_commits = repo.get_commits()[:10]
        for commit in recent_commits:
            repo_details["commits"].append(commit.commit.message)
    except GithubException:
        pass # Silently ignore errors fetching commits
    return repo_details

# --- Main Ingestion Logic ---
async def ingest_github_data():
    """
    Fetches repo data from GitHub, processes it, and uploads to Azure AI Search.
    """
    if not all([GITHUB_TOKEN, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME_REPOS]):
        print("Error: Please ensure all required environment variables are set.")
        return

    print("Initializing clients and models...")
    # Initialize GitHub client with SSL verification
    github_client = Github(GITHUB_TOKEN, verify=CERTIFICATE_PATH)
    
    # Initialize Azure Search client and encoder
    # These will use the REQUESTS_CA_BUNDLE environment variable for SSL
    credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dimensions = encoder.get_sentence_embedding_dimension()

    # 1. Create the search index if it doesn't exist
    print(f"Ensuring index '{AZURE_SEARCH_INDEX_NAME_REPOS}' exists...")
    create_search_index(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME_REPOS, embedding_dimensions)
    
    # Create a client to upload documents to the new index
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME_REPOS, credential=credential)

    documents_to_upload = []

    def upload_batch():
        """Helper to upload the current batch of documents."""
        if not documents_to_upload:
            return
        print(f"Uploading a batch of {len(documents_to_upload)} documents...")
        result = search_client.upload_documents(documents=documents_to_upload)
        successful_uploads = sum(1 for r in result if r.succeeded)
        print(f"Successfully uploaded {successful_uploads} documents in this batch.")
        if successful_uploads < len(documents_to_upload):
            for r in result:
                if not r.succeeded:
                    print(f"  - Failed to upload document {r.key}: {r.error_message}")
        documents_to_upload.clear()

    print(f"Fetching all repositories from organization: {ORGANIZATION_NAME}...")
    repos = get_all_repositories(github_client, ORGANIZATION_NAME)
    
    for repo in repos:
        print(f"Processing repository: {repo.full_name}")
        details = get_repository_details(repo)

        # Process Title and Description
        if details["title"]:
            text = f"Repository: {details['title']}\nDescription: {details['description']}"
            doc = {
                "id": str(uuid.uuid4()), # Use UUID for guaranteed uniqueness
                "text": text,
                "vector": encoder.encode(text).tolist(),
                "source": "description",
                "project": details["name"] # Use repo name as the project
            }
            documents_to_upload.append(doc)

        # Process README (chunked)
        if details["readme"]:
            for chunk in details["readme"].split("\n\n"):
                if chunk.strip():
                    doc = {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "vector": encoder.encode(chunk).tolist(),
                        "source": "readme_chunk",
                        "project": details["name"]
                    }
                    documents_to_upload.append(doc)
        
        # Process Commit Messages
        if details["commits"]:
            clean_commits = [msg.split('\n')[0] for msg in details["commits"] if not msg.startswith('Merge pull request')]
            commit_text = "\n".join([f"- {msg}" for msg in clean_commits])
            text = f"Recent commit messages for {details['title']}:\n{commit_text}"
            doc = {
                "id": str(uuid.uuid4()),
                "text": text,
                "vector": encoder.encode(text).tolist(),
                "source": "commits",
                "project": details["name"]
            }
            documents_to_upload.append(doc)

        # Check if it's time to upload a batch
        if len(documents_to_upload) >= BATCH_SIZE:
            upload_batch()

    # Upload any remaining documents
    upload_batch()
    
    print("\nGitHub data ingestion complete.")

if __name__ == "__main__":
    asyncio.run(ingest_github_data())