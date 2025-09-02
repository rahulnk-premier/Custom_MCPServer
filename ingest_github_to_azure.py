# ingest_github_to_azure.py

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from github import Github, GithubException
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
# CHANGED: Replacing SentenceTransformer with the official OpenAI library
from openai import AzureOpenAI 
import uuid

# Import the helper function to create an index
# This helper should already be updated to handle index deletion/recreation
from utils.azure_search_helpers import create_search_index


# --- Configuration ---
load_dotenv()
# GitHub Config
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ORGANIZATION_NAME = "PremierInc"

# Azure AI Search Config
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME_REPOS = os.getenv("AZURE_SEARCH_INDEX_NAME_REPOS")

# NEW: Azure OpenAI Config
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# The text-embedding-3-large model has a fixed dimension of 3072
EMBEDDING_DIMENSION = 3072
BATCH_SIZE = 100 # Process and upload in batches of 100 documents

# --- GitHub Fetching Functions (Unchanged) ---
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
    required_vars = [
        GITHUB_TOKEN, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, 
        AZURE_SEARCH_INDEX_NAME_REPOS, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY,
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    ]
    if not all(required_vars):
        print("Error: Please ensure all required environment variables are set.")
        return

    print("Initializing clients...")
    github_client = Github(GITHUB_TOKEN)
    
    # NEW: Initialize Azure OpenAI Client
    openai_client = AzureOpenAI(
        api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY
    )

    # 1. Create or update the search index with the correct embedding dimension
    print(f"Ensuring index '{AZURE_SEARCH_INDEX_NAME_REPOS}' exists with dimension {EMBEDDING_DIMENSION}...")
    create_search_index(
        AZURE_SEARCH_ENDPOINT, 
        AZURE_SEARCH_ADMIN_KEY, 
        AZURE_SEARCH_INDEX_NAME_REPOS, 
        EMBEDDING_DIMENSION
    )
    
    # Create a client to upload documents
    credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT, 
        index_name=AZURE_SEARCH_INDEX_NAME_REPOS, 
        credential=credential
    )

    documents_to_upload = []

    def get_embedding(text: str):
        """Helper function to generate embedding for a given text."""
        try:
            response = openai_client.embeddings.create(
                input=text,
                model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"  - Could not generate embedding for text snippet. Error: {e}")
            return None

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
            vector = get_embedding(text)
            if vector:
                documents_to_upload.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "vector": vector,
                    "source": "description",
                    "project": details["name"]
                })

        # Process README (chunked)
        if details["readme"]:
            for chunk in details["readme"].split("\n\n"):
                if chunk.strip():
                    vector = get_embedding(chunk)
                    if vector:
                        documents_to_upload.append({
                            "id": str(uuid.uuid4()),
                            "text": chunk,
                            "vector": vector,
                            "source": "readme_chunk",
                            "project": details["name"]
                        })
        
        # Process Commit Messages
        if details["commits"]:
            clean_commits = [msg.split('\n')[0] for msg in details["commits"] if not msg.startswith('Merge pull request')]
            commit_text_blob = "\n".join([f"- {msg}" for msg in clean_commits])
            text = f"Recent commit messages for {details['title']}:\n{commit_text_blob}"
            vector = get_embedding(text)
            if vector:
                documents_to_upload.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "vector": vector,
                    "source": "commits",
                    "project": details["name"]
                })

        # Check if it's time to upload a batch
        if len(documents_to_upload) >= BATCH_SIZE:
            upload_batch()

    # Upload any remaining documents
    upload_batch()
    
    print("\nGitHub data ingestion complete.")

if __name__ == "__main__":
    asyncio.run(ingest_github_data())