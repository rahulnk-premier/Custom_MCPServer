# retriever_azure.py
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
import ssl
import certifi

# --- Configuration ---
load_dotenv()
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_QUERY_KEY = os.getenv("AZURE_SEARCH_QUERY_KEY") 
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# --- NEW: Azure OpenAI Configuration ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")



class HybridRetriever:
    """
    Manages hybrid search (vector + keyword) using Azure AI Search
    and generates query embeddings using Azure OpenAI.
    """
    def __init__(self):
        if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_QUERY_KEY, AZURE_SEARCH_INDEX_NAME,
                    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_EMBEDDING_DEPLOYMENT]):
            raise ValueError("One or more required environment variables for Azure Search or OpenAI are missing.")

        self.index_name = AZURE_SEARCH_INDEX_NAME
        self.embedding_deployment = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        
        # 1. Initialize the Azure SearchClient with a Query Key
        search_credential = AzureKeyCredential(AZURE_SEARCH_QUERY_KEY)
        self.search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT, 
            index_name=self.index_name, 
            credential=search_credential,
        )
        
        # 2. NEW: Initialize the Azure OpenAI client for generating embeddings
        self.openai_client = AzureOpenAI(
            api_version="2023-05-15",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
        )

    def _get_embedding(self, text: str) -> list[float]:
        """Helper method to generate a vector embedding for a given text."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        return response.data[0].embedding

    def semantic_search(self, query: str, limit: int = 5) -> list:
        """Performs a pure vector search."""
        query_vector = self._get_embedding(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector, 
            k_nearest_neighbors=limit, 
            fields="vector"
        )

        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id", "text", "source", "project"]
        )
        
        return [result for result in results]

    def keyword_search(self, query: str, limit: int = 5) -> list:
        """Performs a pure keyword (full-text) search."""
        results = self.search_client.search(
            search_text=query,
            vector_queries=None,
            top=limit,
            select=["id", "text", "source", "project"]
        )
        return [result for result in results]

    def hybrid_search(self, query: str, limit: int = 10) -> list:
        """
        Performs a hybrid search using both a text query and a vector query.
        """
        query_vector = self._get_embedding(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector, 
            k_nearest_neighbors=limit, 
            fields="vector"
        )
        
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=limit,
            select=["id", "text", "source", "project"]
        )
        
        return [result for result in results]

# --- Example Usage (Should work exactly as before) ---
if __name__ == '__main__':
    try:
        retriever = HybridRetriever()
        user_query = "What is the sourceWidget in Fireball framework?"

        print("--- 1. Performing Hybrid Search (with Azure OpenAI Embeddings) ---")
        hybrid_results = retriever.hybrid_search(user_query, limit=3)
        for result in hybrid_results:
            print(f"Score: {result['@search.score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Text: {result['text'][:150]}...\n")
            
        print("\n--- 2. Performing Pure Semantic (Vector) Search ---")
        semantic_results = retriever.semantic_search(user_query, limit=2)
        for result in semantic_results:
            print(f"Score: {result['@search.score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Text: {result['text'][:150]}...\n")

        print("\n--- 3. Performing Pure Keyword Search ---")
        keyword_results = retriever.keyword_search(user_query, limit=2)
        for result in keyword_results:
            print(f"Score: {result['@search.score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Text: {result['text'][:150]}...\n")

    except ValueError as e:
        print(f"ERROR: Initialization failed. {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the test run: {e}")