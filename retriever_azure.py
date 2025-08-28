# retriever.py
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from sentence_transformers import SentenceTransformer
import ssl
import certifi

# --- SSL Configuration for Corporate Networks ---
# This is crucial for environments with SSL inspection.
CERTIFICATE_PATH = r"trusted_certs.crt"
if os.path.exists(CERTIFICATE_PATH):
    print(f"Found certificate bundle at: {CERTIFICATE_PATH}")
    os.environ['REQUESTS_CA_BUNDLE'] = CERTIFICATE_PATH
else:
    print("Warning: Custom certificate bundle not found. Using default system certificates.")

def create_ssl_context():
    context = ssl.create_default_context(cafile=certifi.where())
    if os.path.exists(CERTIFICATE_PATH):
        context.load_verify_locations(CERTIFICATE_PATH)
    return context
# --- Configuration ---
load_dotenv()
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
# IMPORTANT: Use a Query Key for retrieving data, not an Admin Key.
AZURE_SEARCH_QUERY_KEY = os.getenv("AZURE_SEARCH_QUERY_KEY") 
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")

class HybridRetriever:
    """
    Manages semantic (vector) and keyword (full-text) search using Azure AI Search.
    """
    def __init__(self):
        self.index_name = AZURE_SEARCH_INDEX_NAME
        
        # Initialize the SearchClient with a Query Key for security
        credential = AzureKeyCredential(AZURE_SEARCH_QUERY_KEY)
        self.search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT, 
            index_name=self.index_name, 
            credential=credential,
            verify=ssl.create_default_context(cafile=certifi.where())
        )
        
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_PATH)

    def semantic_search(self, query: str, limit: int = 5) -> list:
        """Performs a pure vector search."""
        query_vector = self.encoder.encode(query).tolist()
        
        vector_query = VectorizedQuery(
            vector=query_vector, 
            k_nearest_neighbors=limit, 
            fields="vector"
        )

        results = self.search_client.search(
            search_text=None,  # No keyword search
            vector_queries=[vector_query],
            select=["id", "text", "source", "project"] # Specify which fields to return
        )
        
        # The SDK returns an iterator, so we convert it to a list
        return [result for result in results]

    def keyword_search(self, query: str, limit: int = 5) -> list:
        """Performs a pure keyword (full-text) search."""
        results = self.search_client.search(
            search_text=query,
            vector_queries=None, # No vector search
            top=limit,
            select=["id", "text", "source", "project"]
        )
        return [result for result in results]

    def hybrid_search(self, query: str, limit: int = 10) -> list:
        """
        Performs a hybrid search by sending both text and vector queries
        to Azure AI Search, which handles the result fusion.
        """
        query_vector = self.encoder.encode(query).tolist()
        
        vector_query = VectorizedQuery(
            vector=query_vector, 
            k_nearest_neighbors=limit, 
            fields="vector"
        )
        
        # This single call performs the hybrid search.
        # Azure AI Search automatically merges the results from the text and vector queries.
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=limit,
            select=["id", "text", "source", "project"]
        )
        
        return [result for result in results]

# --- Example Usage ---
if __name__ == '__main__':
    # Initialize the retriever
    retriever = HybridRetriever()

    # Define a query
    user_query = "What is the sourceWidget in Fireball framework?"

    print("--- 1. Performing Hybrid Search ---")
    hybrid_results = retriever.hybrid_search(user_query, limit=3)
    for result in hybrid_results:
        # The '@search.score' contains the RRF score from the hybrid search
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