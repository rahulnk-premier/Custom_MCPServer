# utils/azure_search_helpers.py

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)

def create_search_index(endpoint: str, key: str, index_name: str, embedding_dimensions: int):
    """
    Creates a new Azure AI Search index with a vector search configuration.
    """
    credential = AzureKeyCredential(key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

    try:
        index_client.get_index(index_name)
        print(f"Index '{index_name}' already exists.")
        return
    except Exception:
        print(f"Index '{index_name}' not found. Creating a new one...")

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="text", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchableField(name="source", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SearchableField(name="project", type=SearchFieldDataType.String, filterable=True, facetable=True, retrievable=True),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name="my-hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-hnsw-profile", algorithm_configuration_name="my-hnsw-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-config")],
    )
    
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="text")]
        )
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])


    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
    
    try:
        index_client.create_index(index)
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Failed to create index '{index_name}': {e}")