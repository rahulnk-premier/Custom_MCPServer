# mcp_app.py

from retriever_azure import HybridRetriever
import os
import asyncio
import logging
import re
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
from collections import defaultdict

### --- CHANGE: Azure AI Search imports for repository search --- ###
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

### --- CHANGE: ChromaDB is no longer used --- ###
# import chromadb 
from sentence_transformers import SentenceTransformer

from utils.image_analyzer import generate_image_summaries_from_incident

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


### --- CHANGE: Using the BGE model for repo search as well --- ###
EMBEDDING_MODEL_REPOS = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
CERTIFICATE_PATH = r"trusted_certs.crt"
import ssl
import certifi

def create_ssl_context():
    """Create SSL context with corporate certificates"""
    context = ssl.create_default_context(cafile=certifi.where())
    if os.path.exists(CERTIFICATE_PATH):
        context.load_verify_locations(CERTIFICATE_PATH)
    return context

### --- CHANGE: Simplified lifespan function --- ###
@asynccontextmanager
async def lifespan(app: FastMCP):
    """
    Initialize the HybridRetriever for the Azure AI Search knowledge base.
    The repository search client will be initialized on-demand.
    """
    logger.info("MCP server starting up. Initializing resources...")
    try:
        os.environ["REQUESTS_CA_BUNDLE"] = CERTIFICATE_PATH
        os.environ["CURL_CA_BUNDLE"] = CERTIFICATE_PATH
        
        # Initialize the HybridRetriever for the main knowledge base
        logger.info("Initializing HybridRetriever for Azure AI Search (Knowledge Base)...")
        retriever = HybridRetriever()
        
        # Store the retriever directly on the app instance
        app.retriever = retriever
        logger.info("HybridRetriever is ready. Server is fully operational.")

        yield
        
    except Exception as e:
        logger.error(f"Error during server startup: {e}", exc_info=True)
        raise
    finally:
        logger.info("Agent MCP server shutting down.")

mcp = FastMCP(name="CodeContextAIAgent", lifespan=lifespan)

def format_results(results: list) -> str:
    """Formats Azure AI Search results into a clean string for the agent."""
    if not results:
        return "No relevant documents found in the knowledge base."
    
    formatted = []
    for i, result in enumerate(results, 1):
        score = result.get('@search.score', 'N/A')
        source = result.get('source', 'Unknown')
        project = result.get('project', 'General')
        content = result.get('text', 'N/A')
        score_str = f"{score:.4f}" if isinstance(score, float) else score
        entry = (f"Result {i}:\n  - Source: {source} (Project: {project})\n  - Relevance Score: {score_str}\n  - Content Snippet: {content}\n")
        formatted.append(entry)
    
    return "\n---\n".join(formatted)

# --- Knowledge Base Tools ---
@mcp.tool()
async def query_knowledge_base(query: str) -> str:
    """Performs a hybrid search on the internal knowledge base for general queries."""
    retriever: HybridRetriever = mcp.retriever
    results = retriever.hybrid_search(query, limit=5)
    return format_results(results)

@mcp.tool()
async def find_incident_resolution_guidance(incident_description: str, project: str = None) -> str:
    """Specialized search for finding solutions to technical incidents."""
    enhanced_query = f"how to fix error: {incident_description} in project {project if project else ''}"
    retriever: HybridRetriever = mcp.retriever
    results = retriever.hybrid_search(enhanced_query, limit=5)
    return format_results(results)

# --- Incident & Repository Tools ---
async def _fetch_azure_incident_details(incident_id: int) -> Dict[str, Any]:
    """Internal helper to retrieve details from Azure DevOps Boards."""
    # (This function is correct, no changes needed)
    try:
        pat = os.getenv("MICROSOFT_TOKEN")
        if not pat:
            raise ValueError("MICROSOFT_TOKEN environment variable is not set.")
        api_url = (f"https://dev.azure.com/premierinc/Fireball/_apis/wit/workitems/{incident_id}?$expand=all&api-version=7.1")
        ssl_context = create_ssl_context()
        async with httpx.AsyncClient(verify=ssl_context) as client:
            response = await client.get(api_url, auth=("", pat))
            response.raise_for_status()
            data = response.json()
        fields = data.get("fields", {})
        return {
            "title": fields.get("System.Title", "Title not found."),
            "work_item_type": fields.get("System.WorkItemType", "Work Item Type not found."),
            "description": fields.get("System.Description", "No description provided."),
            "resolution": fields.get("Microsoft.VSTS.TCM.ReproSteps", "No resolution provided."),
            "state": fields.get("System.State", "State not found."),
            "assigned_to": fields.get("System.AssignedTo", {}).get("displayName", "Unassigned") if fields.get("System.AssignedTo") else "Unassigned"
        }
    except Exception as e:
        logger.error(f"Error getting Azure incident details for ID '{incident_id}': {e}", exc_info=True)
        return {"error": f"Could not retrieve incident details: {str(e)}"}

### --- CHANGE: Rewritten function to use Azure AI Search instead of ChromaDB --- ###
# In mcp_app.py, replace the entire find_relevant_repositories function with this one.

async def find_relevant_repositories(description: str, n_results: int = 3) -> Dict[str, Any]:
    """
    Query Azure AI Search to find the most relevant GitHub repositories based on an incident description.
    """
    try:
        # 1. Initialize Azure Search client and embedding model
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        query_key = os.getenv("AZURE_SEARCH_QUERY_KEY")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME_REPOS")
        
        if not all([endpoint, query_key, index_name]):
            raise ValueError("Azure Search environment variables for repositories are not fully configured.")

        credential = AzureKeyCredential(query_key)
        
        ### --- FIX 2: Reverting to the robust SSL context helper --- ###
        search_client = SearchClient(
            endpoint=endpoint, 
            index_name=index_name, 
            credential=credential,
            verify=CERTIFICATE_PATH 
        )

        encoder = SentenceTransformer(EMBEDDING_MODEL_PATH)

        # 2. Create embedding for the incident description
        query_vector = encoder.encode(description).tolist()
        
        vector_query = VectorizedQuery(
            vector=query_vector, 
            k_nearest_neighbors=15, # Fetch more results to allow for better aggregation
            fields="vector"
        )

        # 3. Perform the vector search
        ### --- FIX 1: Removed "@search.score" from the select list --- ###
        results_iterator = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["project"] # Only select the project field. Score is returned automatically.
        )
        
        # 4. Aggregate scores by repository
        repo_scores = defaultdict(list)
        for result in results_iterator:
            repo_name = result.get("project")
            # The score is still available in the result dictionary
            score = result.get("@search.score") 
            if repo_name and score:
                repo_scores[repo_name].append(score)

        if not repo_scores:
            return {"recommended_repositories": [], "message": "No relevant repositories found."}

        # 5. Calculate average scores and rank repositories
        final_scores = {repo: sum(scores) / len(scores) for repo, scores in repo_scores.items()}
        sorted_repos = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]

        return {
            "recommended_repositories": [
                {
                    "name": repo,
                    "full_name": f"PremierInc/{repo}", # Assumes organization name
                    "confidence_score": round(score, 3)
                }
                for repo, score in sorted_repos
            ],
            "search_metadata": { "total_candidates_evaluated": len(repo_scores) }
        }
        
    except Exception as e:
        logger.error(f"Error finding relevant repositories in Azure AI Search: {e}", exc_info=True)
        return {"recommended_repositories": [], "error": f"Repository search failed: {str(e)}"}

@mcp.tool()
async def analyze_incident_and_suggest_repositories(incident_id: int) -> Dict[str, Any]:
    """FIRST STEP ONLY: Get incident details, analyze text/images, and suggest repositories."""
    # (This function is correct, no changes needed as it calls the rewritten find_relevant_repositories)
    incident_details = await _fetch_azure_incident_details(incident_id)
    if "error" in incident_details:
        return {"error": "Could not retrieve incident details", "details": incident_details}
    pat = os.getenv("MICROSOFT_TOKEN")
    if not pat:
        return {"error": "MICROSOFT_TOKEN not found.", "details": ""}
    image_summaries = await generate_image_summaries_from_incident(
        html_description=incident_details.get('description', ''),
        incident_title=incident_details.get('title', ''),
        pat=pat
    )
    raw_description = incident_details.get('description', '')
    clean_description = re.sub(r'<[^>]+>', ' ', raw_description)
    clean_description = re.sub(r'\s+', ' ', clean_description).strip()
    consolidated_summary = (f"Title: {incident_details.get('title', '')}\nDescription: {clean_description}\nImage Analysis: {image_summaries if image_summaries else 'No images found.'}")
    repo_suggestions = await find_relevant_repositories(consolidated_summary)
    return {
        "incident_id": incident_id,
        "incident_details": {"title": incident_details.get('title'), "state": incident_details.get('state'), "assigned_to": incident_details.get('assigned_to')},
        "consolidated_summary": consolidated_summary,
        "repository_suggestions": repo_suggestions,
        "workflow_step": "incident_analyzed",
        "next_action": "Present summary and suggestions, then ask user to confirm repository."
    }

@mcp.tool()
async def confirm_repository_selection(incident_id: int, selected_repo: str, user_feedback: str = "") -> Dict[str, Any]:
    """Confirm user's repository selection and prepare for code analysis."""
    # (This function is correct, no changes needed)
    if "/" not in selected_repo:
        selected_repo = f"PremierInc/{selected_repo}"
    return {
        "incident_id": incident_id,
        "confirmed_repository": selected_repo,
        "user_feedback": user_feedback,
        "workflow_step": "repository_confirmed",
        "ready_for_code_analysis": True,
        "next_actions": ["Use GitHub tools to explore repository", "Search for files", "Suggest fixes"],
        "message": f"Repository '{selected_repo}' confirmed. Ready to analyze codebase for incident #{incident_id}."
    }

# --- PROMPT IS STILL VALID --- ###
@mcp.prompt("incident_analysis")
async def incident_analysis_prompt(incident_id: int) -> str:
    """System prompt for incident analysis workflow."""
    return f"""You are an expert AI incident resolution assistant. Your goal is to help analyze and fix code issues from Azure DevOps incidents. You have access to a specialized internal knowledge base.

**AVAILABLE KNOWLEDGE BASE TOOLS:**
- `query_knowledge_base(query: str)`: Use for general questions about documentation, code standards, or architecture.
- `find_incident_resolution_guidance(incident_description: str, project: str)`: Use to find existing solutions, runbooks, or post-mortems for a specific technical problem.

**WORKFLOW FOR INCIDENT #{incident_id}:**

1.  **Incident Analysis Phase**:
    *   Call `analyze_incident_and_suggest_repositories({incident_id})` to get initial details.
    *   Present the consolidated summary to the user.
    *   **CRITICAL:** Before suggesting repositories or solutions, STOP and THINK. Does the incident mention a specific technology, internal framework, or error code? If so, use `find_incident_resolution_guidance()` to check for a known solution. This can often solve the problem immediately.

2.  **Repository Selection Phase**:
    *   Show top repository suggestions. Ask user to confirm. Use `confirm_repository_selection()` once confirmed.

3.  **Code Analysis Phase** (after repository confirmation):
    *   Explore the repository using GitHub tools.
    *   **If you encounter unfamiliar code or logic, use `query_knowledge_base()` to get context from documentation before making assumptions.** This is your primary tool for understanding "how things work here."

4.  **Solution Phase**:
    *   Provide specific code suggestions based on your analysis AND knowledge base information.
    *   Explain your reasoning, citing both code and documentation.

**IMPORTANT GUIDELINES:**
- **Leverage the knowledge base first.** Proactively search for existing solutions.
- Your first step is ALWAYS `analyze_incident_and_suggest_repositories`.
- Your second step is OFTEN a knowledge base tool.

Start by analyzing incident #{incident_id}."""

# --- Main execution ---
async def main():
    """Run the MCP server."""
    try:
        await mcp.run_async(transport="streamable-http", host="0.0.0.0", port=9002, path="/mcp")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())