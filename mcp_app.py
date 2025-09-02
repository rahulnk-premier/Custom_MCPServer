# mcp_app.py

import os
import asyncio
import logging
import re
from typing import Dict, Any
from contextlib import asynccontextmanager
from collections import defaultdict

from dotenv import load_dotenv
import httpx
from fastmcp import FastMCP

# --- Azure SDK Imports ---
from azure.core.credentials import AzureKeyCredential
# Use the ASYNC client for an asyncio application
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
# Import the Azure OpenAI client
from openai import AzureOpenAI

# --- Custom Module Imports (Now using Azure services) ---
from retriever_azure import HybridRetriever
from utils.image_analyzer import generate_image_summaries_from_incident

# --- Initial Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Azure OpenAI Config
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

import ssl
import certifi

# --- Application Lifespan: Initialize all clients on startup ---
@asynccontextmanager
async def lifespan(app: FastMCP):
    """
    Initialize and manage the lifecycle of Azure clients.
    """
    logger.info("MCP server starting up. Initializing Azure clients...")
    try:

        # 1. Initialize HybridRetriever (now uses Azure OpenAI internally)
        logger.info("Initializing HybridRetriever for Knowledge Base...")
        app.retriever = HybridRetriever()
        
        # 2. Initialize Azure OpenAI client for repository search embeddings
        logger.info("Initializing AzureOpenAI client...")
        app.openai_client = AzureOpenAI(
            api_version="2023-05-15",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY
        )

        # 3. Initialize Async SearchClient for repository search
        logger.info("Initializing async SearchClient for Repositories...")
        repo_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        repo_query_key = os.getenv("AZURE_SEARCH_QUERY_KEY") # Use a query key
        repo_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME_REPOS")
        
        if not all([repo_endpoint, repo_query_key, repo_index_name]):
            raise ValueError("Azure Search environment variables for repositories are not fully configured.")
        
        app.repo_search_client = SearchClient(
            endpoint=repo_endpoint,
            index_name=repo_index_name,
            credential=AzureKeyCredential(repo_query_key)
        )

        logger.info("All clients are ready. Server is fully operational.")
        yield
        
    except Exception as e:
        logger.error(f"FATAL: Error during server startup: {e}", exc_info=True)
        raise
    finally:
        logger.info("Agent MCP server shutting down.")
        # Gracefully close the async search client
        if hasattr(app, 'repo_search_client'):
            await app.repo_search_client.close()


mcp = FastMCP(name="CodeContextAIAgent", lifespan=lifespan)

def format_results(results: list) -> str:
    """Formats Azure AI Search results into a clean string."""
    if not results:
        return "No relevant documents found in the knowledge base."
    
    formatted = []
    for i, result in enumerate(results, 1):
        score_str = f"{result.get('@search.score', 0):.4f}"
        entry = (f"Result {i}:\n  - Source: {result.get('source', 'N/A')} (Project: {result.get('project', 'N/A')})\n"
                 f"  - Relevance Score: {score_str}\n  - Content Snippet: {result.get('text', '')}\n")
        formatted.append(entry)
    
    return "\n---\n".join(formatted)

# --- Knowledge Base Tools ---
@mcp.tool()
async def query_knowledge_base(query: str) -> str:
    """Performs a hybrid search on the internal knowledge base for general queries."""
    retriever: HybridRetriever = mcp.retriever
    # Run synchronous IO in a separate thread to avoid blocking the event loop
    results = await asyncio.to_thread(retriever.hybrid_search, query, limit=5)
    return format_results(results)

@mcp.tool()
async def find_incident_resolution_guidance(incident_description: str, project: str = None) -> str:
    """Specialized search for finding solutions to technical incidents."""
    enhanced_query = f"how to fix error: {incident_description} in project {project if project else ''}"
    retriever: HybridRetriever = mcp.retriever
    results = await asyncio.to_thread(retriever.hybrid_search, enhanced_query, limit=5)
    return format_results(results)

# --- Incident & Repository Tools ---
async def _fetch_azure_incident_details(incident_id: int) -> Dict[str, Any]:
    """Internal helper to retrieve details from Azure DevOps Boards."""
    try:
        pat = os.getenv("MICROSOFT_TOKEN")
        if not pat: raise ValueError("MICROSOFT_TOKEN not set.")
        api_url = f"https://dev.azure.com/premierinc/Fireball/_apis/wit/workitems/{incident_id}?$expand=all&api-version=7.1"
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, auth=("", pat))
            response.raise_for_status()
            fields = response.json().get("fields", {})
        return {
            "title": fields.get("System.Title", "N/A"),
            "description": fields.get("System.Description", ""),
            "state": fields.get("System.State", "N/A"),
            "assigned_to": fields.get("System.AssignedTo", {}).get("displayName", "Unassigned")
        }
    except Exception as e:
        logger.error(f"Error fetching Azure incident {incident_id}: {e}", exc_info=True)
        return {"error": f"Could not retrieve incident details: {e}"}

async def find_relevant_repositories(description: str, n_results: int = 3) -> Dict[str, Any]:
    """Query Azure AI Search to find relevant GitHub repositories using Azure OpenAI embeddings."""
    try:
        openai_client: AzureOpenAI = mcp.openai_client
        search_client: SearchClient = mcp.repo_search_client
        
        embedding_response = await asyncio.to_thread(
            openai_client.embeddings.create,
            input=description,
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        query_vector = embedding_response.data[0].embedding
        
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=15, fields="vector")

        results_iterator = await search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["project"]  # <-- THE FIX IS APPLIED HERE
        )
        
        repo_scores = defaultdict(list)
        async for result in results_iterator:
            repo_name = result.get("project")
            score = result.get("@search.score") 
            if repo_name and score:
                repo_scores[repo_name].append(score)

        if not repo_scores:
            return {"recommended_repositories": [], "message": "No relevant repositories found."}

        final_scores = {repo: sum(scores) / len(scores) for repo, scores in repo_scores.items()}
        sorted_repos = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]

        return {
            "recommended_repositories": [
                {"name": repo, "full_name": f"PremierInc/{repo}", "confidence_score": round(score, 3)}
                for repo, score in sorted_repos
            ],
            "search_metadata": {"total_candidates_evaluated": len(repo_scores)}
        }
    except Exception as e:
        logger.error(f"Error in find_relevant_repositories: {e}", exc_info=True)
        return {"recommended_repositories": [], "error": f"Repository search failed: {e}"}


@mcp.tool()
async def analyze_incident_and_suggest_repositories(incident_id: int) -> Dict[str, Any]:
    """FIRST STEP: Get details, analyze text/images, and suggest repositories."""
    incident_details = await _fetch_azure_incident_details(incident_id)
    if "error" in incident_details:
        return incident_details
    
    pat = os.getenv("MICROSOFT_TOKEN")
    if not pat: return {"error": "MICROSOFT_TOKEN not found."}
    
    # Calls the updated util that uses Azure Vision
    image_summaries = await generate_image_summaries_from_incident(
        html_description=incident_details.get('description', ''),
        incident_title=incident_details.get('title', ''),
        pat=pat
    )
    
    clean_description = re.sub(r'<[^>]+>', ' ', incident_details.get('description', ''))
    consolidated_summary = (f"Title: {incident_details.get('title')}\n"
                            f"Description: {clean_description.strip()}\n"
                            f"Image Analysis: {image_summaries if image_summaries else 'No images found.'}")
    print(consolidated_summary)
    # Calls the updated function that uses Azure OpenAI
    repo_suggestions = await find_relevant_repositories(consolidated_summary)
    
    return {
        "incident_id": incident_id,
        "incident_details": {k: incident_details[k] for k in ['title', 'state', 'assigned_to']},
        "consolidated_summary": consolidated_summary,
        "repository_suggestions": repo_suggestions,
        "workflow_step": "incident_analyzed",
        "next_action": "Present summary and suggestions, ask user to confirm repository."
    }

@mcp.tool()
async def confirm_repository_selection(incident_id: int, selected_repo: str, user_feedback: str = "") -> Dict[str, Any]:
    """Confirm user's repository selection and prepare for code analysis."""
    if "/" not in selected_repo:
        selected_repo = f"PremierInc/{selected_repo}"
    return {
        "incident_id": incident_id, "confirmed_repository": selected_repo, "user_feedback": user_feedback,
        "workflow_step": "repository_confirmed", "ready_for_code_analysis": True,
        "next_actions": ["Use GitHub tools", "Search for files", "Suggest fixes"],
        "message": f"Repository '{selected_repo}' confirmed. Ready to analyze codebase."
    }

# --- System Prompt ---
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

# --- Main Execution ---
async def main():
    """Run the MCP server."""
    try:
        # Bind to 0.0.0.0 to be accessible within a container or from other machines
        await mcp.run_async(transport="streamable-http", host="127.0.0.1", port=9003, path="/mcp")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())