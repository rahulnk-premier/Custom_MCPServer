# mcp_app.py
import os
import asyncio
import logging
import re
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastmcp import FastMCP
from dotenv import load_dotenv
import httpx

# For working with Chroma
import chromadb
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# --- NEW: Import our image analysis utility ---
from utils.image_analyzer import generate_image_summaries_from_incident

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Config ---
CHROMA_DB_PATH = r"E:\rnakka\Documents\AI Agent\repo_db"
CHROMA_COLLECTION_NAME = "github_repositories"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_RESULTS = 5
CERTIFICATE_PATH = r"E:\rnakka\Downloads\trusted_certs.crt"
import ssl
import certifi

def create_ssl_context():
    """Create SSL context with corporate certificates"""
    context = ssl.create_default_context(cafile=certifi.where())
    
    if os.path.exists(CERTIFICATE_PATH):
        context.load_verify_locations(CERTIFICATE_PATH)
    
    return context

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastMCP):
    """
    Initialize ChromaDB and embedding model on startup.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        
        try:
            os.environ["REQUESTS_CA_BUNDLE"] = CERTIFICATE_PATH
            os.environ["CURL_CA_BUNDLE"] = CERTIFICATE_PATH
            model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as ssl_error:
            logger.warning(f"SSL error with corporate cert: {ssl_error}")
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            os.environ["CURL_CA_BUNDLE"] = ""
            os.environ["REQUESTS_CA_BUNDLE"] = ""
            
            model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        
        app.chroma_client = client
        app.chroma_collection = collection
        app.embed_model = model
        
        logger.info("ChromaDB and embedding model initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Error initializing resources: {e}", exc_info=True)
        raise
    finally:
        logger.info("Agent MCP server shutting down")

# Initialize MCP server
mcp = FastMCP(name="AgentIncidentFixer", lifespan=lifespan)

# --- Tools ---

# _fetch_azure_incident_details and get_azure_incident_details can be removed
# as their logic is now fully contained within the main analysis tool.
# Keeping the helper makes the code cleaner.
async def _fetch_azure_incident_details(incident_id: int) -> Dict[str, Any]:
    """
    Internal helper to retrieve details from Azure DevOps Boards.
    """
    try:
        pat = os.getenv("MICROSOFT_TOKEN")
        if not pat:
            raise ValueError("MICROSOFT_TOKEN environment variable is not set.")

        api_url = (
            f"https://dev.azure.com/premierinc/Fireball/_apis/wit/workitems/{incident_id}"
            "?$expand=all&api-version=7.1"
        )
        ssl_context = create_ssl_context()

        async with httpx.AsyncClient(verify=ssl_context) as client:
            response = await client.get(api_url, auth=("", pat))
            response.raise_for_status()
            data = response.json()
        
        fields = data.get("fields", {})
        
        # NOTE: We return the raw HTML description for parsing later
        return {
            "title": fields.get("System.Title", "Title not found."),
            "work_item_type": fields.get("System.WorkItemType", "Work Item Type not found."),
            "description": fields.get("System.Description", "No description provided."), # This contains HTML
            "resolution": fields.get("Microsoft.VSTS.TCM.ReproSteps", "No resolution provided."), # Adjusted field for better info
            "state": fields.get("System.State", "State not found."),
            "assigned_to": fields.get("System.AssignedTo", {}).get("displayName", "Unassigned") if fields.get("System.AssignedTo") else "Unassigned"
        }

    except Exception as e:
        logger.error(f"Error getting Azure incident details for ID '{incident_id}': {e}", exc_info=True)
        return {"error": f"Could not retrieve incident details: {str(e)}"}

async def find_relevant_repositories(description: str, n_results: int = 3) -> Dict[str, Any]:
    """
    Query ChromaDB to find most relevant GitHub repositories based on description.
    """
    try:
        model = mcp.embed_model
        collection = mcp.chroma_collection
        
        query_embedding = await asyncio.to_thread(model.encode, description)
        
        results = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 2,
            include=["metadatas", "distances"]
        )
        
        if not results or not results.get('metadatas') or not results['metadatas'][0]:
            return {
                "recommended_repositories": [],
                "message": "No relevant repositories found in the database."
            }
        
        repo_scores = defaultdict(list)
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            repo_name = metadata['repo_name']
            similarity_score = 1.0 - distance
            repo_scores[repo_name].append(similarity_score)
        
        final_scores = {
            repo: sum(scores) / len(scores) 
            for repo, scores in repo_scores.items()
        }
        
        sorted_repos = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        
        return {
            "recommended_repositories": [
                {
                    "name": repo,
                    "full_name": f"PremierInc/{repo}",
                    "confidence_score": round(score, 3)
                }
                for repo, score in sorted_repos
            ],
            "search_metadata": {
                "total_candidates_evaluated": len(repo_scores),
                "embedding_model": EMBEDDING_MODEL,
                "query_length": len(description)
            }
        }
        
    except Exception as e:
        logger.error(f"Error finding relevant repositories: {e}", exc_info=True)
        return {
            "recommended_repositories": [],
            "error": f"Search failed: {str(e)}"
        }

# --- MODIFIED: Main analysis tool with integrated vision capability ---
@mcp.tool()
async def analyze_incident_and_suggest_repositories(incident_id: int) -> Dict[str, Any]:
    """
    FIRST STEP ONLY: Get incident details, analyze text and IMAGES, and suggest relevant repositories.
    
    IMPORTANT WORKFLOW REQUIREMENTS:
    1. This tool is the only FIRST STEP in the incident resolution workflow.
    2. After running this, present the consolidated summary and repository suggestions to the user.
    3. You MUST then ask the user to confirm the repository selection.
    
    Args:
        incident_id (int): Azure DevOps incident ID
        
    Returns:
        Combined incident details, image analysis, and repository suggestions.
    """
    # 1. Get incident details from Azure
    incident_details = await _fetch_azure_incident_details(incident_id)
    if "error" in incident_details:
        return {"error": "Could not retrieve incident details", "details": incident_details}
    
    pat = os.getenv("MICROSOFT_TOKEN")
    if not pat:
        return {"error": "MICROSOFT_TOKEN environment variable not found.", "details": ""}

    # 2. --- NEW: Analyze images from the description ---
    logger.info(f"Analyzing images for incident #{incident_id}...")
    image_summaries = await generate_image_summaries_from_incident(
        html_description=incident_details.get('description', ''),
        incident_title=incident_details.get('title', ''),
        pat=pat
    )
    logger.info(f"Image analysis complete. Summary: {image_summaries}")

    # 3. Clean the text description for display and search
    raw_description = incident_details.get('description', '')
    clean_description = re.sub(r'<[^>]+>', ' ', raw_description)
    clean_description = re.sub(r'\s+', ' ', clean_description).strip()

    # 4. --- NEW: Create the consolidated summary for a better search ---
    consolidated_summary = (
        f"Title: {incident_details.get('title', '')}\n"
        f"Description: {clean_description}\n"
        f"Image Analysis: {image_summaries if image_summaries else 'No images found.'}"
    )

    # 5. Find relevant repositories using the enhanced summary
    repo_suggestions = await find_relevant_repositories(consolidated_summary)
    
    # 6. Return the comprehensive result
    return {
        "incident_id": incident_id,
        "incident_details": {
            "title": incident_details.get('title'),
            "state": incident_details.get('state'),
            "assigned_to": incident_details.get('assigned_to'),
        },
        "consolidated_summary": consolidated_summary, # Return this for the LLM to use
        "repository_suggestions": repo_suggestions,
        "workflow_step": "incident_analyzed",
        "next_action": "Please present the consolidated summary and repository suggestions, then ask the user to confirm which repository to analyze."
    }

@mcp.tool()
async def confirm_repository_selection(incident_id: int, selected_repo: str, user_feedback: str = "") -> Dict[str, Any]:
    """
    Confirm user's repository selection and prepare for code analysis.
    
    Args:
        incident_id (int): The incident ID
        selected_repo (str): User-selected repository name
        user_feedback (str): Optional user feedback
        
    Returns:
        Confirmation and next steps
    """
    if "/" not in selected_repo:
        selected_repo = f"PremierInc/{selected_repo}"
    
    return {
        "incident_id": incident_id,
        "confirmed_repository": selected_repo,
        "user_feedback": user_feedback,
        "workflow_step": "repository_confirmed",
        "ready_for_code_analysis": True,
        "next_actions": [
            "Use GitHub MCP tools to explore repository structure",
            "Search for files related to the incident",
            "Analyze code patterns and suggest fixes"
        ],
        "message": f"Repository '{selected_repo}' confirmed. Ready to analyze codebase for incident #{incident_id}."
    }

# --- Prompts ---
# (No changes needed for prompts, they are generic enough to handle the improved tool output)
@mcp.prompt("incident_analysis")
async def incident_analysis_prompt(incident_id: int) -> str:
    """
    System prompt for incident analysis workflow.
    """
    return f"""You are an expert AI incident resolution assistant. Your goal is to help analyze and fix code issues from Azure DevOps incidents.

WORKFLOW FOR INCIDENT #{incident_id}:

1. **Incident Analysis Phase**:
   - Use `analyze_incident_and_suggest_repositories({incident_id})` to get incident details, image analysis, and repository suggestions.
   - Present the **consolidated summary** (including the image analysis) clearly to the user.
   - Show the top repository suggestions with confidence scores.
   - Explain that the suggestions are based on both the text and the content of any images found.

2. **Repository Selection Phase**:
   - Ask the user to confirm which repository is correct.
   - Accept user input if they want to specify a different repository.
   - Use `confirm_repository_selection()` once the user provides confirmation.
   - Do NOT proceed to code analysis until the repository is confirmed.

3. **Code Analysis Phase** (only after repository confirmation):
   - Use GitHub MCP tools to explore the confirmed repository.
   - Look for files and code patterns related to the incident.
   - Analyze the codebase systematically.

4. **Solution Phase**:
   - Provide specific code suggestions to fix the issue.
   - Explain the reasoning behind each suggestion.

IMPORTANT GUIDELINES:
- Always wait for user confirmation before analyzing code.
- Be specific and actionable in your suggestions.
- Your first step is ALWAYS to call `analyze_incident_and_suggest_repositories`.

Start by analyzing incident #{incident_id}."""


# --- Main execution ---
async def main():
    """Run the MCP server."""
    try:
        await mcp.run_async(transport="streamable-http", host="127.0.0.1", port=9002, path="/mcp")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())