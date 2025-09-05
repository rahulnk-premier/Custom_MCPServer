"""
Phase 1: Work Item Analysis Server
Clean, organized MCP server with tools only - business logic moved to utils.
Handles Azure DevOps work item analysis with image processing and documentation search.
"""

import os
import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context

# Import utilities and constants
from utils.phase1_helpers import (
    create_consolidated_analysis,
    fetch_azure_work_item_details,
    search_ado_work_items_by_text
)
from constants.docstrings import (
    GET_WORK_ITEM_DETAILS_DOCSTRING,
    SEARCH_DOCUMENTATION_DOCSTRING,
    GET_PHASE1_SUMMARY_DOCSTRING
)
from retriever_azure import HybridRetriever

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class WorkItemState:
    """Maintain state across Phase 1 tools"""
    input_type: str  # "single_id" or "natural_language"
    original_input: str
    work_items: List[Dict] = None
    summary: Optional[str] = None
    relevant_docs: List[Dict] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Initialize and manage the lifecycle of Azure clients."""
    logger.info("Phase 1 MCP server starting up...")
    
    try:
        # Initialize HybridRetriever for documentation search
        logger.info("Initializing HybridRetriever...")
        app.retriever = HybridRetriever()
        
        logger.info("Phase 1 server is fully operational.")
        yield
        
    except Exception as e:
        logger.error(f"FATAL: Error during Phase 1 server startup: {e}", exc_info=True)
        raise
    finally:
        logger.info("Phase 1 MCP server shutting down.")


# Create FastMCP server
mcp = FastMCP("Phase1 Work Item Analyzer", lifespan=lifespan)

# Configure server settings
mcp.settings.host = "127.0.0.1"
mcp.settings.port = 9003
mcp.settings.mount_path = "/mcp"

# Global state storage
work_item_sessions: Dict[str, WorkItemState] = {}


@mcp.tool()
async def get_work_item_details(
    user_input: str, 
    session_id: str = "default",
    ctx: Context = None
) -> Dict[str, Any]:
    GET_WORK_ITEM_DETAILS_DOCSTRING
    await ctx.info(f"Phase 1: Processing work item input and creating analysis: {user_input}")
    
    try:
        # Determine input type
        input_type = "single_id" if re.match(r'^-?\d+$', user_input.strip()) else "natural_language"
        
        # Initialize session state
        state = WorkItemState(
            input_type=input_type,
            original_input=user_input
        )
        
        await ctx.report_progress(0.2, 1.0, "Fetching work item(s)...")
        
        if input_type == "single_id":
            work_item_id = user_input.strip().lstrip('-')
            await ctx.info(f"Fetching single work item: {work_item_id}")
            
            work_item = await fetch_azure_work_item_details(int(work_item_id))
            state.work_items = [work_item] if work_item and "error" not in work_item else []
            
        else:
            await ctx.info(f"Searching ADO work items for: {user_input}")
            work_items = await search_ado_work_items_by_text(user_input)
            state.work_items = work_items
        
        if not state.work_items:
            return {
                "status": "error",
                "error": "No work items found",
                "session_id": session_id,
                "suggested_action": "Check the work item ID or refine your natural language query"
            }
        
        await ctx.report_progress(0.5, 1.0, "Analyzing images and creating consolidated summary...")
        
        # Create consolidated analysis automatically
        consolidated_analysis = await create_consolidated_analysis(state.work_items, ctx)
        state.summary = json.dumps(consolidated_analysis, indent=2)
        
        # Store session state
        work_item_sessions[session_id] = state
        
        await ctx.report_progress(1.0, 1.0, f"Analysis complete: {len(state.work_items)} work item(s) processed")
        
        return {
            "status": "success",
            "session_id": session_id,
            "input_type": input_type,
            "original_input": user_input,
            "work_items_count": len(state.work_items),
            "consolidated_analysis": consolidated_analysis,
            "next_steps": {
                "step_1": "Call search_documentation with specific search terms based on the analysis",
                "step_2": "Call get_phase1_summary for final overview"
            },
            "metadata": {
                "created_at": state.created_at,
                "phase": "1-work-item-analysis-complete",
                "includes_image_analysis": consolidated_analysis.get("image_analysis_included", False)
            }
        }
        
    except Exception as e:
        await ctx.error(f"Error processing work item input: {str(e)}")
        return {
            "status": "error", 
            "error": str(e),
            "session_id": session_id,
            "suggested_action": "Check the work item ID or refine your natural language query"
        }


@mcp.tool()
async def search_documentation(
    session_id: str = "default",
    search_query: str = None,
    ctx: Context = None
) -> Dict[str, Any]:
    SEARCH_DOCUMENTATION_DOCSTRING
    await ctx.info(f"Phase 1: Searching documentation for session: {session_id}")
    
    if not search_query:
        raise ValueError("search_query parameter is required. Please provide a specific search query for documentation.")
    
    try:
        # Get session state
        if session_id not in work_item_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        state = work_item_sessions[session_id]
        
        await ctx.info(f"Using search query: {search_query}")
        await ctx.report_progress(0.3, 1.0, "Executing documentation search...")
        
        # Use HybridRetriever
        retriever: HybridRetriever = mcp.retriever
        search_results = await asyncio.to_thread(retriever.hybrid_search, search_query, limit=10)
        
        await ctx.report_progress(0.8, 1.0, "Processing search results...")
        
        # Structure the results
        relevant_docs = []
        for result in search_results:
            doc_info = {
                "title": result.get("source", "Unknown Document"),
                "content": result.get("text", ""),
                "project": result.get("project", "Unknown"),
                "relevance_score": result.get("@search.score", 0),
                "document_id": result.get("id", ""),
                "snippet": result.get("text", "")[:300] + "..." if len(result.get("text", "")) > 300 else result.get("text", "")
            }
            relevant_docs.append(doc_info)
        
        # Update session state
        state.relevant_docs = relevant_docs
        work_item_sessions[session_id] = state
        
        await ctx.report_progress(1.0, 1.0, f"Found {len(relevant_docs)} relevant documents")
        
        return {
            "status": "success",
            "session_id": session_id,
            "search_query_used": search_query,
            "documents_found": len(relevant_docs),
            "relevant_documentation": relevant_docs,
            "search_metadata": {
                "search_method": "hybrid_search (vector + keyword)",
                "max_results": 10,
                "search_fields": ["text", "source", "project"]
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "phase": "1-documentation-search"
            },
            "next_action": "Call get_phase1_summary for final overview"
        }
        
    except Exception as e:
        await ctx.error(f"Error in documentation search: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "session_id": session_id
        }


@mcp.tool()
async def get_phase1_summary(
    session_id: str = "default",
    ctx: Context = None
) -> Dict[str, Any]:
    GET_PHASE1_SUMMARY_DOCSTRING
    await ctx.info(f"Phase 1: Generating complete summary for session: {session_id}")
    
    try:
        if session_id not in work_item_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        state = work_item_sessions[session_id]
        
        # Assess completeness
        work_items_ready = bool(state.work_items and len(state.work_items) > 0)
        documentation_ready = bool(state.relevant_docs and len(state.relevant_docs) > 0)
        overall_ready = work_items_ready and documentation_ready
        
        # Create simple summary focused on key findings
        summary = {
            "session_info": {
                "session_id": session_id,
                "input_type": state.input_type,
                "original_input": state.original_input,
                "created_at": state.created_at,
                "analysis_completed_at": datetime.now().isoformat()
            },
            "work_items": {
                "status": "completed" if work_items_ready else "incomplete",
                "count": len(state.work_items) if state.work_items else 0,
                "items": state.work_items or [],
                "types_found": list(set([
                    item.get('fields', {}).get('System.WorkItemType', 'Unknown') 
                    for item in (state.work_items or [])
                ]))
            },
            "documentation": {
                "status": "completed" if documentation_ready else "incomplete", 
                "count": len(state.relevant_docs) if state.relevant_docs else 0,
                "documents": state.relevant_docs or [],
                "documentation_summary": "",  # Will be filled by LLM
                "top_sources": []  # Will be filled by LLM
            },
            "phase1_complete": overall_ready,
            "missing_components": []
        }
        
        # Add missing components if any
        if not work_items_ready:
            summary["missing_components"].append("work_items")
            
        if not documentation_ready:
            summary["missing_components"].append("documentation")
        
        return {
            "status": "success",
            "phase1_complete": overall_ready,
            "summary": summary,
            "next_steps": [
                "Agent should summarize the work items and documentation findings",
                "Agent should list the top 3 documentation sources found for user reference",
                "Present complete findings to user for review",
                "Collect user feedback and proceed to GitHub analysis if user confirms"
            ],
            "llm_instructions": {
                "if_documentation_found": "Create a summary of documentation findings relevant to the work items context and list top 3 sources",
                "next_phase": "Present findings to user, collect feedback, then use GitHub MCP for repository analysis"
            }
        }
        
    except Exception as e:
        await ctx.error(f"Error generating Phase 1 summary: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "session_id": session_id
        }


if __name__ == "__main__":
    def main():
        """Run the Phase 1 MCP server with streamable HTTP transport."""
        try:
            mcp.run(transport="streamable-http")
        except Exception as e:
            logger.error(f"Error running Phase 1 MCP server: {e}", exc_info=True)

    main()
