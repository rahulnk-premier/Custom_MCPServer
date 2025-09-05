"""
Phase 1 Helper Functions
Contains all business logic functions for Phase 1 work item analysis.
"""

import os
import logging
import asyncio
import json
import re
from typing import Dict, List, Any
from datetime import datetime

import httpx
from .image_analyzer import generate_image_summaries_from_incident

logger = logging.getLogger(__name__)


async def create_consolidated_analysis(work_items: List[Dict], ctx) -> Dict[str, Any]:
    """
    Create consolidated analysis of work items including image analysis.
    This replaces the old create_consolidated_summary tool.
    """
    try:
        # Structure the analysis data
        analysis = {
            "summary_metadata": {
                "total_work_items": len(work_items),
                "analysis_timestamp": datetime.now().isoformat()
            },
            "work_items_analysis": [],
            "technical_insights": {
                "common_technologies": [],
                "issue_patterns": [],
                "business_impact": "",
                "urgency_indicators": []
            },
            "image_analysis_included": False
        }
        
        pat = os.getenv("MICROSOFT_TOKEN")
        
        # Process each work item
        for idx, item in enumerate(work_items):
            await ctx.report_progress(
                (idx + 1) / len(work_items) * 0.8, 
                1.0, 
                f"Analyzing work item {idx + 1}/{len(work_items)} (including images)"
            )
            
            fields = item.get('fields', {})
            
            # Extract structured data
            work_item_analysis = {
                "id": item.get('id'),
                "title": fields.get('System.Title', ''),
                "work_item_type": fields.get('System.WorkItemType', 'Unknown'),
                "description": fields.get('System.Description', ''),
                "history": fields.get('System.History', ''),
                "state": fields.get('System.State', 'Unknown'),
                "assigned_to": fields.get('System.AssignedTo', {}).get('displayName', 'Unassigned'),
                "priority": fields.get('Microsoft.VSTS.Common.Priority', 'Not Set'),
                "area_path": fields.get('System.AreaPath', ''),
                "image_insights": ""
            }
            
            # Analyze images in description if available
            description = fields.get('System.Description', '')
            title = fields.get('System.Title', '')
            
            if description and pat and '<img' in description:
                try:
                    await ctx.info(f"Analyzing images in work item {item.get('id')}")
                    image_summary = await generate_image_summaries_from_incident(
                        description, 
                        title, 
                        pat
                    )
                    if image_summary:
                        work_item_analysis["image_insights"] = image_summary
                        analysis["image_analysis_included"] = True
                        await ctx.info(f"Image analysis completed for work item {item.get('id')}")
                except Exception as e:
                    logger.warning(f"Image analysis failed for work item {item.get('id')}: {e}")
                    work_item_analysis["image_insights"] = "Image analysis failed"
            
            analysis["work_items_analysis"].append(work_item_analysis)
            
            # Let LLM agent decide search terms based on work item context
            # Remove automatic technical term extraction
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error creating consolidated analysis: {e}", exc_info=True)
        return {
            "error": str(e),
            "summary_metadata": {"total_work_items": len(work_items)},
            "work_items_analysis": [],
            "image_analysis_included": False
        }


async def fetch_azure_work_item_details(work_item_id: int) -> Dict[str, Any]:
    """
    Fetch single work item from Azure DevOps using your existing pattern.
    Enhanced to include the important fields you specified.
    """
    try:
        pat = os.getenv("MICROSOFT_TOKEN")
        if not pat:
            raise ValueError("MICROSOFT_TOKEN not set.")
            
        # Enhanced API call to get more fields including history
        api_url = f"https://dev.azure.com/premierinc/Fireball/_apis/wit/workitems/{work_item_id}?$expand=all&api-version=7.1"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, auth=("", pat))
            response.raise_for_status()
            
            work_item_data = response.json()
            fields = work_item_data.get("fields", {})
            
            # Return structured data with all important fields
            return {
                "id": work_item_data.get("id"),
                "url": work_item_data.get("url"),
                "fields": {
                    "System.WorkItemType": fields.get("System.WorkItemType"),
                    "System.Title": fields.get("System.Title"),
                    "System.Description": fields.get("System.Description", ""),
                    "System.History": fields.get("System.History", ""),  # Discussions
                    "System.State": fields.get("System.State"),
                    "System.AssignedTo": fields.get("System.AssignedTo", {}),
                    "System.CreatedDate": fields.get("System.CreatedDate"),
                    "System.ChangedDate": fields.get("System.ChangedDate"),
                    "System.AreaPath": fields.get("System.AreaPath"),
                    "System.IterationPath": fields.get("System.IterationPath"),
                    "Microsoft.VSTS.Common.Priority": fields.get("Microsoft.VSTS.Common.Priority"),
                    "Microsoft.VSTS.Common.Severity": fields.get("Microsoft.VSTS.Common.Severity")
                }
            }
            
    except Exception as e:
        logger.error(f"Error fetching Azure work item {work_item_id}: {e}", exc_info=True)
        return {"error": f"Could not retrieve work item details: {e}"}


async def search_ado_work_items_by_text(query: str) -> List[Dict]:
    """
    Search ADO work items using text search API.
    Fixed version based on Azure DevOps REST API documentation.
    """
    try:
        pat = os.getenv("MICROSOFT_TOKEN")
        if not pat:
            raise ValueError("MICROSOFT_TOKEN not set.")
        
        # Corrected search URL - use project-specific endpoint
        search_url = "https://almsearch.dev.azure.com/premierinc/Fireball/_apis/search/workitemsearchresults?api-version=7.1"
        
        # Corrected search payload based on documentation
        search_payload = {
            "searchText": query,
            "$skip": 0,
            "$top": 10,
            "filters": {
                "System.TeamProject": ["Fireball"] 
            },
            "includeFacets": False
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                search_url, 
                json=search_payload,  # Use json= instead of data=
                auth=("", pat),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            search_results = response.json()
            work_items = []
            
            # Process search results to get full work item details
            if search_results.get("results"):
                for result in search_results["results"]:
                    # Extract work item ID from the result
                    work_item_id = result.get("fields", {}).get("system.id")
                    if work_item_id:
                        # Fetch full work item details using the existing function
                        work_item_details = await fetch_azure_work_item_details(int(work_item_id))
                        if work_item_details and "error" not in work_item_details:
                            work_items.append(work_item_details)
            
            logger.info(f"Found {len(work_items)} work items for query: {query}")
            return work_items
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error searching ADO work items: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        logger.error(f"Error searching ADO work items: {e}", exc_info=True)
        return []
