"""
Phase 1 MCP Tool Docstrings
Contains all docstring constants for clean code organization.
"""

# Tool docstrings as constants
GET_WORK_ITEM_DETAILS_DOCSTRING = """
**START HERE**: Get work item details from Azure DevOps and create consolidated analysis.

This tool handles two types of input and automatically creates a comprehensive analysis:
1. **Single Work Item ID**: Enter a numeric ID (e.g., "1248000") to fetch one specific work item
2. **Natural Language**: Enter a description (e.g., "login issues with authentication") to search for related work items

**What it does automatically**:
- Fetches work item details from Azure DevOps
- Analyzes any image attachments using Azure Vision AI
- Creates consolidated summary with technical terms extraction
- Prepares structured data for next steps

**When to use**: Always call this tool first when starting work item analysis. It's the entry point for Phase 1.

**Returns**: Complete work item analysis with image insights and structured summary ready for documentation search.
"""

SEARCH_DOCUMENTATION_DOCSTRING = """
**STEP 2**: Search documentation using Azure AI Search vector embeddings.

**When to use**: Call this after get_work_item_details() has completed the work item analysis.

**What it does**: 
- Uses your existing HybridRetriever (hybrid_search method) to search documentation
- Searches based on your specific search query
- Returns relevant documentation with relevance scores

**Parameters**:
- session_id: The session containing work item context (default: "default")
- search_query: **REQUIRED** - Specific search query for documentation (e.g., "Azure authentication error troubleshooting", "login page issues resolution")

**Example search_queries**:
- "authentication errors Azure AD troubleshooting"
- "login page performance issues solutions" 
- "API timeout error resolution steps"

**Example**: search_documentation(session_id="default", search_query="authentication error Azure AD")
"""

GET_PHASE1_SUMMARY_DOCSTRING = """
**FINAL STEP**: Get complete Phase 1 summary and readiness assessment.

**When to use**: Call this after completing the previous steps (get_work_item_details, search_documentation) to get a comprehensive overview.

**What it provides**:
- Complete summary of all Phase 1 activities
- Work items and documentation found
- Simple, clean structure ready for LLM processing

**LLM Responsibility**: After receiving this summary, the LLM should:
1. Create a concise summary of the work items and the documentation findings relevant to the work items context
2. Identify and list the top 3 documentation sources in short for user reference
3. Present the complete findings to the user for review

**Next Phase**: User reviews the summary and provides feedback. Then GitHub MCP server can be used for intelligent repository analysis based on user input.

**Returns**: Simplified summary structure with all Phase 1 findings ready for LLM processing and user presentation.
"""
