

# AI Incident Resolution Agent 

This project implements an AI-powered agent designed to assist developers in resolving incidents logged in Azure DevOps. It leverages a Multi-Capability Protocol (MCP) server built with `FastMCP` to provide tools that analyze incident details, suggest relevant GitHub repositories, and prepare for code analysis.

A key feature of this agent is its **vision capability**: it can parse images (like screenshots of UI bugs) embedded in Azure DevOps work items, generate a textual description of the image content, and use this enriched context to improve the accuracy of its repository suggestions.

## Features

-   **Azure DevOps Integration**: Fetches incident details (title, description, state) directly from Azure DevOps work items.
-   **Vision-Powered Analysis**: Extracts and analyzes images from incident descriptions using the Salesforce BLIP model to understand visual context.
-   **Vector-Based Repository Search**: Uses a consolidated summary (text + image analysis) to perform a semantic search against a ChromaDB vector store of GitHub repository contents.
-   **Guided Workflow**: Follows a structured workflow from incident analysis to repository confirmation and finally to code analysis.
-   **Extensible Toolset**: Built on `FastMCP` for easy addition of new tools (e.g., GitHub file searching, code generation).

## Project Structure

```
.
├── utils/
│   ├── __init__.py
│   └── image_analyzer.py   # Module for vision analysis
├── test.ipynb                # Jupyter notebook to create embeddings
├── mcp_app.py                # The main FastMCP server application
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment file
└── README.md                 # This file
```

---

## 🚀 Getting Started: Setup and Usage

Follow these steps to set up and run the AI Incident Resolution Agent.

### Step 1: Environment Setup

**1. Create a `.env` file**

Create a file named `.env` in the root of the project directory. This file will store your secret tokens. The `GITHUB_TOKEN` requires **read access to the target repositories**, and the `MICROSOFT_TOKEN` requires read access to Azure DevOps work items.

```ini
# .env file

# Personal Access Token for Azure DevOps
# Permissions needed: "Work Items - Read"
MICROSOFT_TOKEN="your_azure_devops_pat_here"

# Personal Access Token for GitHub
# Permissions needed: "repo" (with read access to PremierInc/Fireball repositories)
GITHUB_TOKEN="your_github_pat_here"
```

**2. Install Dependencies**

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv mcp

# Activate it
# On Windows:
mcp\Scripts\activate
# On macOS/Linux:
source mcp/bin/activate

# Install the required libraries from requirements.txt
pip install -r requirements.txt
```

### Step 2: Create Repository Embeddings

Before you can search for relevant repositories, you need to create a vector database of their contents.

**1. Run `test.ipynb`**

Open and execute all cells in the `test.ipynb` Jupyter notebook. This script will:
-   Clone the specified GitHub repositories.
-   Chunk the code files.
-   Generate embeddings using a sentence-transformer model.
-   Store the embeddings and metadata in a local ChromaDB instance (a folder named `repo_db` will be created).

This step only needs to be performed once or whenever you want to update the repository data.

### Step 3: Launch the MCP Server

With the environment set up and the database created, you can now run the agent server.

```bash
# In your terminal, from the project root
python mcp_app.py
```

If successful, you will see output indicating that the server is running and listening on `http://127.0.0.1:9002`.

```
INFO:     Uvicorn running on http://1.2.3.4:9002 (Press CTRL+C to quit)
...
INFO:     Application startup complete.
```

### Step 4: Connect VS Code to the MCP Server

To interact with your custom agent, you need to connect your VS Code AI client (e.g., GitHub Copilot) to it.

1.  Open the Command Palette in VS Code (`Ctrl+Shift+P` or `Cmd+Shift+P`).
2.  Type and select **"MCP: Add Server"**.
3.  An input box will appear. Paste the local server URL into it:
    ```
    http://127.0.0.1:9002/mcp
    ```
4.  Press Enter. A notification will confirm that the server has been added. Your VS Code is now configured to use your local agent for AI-related tasks.

### Step 5: Interact with the Agent

You can now start a chat session and interact with the agent. To begin the incident resolution workflow, use a prompt like:

> `/incident_analysis 45678`

Replace `45678` with a valid Azure DevOps work item ID. The agent will then use its tools to fetch the details, analyze any images, and suggest relevant repositories.
