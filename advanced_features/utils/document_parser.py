# utils/document_parser.py
import fitz  # PyMuPDF
import docx
import json
from pathlib import Path
from typing import Dict, List

def parse_pdf(file_path: Path) -> str:
    """Extracts text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def parse_docx(file_path: Path) -> str:
    """Extracts text from a DOCX file."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def parse_json(file_path: Path) -> str:
    """Loads a JSON file and returns a formatted string."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Convert JSON to a string representation for embedding
    return json.dumps(data, indent=2)

# This is our dispatcher function
def parse_document(file_path: str) -> str:
    """
    Parses a document based on its file extension.
    This is our "Partition" stage.
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.pdf':
        return parse_pdf(path)
    elif extension == '.docx':
        return parse_docx(path)
    elif extension == '.json':
        return parse_json(path)
    elif extension == '.md' or extension == '.txt':
        return path.read_text()
    else:
        print(f"Warning: Unsupported file type '{extension}'. Skipping file {path.name}.")
        return None