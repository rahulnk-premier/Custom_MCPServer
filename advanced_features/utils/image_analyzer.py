# utils/image_analyzer.py

import os
import logging
import asyncio
from typing import List
import io
import ssl
import certifi

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# NEW: Import Azure AI Vision SDK components
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv() # Load environment variables from .env file

# NEW: Azure Computer Vision Configuration
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")



# if os.path.exists(CERTIFICATE_PATH):
#     os.environ['REQUESTS_CA_BUNDLE'] = CERTIFICATE_PATH



# --- Helper Functions (Unchanged) ---
def extract_image_urls(html_content: str) -> List[str]:
    """Parses an HTML string to find and extract all image URLs."""
    if not html_content:
        return []
    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.find_all('img')
    urls = [img['src'] for img in img_tags if 'src' in img.attrs]
    logger.info(f"Found {len(urls)} image URL(s) in HTML content.")
    return urls

# --- REWRITTEN: Image Analysis function now uses Azure ---
async def _analyze_single_image_with_azure(
    vision_client: ImageAnalysisClient, 
    image_bytes: bytes, 
    context_text: str
) -> str:
    """
    Analyzes a single image using the Azure AI Vision service and frames it with context.
    """
    try:
        # The Azure Vision SDK's 'analyze' method is synchronous.
        # We run it in a separate thread to avoid blocking the asyncio event loop.
        def perform_azure_analysis():
            result = vision_client.analyze(
                image_data=image_bytes,
                visual_features=[VisualFeatures.CAPTION]
            )
            
            # If a caption is found, prepend our context to it.
            if result.caption and result.caption.text:
                # This is how we re-introduce the context to the final summary
                return f"{context_text}: '{result.caption.text}'"
            return "No caption could be generated for this image."

        # Execute the synchronous SDK call in a non-blocking way
        summary = await asyncio.to_thread(perform_azure_analysis)
        logger.info(f"Successfully generated summary from Azure Vision.")
        return summary

    except Exception as e:
        logger.error(f"Azure Vision API call failed: {e}", exc_info=True)
        return "Failed to analyze image with Azure service."

# --- Main Public Function (MODIFIED to use Azure) ---
async def generate_image_summaries_from_incident(
    html_description: str,
    incident_title: str,
    pat: str
) -> str:
    """
    Orchestrates extracting, downloading, and analyzing images using Azure AI Vision.
    """
    if not all([AZURE_VISION_ENDPOINT, AZURE_VISION_KEY]):
        logger.error("Azure Vision endpoint or key is not configured. Skipping image analysis.")
        return ""

    image_urls = extract_image_urls(html_description)
    if not image_urls:
        return ""

    # NEW: Initialize the Azure Vision client once
    vision_client = ImageAnalysisClient(
        endpoint=AZURE_VISION_ENDPOINT,
        credential=AzureKeyCredential(AZURE_VISION_KEY)
    )

    summaries = []
        
    async with httpx.AsyncClient(auth=("", pat)) as client:
        for i, url in enumerate(image_urls, 1):
            try:
                logger.info(f"Downloading image from: {url}")
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                image_bytes = response.content
                context = f"A screenshot of a software error related to '{incident_title}'"
                
                # CHANGED: Call the new Azure-based analysis function
                summary = await _analyze_single_image_with_azure(vision_client, image_bytes, context)
                
                summaries.append(f"Image {i} Analysis: {summary}")

            except httpx.RequestError as e:
                logger.warning(f"Failed to download image from {url}. Error: {e}")
                summaries.append(f"Image {i} Analysis: Could not download image.")
            except Exception as e:
                logger.error(f"An unexpected error occurred processing image {url}. Error: {e}")
                summaries.append(f"Image {i} Analysis: An unexpected error occurred.")
                
    return " ".join(summaries)