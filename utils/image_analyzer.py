# utils/image_analyzer.py

import os
import logging
import asyncio
from typing import List, Optional
import io
import ssl
import certifi

import httpx
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from bs4 import BeautifulSoup

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SSL Configuration ---
# Use the same certificate path you defined in mcp_app.py
CERTIFICATE_PATH = r"E:\rnakka\Downloads\trusted_certs.crt"

def create_ssl_context():
    """Create SSL context with corporate certificates"""
    # Start with the default certs from certifi
    context = ssl.create_default_context(cafile=certifi.where())
    
    # Add your corporate certificate if it exists
    if os.path.exists(CERTIFICATE_PATH):
        logger.info(f"Loading custom corporate certificate from: {CERTIFICATE_PATH}")
        context.load_verify_locations(CERTIFICATE_PATH)
    else:
        logger.warning(f"Custom certificate file not found at: {CERTIFICATE_PATH}. Using default certs.")
        
    return context


if os.path.exists(CERTIFICATE_PATH):
    os.environ['REQUESTS_CA_BUNDLE'] = CERTIFICATE_PATH

# --- Model Configuration ---
MODEL_ID = "Salesforce/blip-image-captioning-base"

# --- Global Model Cache ---
PROCESSOR: Optional[BlipProcessor] = None
MODEL: Optional[BlipForConditionalGeneration] = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    logger.info(f"Loading vision model '{MODEL_ID}' onto device '{DEVICE}'...")
    # Transformers/requests will now use the REQUESTS_CA_BUNDLE env var for this download
    PROCESSOR = BlipProcessor.from_pretrained(MODEL_ID)
    MODEL = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
    logger.info("Vision model loaded successfully.")
except Exception as e:
    logger.error(f"FATAL: Failed to load the vision model '{MODEL_ID}'. Image analysis will be disabled. Error: {e}", exc_info=True)
    


def extract_image_urls(html_content: str) -> List[str]:
    """
    Parses an HTML string to find and extract all image URLs.
    (This function does not need changes)
    """
    if not html_content:
        return []
    
    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.find_all('img')
    urls = [img['src'] for img in img_tags if 'src' in img.attrs]
    logger.info(f"Found {len(urls)} image URL(s) in HTML content.")
    return urls

async def _analyze_single_image(image_bytes: bytes, context_text: str) -> str:
    """
    Analyzes a single image using the pre-loaded BLIP model, guided by context.
    (This function does not need changes)
    """
    if not MODEL or not PROCESSOR:
        logger.warning("Vision model is not available. Skipping image analysis.")
        return "Vision model is not loaded, analysis skipped."

    try:
        raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        def perform_inference():
            inputs = PROCESSOR(raw_image, context_text, return_tensors="pt").to(DEVICE)
            output = MODEL.generate(**inputs)
            caption = PROCESSOR.decode(output[0], skip_special_tokens=True)
            return caption

        caption = await asyncio.to_thread(perform_inference)
        logger.info(f"Generated image caption: '{caption}'")
        return caption.strip()

    except Exception as e:
        logger.error(f"Error during image analysis: {e}", exc_info=True)
        return "Failed to analyze image."

# --- Main Public Function (MODIFIED) ---

async def generate_image_summaries_from_incident(
    html_description: str,
    incident_title: str,
    pat: str
) -> str:
    """
    Orchestrates the process of extracting, downloading, and analyzing images
    from an Azure DevOps incident description.
    """
    image_urls = extract_image_urls(html_description)
    if not image_urls:
        return ""

    summaries = []
    
    # *** MODIFICATION HERE ***
    # Create the SSL context and pass it to the httpx client.
    ssl_context = create_ssl_context()
    
    async with httpx.AsyncClient(auth=("", pat), verify=ssl_context) as client:
        for i, url in enumerate(image_urls, 1):
            try:
                logger.info(f"Downloading image from: {url}")
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                image_bytes = response.content
                context = f"a screenshot of a software error related to: {incident_title}"
                summary = await _analyze_single_image(image_bytes, context)
                
                summaries.append(f"Image {i} Analysis: {summary}")

            except httpx.RequestError as e:
                logger.warning(f"Failed to download image from {url}. Error: {e}")
                summaries.append(f"Image {i} Analysis: Could not download image.")
            except Exception as e:
                logger.error(f"An unexpected error occurred processing image {url}. Error: {e}")
                summaries.append(f"Image {i} Analysis: An unexpected error occurred.")
                
    return " ".join(summaries)