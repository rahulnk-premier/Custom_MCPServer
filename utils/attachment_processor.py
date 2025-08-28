# attachment_processor.py
"""
Utility functions for processing Azure DevOps attachments with multimodal capabilities.
Uses Transformers library for open-source vision models.
"""

import os
import re
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
import torch
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq
)

logger = logging.getLogger(__name__)

class AttachmentProcessor:
    """Handles extraction and analysis of attachments from Azure DevOps incidents using Transformers."""
    
    def __init__(
        self, 
        certificate_path: str = None,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = None
    ):
        """
        Initialize the attachment processor with Transformers models.
        
        Args:
            certificate_path: Path to SSL certificate for corporate environments
            model_name: Name of the vision model to use from HuggingFace
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.certificate_path = certificate_path
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = None
        self.model = None
        
        # Initialize vision model
        self._initialize_vision_model()
    
    def _initialize_vision_model(self):
        """Initialize the Transformers vision model."""
        try:
            logger.info(f"Loading vision model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Try different model options based on availability and memory
            model_options = [
                self.model_name,
                "Salesforce/blip-image-captioning-base",  # Smaller fallback
                "microsoft/git-base-coco"  # Alternative model
            ]
            
            for model_name in model_options:
                try:
                    if "blip2" in model_name.lower():
                        # BLIP-2 models (more powerful but larger)
                        self.processor = Blip2Processor.from_pretrained(model_name)
                        self.model = Blip2ForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                            low_cpu_mem_usage=True
                        )
                    elif "blip" in model_name.lower():
                        # BLIP models
                        self.processor = BlipProcessor.from_pretrained(model_name)
                        self.model = BlipForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                            low_cpu_mem_usage=True
                        )
                    else:
                        # Generic vision-to-text models
                        self.processor = AutoProcessor.from_pretrained(model_name)
                        self.model = AutoModelForVision2Seq.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                            low_cpu_mem_usage=True
                        )
                    
                    self.model = self.model.to(self.device)
                    self.model.eval()  # Set to evaluation mode
                    logger.info(f"Successfully loaded model: {model_name}")
                    self.model_name = model_name
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.model is None:
                raise RuntimeError("Failed to load any vision model")
                
        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            logger.info("Please ensure transformers and torch are installed:")
            logger.info("pip install transformers torch torchvision")
            raise
    
    async def extract_attachment_info(self, html_content: str) -> List[Dict[str, str]]:
        """
        Extract attachment URLs and metadata from HTML content.
        
        Args:
            html_content: HTML string from Azure DevOps description/history
            
        Returns:
            List of attachment dictionaries with type, url, and metadata
        """
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        attachments = []
        
        # Extract images
        for img in soup.find_all('img'):
            if 'src' in img.attrs:
                attachments.append({
                    'type': 'image',
                    'url': img['src'],
                    'alt': img.get('alt', 'No description'),
                    'element': 'img'
                })
        
        # Extract attachment links
        attachment_pattern = re.compile(r'.*(attachments|attachment).*', re.IGNORECASE)
        for link in soup.find_all('a', href=attachment_pattern):
            attachments.append({
                'type': 'document',
                'url': link.get('href', ''),
                'text': link.get_text(strip=True),
                'element': 'link'
            })
        
        # Also check for embedded file references in text
        text_content = soup.get_text()
        file_patterns = [
            r'(\w+\.(png|jpg|jpeg|gif|bmp|svg))',
            r'(\w+\.(pdf|docx|xlsx|txt|log))'
        ]
        for pattern in file_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in matches:
                filename = match[0] if isinstance(match, tuple) else match
                if not any(att.get('text') == filename for att in attachments):
                    attachments.append({
                        'type': 'referenced',
                        'filename': filename,
                        'element': 'text_reference'
                    })
        
        logger.info(f"Extracted {len(attachments)} attachments from HTML content")
        return attachments
    
    async def download_attachment(self, url: str, pat: str) -> bytes:
        """
        Download attachment from Azure DevOps.
        
        Args:
            url: Attachment URL
            pat: Personal Access Token for Azure DevOps
            
        Returns:
            Binary content of the attachment
        """
        try:
            verify = self.certificate_path if self.certificate_path else True
            
            async with httpx.AsyncClient(verify=verify, timeout=30.0) as client:
                response = await client.get(
                    url,
                    auth=("", pat),
                    follow_redirects=True
                )
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.error(f"Failed to download attachment from {url}: {e}")
            raise
    
    async def analyze_image_with_context(
        self, 
        image_data: bytes, 
        incident_context: Dict[str, str]
    ) -> str:
        """
        Analyze image content with incident context using Transformers vision model.
        
        Args:
            image_data: Binary image data
            incident_context: Dictionary with title, description, etc.
            
        Returns:
            Analysis text describing the image content
        """
        try:
            # Open and prepare image
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Build context prompt
            context_prompt = self._build_context_prompt(incident_context)
            
            # Perform vision analysis based on model type
            if "blip2" in self.model_name.lower():
                analysis = await self._analyze_with_blip2(image, context_prompt)
            elif "blip" in self.model_name.lower():
                analysis = await self._analyze_with_blip(image, context_prompt)
            else:
                analysis = await self._analyze_with_generic_model(image, context_prompt)
            
            # Enhance analysis with context
            enhanced_analysis = self._enhance_analysis_with_context(
                analysis, 
                incident_context
            )
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return f"Image analysis failed: {str(e)}"
    
    def _build_context_prompt(self, incident_context: Dict[str, str]) -> str:
        """Build a context-aware prompt for image analysis."""
        title = incident_context.get('title', 'Unknown issue')
        description = incident_context.get('description', '')
        
        # Clean description from HTML
        if description:
            soup = BeautifulSoup(description, 'html.parser')
            description = soup.get_text(strip=True)[:200]
        
        # Create a detailed prompt for better analysis
        prompt = f"""Analyzing screenshot for incident: {title}.
Context: {description}
Describe the UI elements, any visible errors, misalignments, or technical issues in detail."""
        
        return prompt
    
    async def _analyze_with_blip(self, image: Image.Image, context_prompt: str) -> str:
        """Analyze image using BLIP model."""
        try:
            # Prepare inputs with context
            inputs = self.processor(
                image, 
                text=context_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate detailed caption
            with torch.no_grad():
                output = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_length=150,
                    num_beams=5,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            # Decode the output
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Try to get more details with conditional generation
            question_prompts = [
                "What UI elements are visible?",
                "What alignment issues are present?",
                "Describe any error messages or problems shown."
            ]
            
            additional_details = []
            for question in question_prompts:
                try:
                    q_inputs = self.processor(image, text=question, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        q_output = self.model.generate(**q_inputs, max_length=50)
                    detail = self.processor.decode(q_output[0], skip_special_tokens=True)
                    if detail and detail != caption:
                        additional_details.append(detail)
                except:
                    continue
            
            # Combine all analyses
            full_analysis = caption
            if additional_details:
                full_analysis += ". Additional observations: " + ". ".join(additional_details)
            
            return full_analysis
            
        except Exception as e:
            logger.error(f"BLIP analysis failed: {e}")
            return "Failed to analyze image with BLIP model"
    
    async def _analyze_with_blip2(self, image: Image.Image, context_prompt: str) -> str:
        """Analyze image using BLIP-2 model (more advanced)."""
        try:
            # BLIP-2 can handle question-answering better
            inputs = self.processor(image, text=context_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_length=200,
                    num_beams=5,
                    temperature=0.7
                )
            
            analysis = self.processor.decode(output[0], skip_special_tokens=True)
            return analysis
            
        except Exception as e:
            logger.error(f"BLIP-2 analysis failed: {e}")
            return "Failed to analyze image with BLIP-2 model"
    
    async def _analyze_with_generic_model(self, image: Image.Image, context_prompt: str) -> str:
        """Analyze image using a generic vision-to-text model."""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_length=100
                )
            
            analysis = self.processor.decode(output[0], skip_special_tokens=True)
            return analysis
            
        except Exception as e:
            logger.error(f"Generic model analysis failed: {e}")
            return "Failed to analyze image"
    
    def _enhance_analysis_with_context(
        self, 
        raw_analysis: str, 
        incident_context: Dict[str, str]
    ) -> str:
        """Enhance the raw analysis with incident context."""
        title = incident_context.get('title', '')
        
        # Add context-specific interpretations
        enhanced = f"Image Analysis for '{title}': {raw_analysis}"
        
        # Add specific observations based on keywords in title/description
        if 'alignment' in title.lower() or 'align' in title.lower():
            enhanced += " The image appears to show alignment-related UI issues."
        
        if 'sort' in title.lower():
            enhanced += " Focus on sorting controls and their positioning."
        
        if 'icon' in title.lower():
            enhanced += " Icon positioning and rendering should be examined."
        
        if 'ellipses' in title.lower() or 'ellipsis' in title.lower():
            enhanced += " Text truncation with ellipsis may be affecting layout."
        
        return enhanced
    
    async def parse_discussion_history(self, history_html: str) -> List[Dict[str, str]]:
        """
        Parse and extract meaningful information from discussion history.
        
        Args:
            history_html: HTML content from System.History field
            
        Returns:
            List of discussion entries with metadata
        """
        if not history_html:
            return []
        
        soup = BeautifulSoup(history_html, 'html.parser')
        discussions = []
        
        # Try to parse structured discussions
        entries = soup.find_all(['div', 'p', 'li'])
        
        for entry in entries:
            text = entry.get_text(strip=True)
            if text and len(text) > 10:
                # Extract potential metadata
                discussion_entry = {
                    'text': text,
                    'type': 'comment'
                }
                
                # Categorize based on content
                if re.search(r'(fix|fixed|resolve|resolved|solution)', text, re.IGNORECASE):
                    discussion_entry['type'] = 'resolution'
                elif re.search(r'(test|testing|tested|verify|verified)', text, re.IGNORECASE):
                    discussion_entry['type'] = 'testing'
                elif re.search(r'(error|bug|issue|problem|fail)', text, re.IGNORECASE):
                    discussion_entry['type'] = 'problem'
                elif re.search(r'(update|change|modify|adjust)', text, re.IGNORECASE):
                    discussion_entry['type'] = 'update'
                
                discussions.append(discussion_entry)
        
        # If no structured content, fall back to text splitting
        if not discussions:
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 10]
            discussions = [{'text': line, 'type': 'comment'} for line in lines[:10]]
        
        logger.info(f"Extracted {len(discussions)} discussion entries")
        return discussions
    
    async def create_consolidated_summary(
        self,
        incident_details: Dict[str, Any],
        image_analyses: List[str],
        discussions: List[Dict[str, str]]
    ) -> str:
        """
        Create a comprehensive summary combining all information.
        
        Args:
            incident_details: Basic incident information
            image_analyses: Results from image analysis
            discussions: Parsed discussion history
            
        Returns:
            Consolidated summary text
        """
        summary_parts = []
        
        # Header with key information
        summary_parts.append(f"## Incident: {incident_details.get('title', 'Unknown')}")
        summary_parts.append(f"**State**: {incident_details.get('state', 'Unknown')}")
        summary_parts.append(f"**Type**: {incident_details.get('work_item_type', 'Unknown')}")
        summary_parts.append(f"**Assigned To**: {incident_details.get('assigned_to', 'Unassigned')}")
        
        # Clean and add description
        description = incident_details.get('description', '')
        if description:
            soup = BeautifulSoup(description, 'html.parser')
            clean_desc = soup.get_text(strip=True)
            if clean_desc:
                # Limit description length and clean it up
                clean_desc = ' '.join(clean_desc.split())[:500]
                summary_parts.append(f"\n**Description**: {clean_desc}")
        
        # Add visual evidence analysis
        if image_analyses:
            summary_parts.append("\n**Visual Evidence Analysis**:")
            for i, analysis in enumerate(image_analyses, 1):
                # Clean and format each analysis
                clean_analysis = ' '.join(analysis.split())[:300]
                summary_parts.append(f"{i}. {clean_analysis}")
        
        # Add relevant discussions
        if discussions:
            summary_parts.append("\n**Key Discussion Points**:")
            # Prioritize different types of discussions
            priority_types = ['resolution', 'problem', 'testing', 'update', 'comment']
            added_discussions = 0
            
            for disc_type in priority_types:
                for disc in discussions:
                    if disc['type'] == disc_type and added_discussions < 5:
                        text = ' '.join(disc['text'].split())[:200]
                        summary_parts.append(f"- [{disc['type'].upper()}] {text}")
                        added_discussions += 1
        
        # Add resolution information if available
        resolution = incident_details.get('resolution', '')
        if resolution and resolution != 'No resolution provided.':
            soup = BeautifulSoup(resolution, 'html.parser')
            clean_resolution = soup.get_text(strip=True)
            if clean_resolution:
                clean_resolution = ' '.join(clean_resolution.split())[:300]
                summary_parts.append(f"\n**Previous Resolution Attempts**: {clean_resolution}")
        
        # Combine all parts
        consolidated_summary = "\n".join(summary_parts)
        
        # Add technical analysis based on all gathered information
        tech_analysis = self._generate_technical_analysis(
            incident_details, 
            image_analyses, 
            discussions
        )
        consolidated_summary += f"\n\n**Technical Analysis**: {tech_analysis}"
        
        return consolidated_summary
    
    def _generate_technical_analysis(
        self,
        incident_details: Dict[str, Any],
        image_analyses: List[str],
        discussions: List[Dict[str, str]]
    ) -> str:
        """Generate technical analysis based on all available information."""
        title = incident_details.get('title', '').lower()
        all_text = ' '.join(image_analyses).lower() + ' ' + ' '.join([d['text'] for d in discussions]).lower()
        
        analysis_points = []
        
        # Identify technical areas based on keywords
        if 'ui' in all_text or 'alignment' in title or 'display' in all_text:
            analysis_points.append("UI/UX rendering issue detected")
        
        if 'css' in all_text or 'style' in all_text or 'position' in all_text:
            analysis_points.append("CSS/styling problem likely")
        
        if 'grid' in title or 'table' in title or 'column' in title:
            analysis_points.append("Data table/grid component affected")
        
        if 'responsive' in all_text or 'screen' in all_text or 'width' in all_text:
            analysis_points.append("Responsive design issue possible")
        
        if 'sort' in title or 'filter' in all_text:
            analysis_points.append("Data manipulation controls involved")
        
        if not analysis_points:
            analysis_points.append("General application issue requiring investigation")
        
        # Add recommended investigation areas
        recommendations = "Recommended investigation areas: "
        if 'css' in ' '.join(analysis_points).lower():
            recommendations += "Check CSS positioning properties, z-index, overflow handling. "
        if 'responsive' in ' '.join(analysis_points).lower():
            recommendations += "Test at different screen widths, check media queries. "
        if 'grid' in ' '.join(analysis_points).lower():
            recommendations += "Review table component configuration, column width calculations. "
        
        return f"{'. '.join(analysis_points)}. {recommendations}"


# Helper function for testing
async def test_processor():
    """Test function to verify the processor is working."""
    try:
        processor = AttachmentProcessor()
        print(f"✓ Processor initialized with model: {processor.model_name}")
        print(f"✓ Using device: {processor.device}")
        
        # Test with a sample HTML
        sample_html = '<img src="test.png" alt="Test Image">'
        attachments = await processor.extract_attachment_info(sample_html)
        print(f"✓ HTML parsing works: {len(attachments)} attachments found")
        
        return True
    except Exception as e:
        print(f"✗ Processor test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test if executed directly
    import asyncio
    asyncio.run(test_processor())