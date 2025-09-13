import fitz  # PyMuPDF
from PIL import Image
import io
import os
from typing import List, Dict, Tuple, Optional
import base64
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles extraction of text and images from PDF documents."""
    
    def __init__(self, output_dir: str = "./data/extracted_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_and_images(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract both text and images from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text chunks and image information
        """
        doc = fitz.open(pdf_path)
        
        extracted_data = {
            "text_chunks": [],
            "images": [],
            "metadata": {
                "total_pages": len(doc),
                "document_name": Path(pdf_path).stem
            }
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text
            text = page.get_text()
            if text.strip():
                extracted_data["text_chunks"].append({
                    "content": text,
                    "page_number": page_num + 1,
                    "chunk_type": "text",
                    "metadata": {"source": pdf_path}
                })
            
        # Extract images (temporarily disabled to avoid issues)
        logger.info("Image extraction temporarily disabled for testing")
        # image_list = page.get_images(full=True)
        # for img_index, img_ref in enumerate(image_list):
        #     try:
        #         image_data = self._extract_image(doc, img_ref, page_num, img_index)
        #         if image_data:
        #             extracted_data.append(image_data)
        #     except Exception as e:
        #         logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")        doc.close()
        return extracted_data
    
    def _extract_image(self, doc: fitz.Document, img_ref: Tuple, page_num: int, img_index: int) -> Optional[Dict]:
        """Extract and save an individual image from the PDF."""
        try:
            xref = img_ref[0]
            pix = fitz.Pixmap(doc, xref)
            
            # Skip images that cause colorspace issues for now
            if pix.n >= 5:  # CMYK or other complex colorspaces
                logger.warning(f"Skipping image {img_index} on page {page_num}: Complex colorspace not supported")
                pix = None
                return None
            
            # Use get_png_data() method which is more reliable
            try:
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
            except Exception as e:
                # If PNG fails, try get_png_data method
                try:
                    png_data = pix.get_png_data()
                    img = Image.open(io.BytesIO(png_data))
                except Exception as e2:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e2}")
                    pix = None
                    return None
            
            # Save image
            filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
            filepath = self.output_dir / filename
            img.save(filepath, "PNG")
            
            # Convert to base64 for embedding
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            pix = None
            
            return {
                "filename": filename,
                "filepath": str(filepath),
                "page_number": page_num + 1,
                "image_index": img_index,
                "base64": img_base64,
                "chunk_type": "image",
                "size": img.size,
                "metadata": {"source": "extracted_from_pdf"}
            }
            
        except Exception as e:
            logger.error(f"Error extracting image: {e}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            if start < 0:
                start = 0
                
        return chunks
    
    def process_document_with_chunking(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, any]:
        """
        Process document and create text chunks suitable for embedding.
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Processed document data with chunked text
        """
        extracted_data = self.extract_text_and_images(pdf_path)
        
        # Process text chunks
        all_text_chunks = []
        for text_chunk in extracted_data["text_chunks"]:
            chunked_texts = self.chunk_text(text_chunk["content"], chunk_size, chunk_overlap)
            
            for i, chunk_text in enumerate(chunked_texts):
                all_text_chunks.append({
                    "content": chunk_text,
                    "page_number": text_chunk["page_number"],
                    "chunk_id": f"page_{text_chunk['page_number']}_chunk_{i}",
                    "chunk_type": "text",
                    "metadata": text_chunk["metadata"]
                })
        
        # Add image descriptions as searchable content
        for image in extracted_data["images"]:
            image["chunk_id"] = f"page_{image['page_number']}_img_{image['image_index']}"
        
        return {
            "text_chunks": all_text_chunks,
            "images": extracted_data["images"],
            "metadata": extracted_data["metadata"]
        }
