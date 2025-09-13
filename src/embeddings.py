import openai
from typing import List, Dict, Any, Union
import base64
import io
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class MultimodalEmbedder:
    """Handles both text and image embeddings for multimodal RAG."""
    
    def __init__(self, openai_api_key: str, text_model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.text_model = text_model
        
        # Initialize sentence transformer as backup for text embeddings
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_transformer = None
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.text_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI text embedding failed: {e}")
            # Fallback to sentence transformer
            if self.sentence_transformer:
                return self.sentence_transformer.encode(text).tolist()
            else:
                raise e
    
    def get_image_description(self, image_base64: str, prompt: str = None) -> str:
        """Get description of image using GPT-4 Vision."""
        if prompt is None:
            prompt = """Analyze this image and provide a detailed description. 
            Focus on:
            1. What type of chart, graph, or diagram this is
            2. Key data points, trends, or patterns visible
            3. Any text, labels, or numbers you can read
            4. The overall context and what insights this image conveys
            
            Be specific and detailed as this description will be used for searching and retrieval."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to get image description: {e}")
            return "Image description unavailable"
    
    def get_image_embedding(self, image_base64: str) -> List[float]:
        """
        Get embedding for image by first describing it and then embedding the description.
        For a more sophisticated approach, you could use CLIP embeddings.
        """
        description = self.get_image_description(image_base64)
        return self.get_text_embedding(description)
    
    def embed_document_chunks(self, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create embeddings for all chunks (text and images) in a document.
        
        Args:
            document_data: Output from DocumentProcessor.process_document_with_chunking()
            
        Returns:
            List of chunks with embeddings
        """
        embedded_chunks = []
        
        # Process text chunks
        for chunk in document_data["text_chunks"]:
            try:
                embedding = self.get_text_embedding(chunk["content"])
                embedded_chunk = {
                    **chunk,
                    "embedding": embedding,
                    "description": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
                }
                embedded_chunks.append(embedded_chunk)
            except Exception as e:
                logger.error(f"Failed to embed text chunk {chunk.get('chunk_id', 'unknown')}: {e}")
        
        # Process image chunks
        for image in document_data["images"]:
            try:
                description = self.get_image_description(image["base64"])
                embedding = self.get_text_embedding(description)
                
                embedded_chunk = {
                    **image,
                    "embedding": embedding,
                    "description": description,
                    "content": description  # Use description as searchable content
                }
                embedded_chunks.append(embedded_chunk)
            except Exception as e:
                logger.error(f"Failed to embed image {image.get('chunk_id', 'unknown')}: {e}")
        
        return embedded_chunks
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a search query."""
        return self.get_text_embedding(query)

class CLIPEmbedder:
    """Alternative embedder using CLIP for better image-text alignment."""
    
    def __init__(self):
        logger.warning("CLIP embedder disabled due to dependency conflicts. Using OpenAI Vision API instead.")
        self.available = False
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get CLIP text embedding."""
        raise RuntimeError("CLIP not available due to dependency conflicts. Use MultimodalEmbedder instead.")
    
    def get_image_embedding(self, image_path: str) -> List[float]:
        """Get CLIP image embedding."""
        raise RuntimeError("CLIP not available due to dependency conflicts. Use MultimodalEmbedder instead.")
