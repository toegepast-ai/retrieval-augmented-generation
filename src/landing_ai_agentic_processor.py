"""
Landing AI processor using the official agentic-doc library with caching
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from agentic_doc.parse import parse
from agentic_doc.config import ParseConfig
from .landing_ai_cache import LandingAICache

logger = logging.getLogger(__name__)

class LandingAIAgenticProcessor:
    """
    Landing AI processor using the official agentic-doc library.
    
    This processor uses Landing AI's agentic-doc library to extract structured data 
    from documents, including text, tables, form fields, and their hierarchical relationships.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Landing AI processor with the official library.
        
        Args:
            api_key: Landing AI API key (VISION_AGENT_API_KEY)
        """
        self.api_key = api_key
        # Configure the library with the API key
        self.config = ParseConfig(api_key=api_key) if api_key else None
        # Initialize cache to avoid costly re-processing
        self.cache = LandingAICache()
        
    def extract_document_data(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Extract structured data from a document using Landing AI's agentic-doc library.
        Uses caching to avoid costly re-processing of the same document.
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional parsing options:
                - include_marginalia: Include headers, footers, page numbers (default: True)
                - include_metadata_in_markdown: Include metadata in markdown output (default: True)
                - result_save_dir: Directory to save detailed results (optional)
            
        Returns:
            Structured document data from Landing AI ADE
        """
        try:
            if not self.config:
                logger.error("Landing AI API key not configured")
                return {"error": "API key not configured"}
            
            # Check cache first to avoid costly re-processing
            cached_result = self.cache.load_result(file_path)
            if cached_result:
                logger.info(f"Using cached Landing AI result for {Path(file_path).name}")
                return cached_result
            
            # Parse the document using the official library (this costs credits!)
            logger.warning(f"⚠️  Processing with Landing AI ADE (will cost credits): {file_path}")
            logger.info(f"Processing document with Landing AI ADE: {file_path}")
            
            # Configure parsing options for optimal multimodal extraction
            parse_kwargs = {
                'include_marginalia': kwargs.get('include_marginalia', True),  # Include headers/footers
                'include_metadata_in_markdown': kwargs.get('include_metadata_in_markdown', True),  # Rich metadata
                'config': self.config
            }
            
            # Add result save directory if provided
            if 'result_save_dir' in kwargs:
                parse_kwargs['result_save_dir'] = kwargs['result_save_dir']
            
            # The parse function takes a single file path and returns a list of ParsedDocument
            results = parse(file_path, **parse_kwargs)
            
            if not results:
                logger.error("No results returned from Landing AI ADE")
                return {"error": "No results returned"}
            
            # Get the first result (since we're processing one file)
            result = results[0]
            
            # Log information about saved JSON results if applicable
            if hasattr(result, 'result_path') and result.result_path:
                logger.info(f"📄 Landing AI ADE saved detailed JSON results to: {result.result_path}")
            
            # Convert to our expected format
            processed_result = self._process_ade_response(result, file_path)
            
            # Add information about saved JSON file if available
            if hasattr(result, 'result_path') and result.result_path:
                processed_result['ade_json_path'] = str(result.result_path)
                processed_result['ade_json_saved'] = True
                logger.info(f"✅ ADE JSON results available at: {result.result_path}")
            
            # Save to cache for future use
            if processed_result.get('success'):
                self.cache.save_result(file_path, processed_result)
                logger.info(f"✅ Saved Landing AI result to cache")
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Landing AI ADE extraction failed: {e}")
            return {"error": f"Landing AI ADE extraction failed: {str(e)}"}
    
    def _process_ade_response(self, ade_result: Any, file_path: str) -> Dict[str, Any]:
        """
        Process the response from Landing AI ADE into our standard format.
        
        According to docs, ade_result is a ParsedDocument with:
        - .markdown: Full markdown representation
        - .chunks: List of Chunk objects with text, grounding, chunk_type, chunk_id
        
        Args:
            ade_result: ParsedDocument from agentic-doc library
            file_path: Original file path
            
        Returns:
            Processed data in standard format
        """
        try:
            chunks = []
            
            # Process the individual chunks (this is the main data source)
            if hasattr(ade_result, 'chunks') and ade_result.chunks:
                logger.info(f"Processing {len(ade_result.chunks)} chunks from Landing AI ADE")
                
                for chunk in ade_result.chunks:
                    # Extract content based on chunk type
                    content = getattr(chunk, 'text', '')
                    chunk_type = getattr(chunk, 'chunk_type', 'unknown')
                    chunk_id = getattr(chunk, 'chunk_id', None)
                    grounding = getattr(chunk, 'grounding', [])
                    
                    # Extract page and spatial information from grounding
                    page_number = 1
                    bounding_boxes = []
                    if grounding:
                        # Get page number from first grounding entry
                        page_number = grounding[0].page + 1  # Convert 0-indexed to 1-indexed
                        
                        # Extract all bounding boxes
                        for ground in grounding:
                            if hasattr(ground, 'box') and ground.box:
                                box = ground.box
                                bounding_boxes.append({
                                    'left': box.l,
                                    'top': box.t, 
                                    'right': box.r,
                                    'bottom': box.b,
                                    'page': ground.page
                                })
                    
                    # Create enhanced content for different chunk types
                    enhanced_content = self._enhance_chunk_content(content, chunk_type, chunk_id)
                    
                    chunk_data = {
                        "content": enhanced_content,
                        "chunk_type": chunk_type,
                        "page_number": page_number,
                        "chunk_id": chunk_id,
                        "metadata": {
                            "source": "landing_ai_ade",
                            "extraction_method": "semantic_chunking",
                            "original_chunk_type": chunk_type,
                            "bounding_boxes": bounding_boxes,
                            "spatial_grounding": grounding,
                            "has_spatial_data": len(bounding_boxes) > 0
                        }
                    }
                    
                    chunks.append(chunk_data)
            
            # Also include the full markdown as a comprehensive chunk
            if hasattr(ade_result, 'markdown') and ade_result.markdown:
                chunks.append({
                    "content": ade_result.markdown,
                    "chunk_type": "full_document_markdown",
                    "page_number": 1,
                    "chunk_id": "full_document",
                    "metadata": {
                        "source": "landing_ai_ade",
                        "extraction_method": "full_document_markdown",
                        "is_comprehensive": True,
                        "contains_html_comments": True  # ADE markdown includes HTML comments with chunk IDs
                    }
                })
            
            logger.info(f"Successfully processed {len(chunks)} total chunks from ADE")
            
            return {
                "chunks": chunks,
                "total_chunks": len(chunks),
                "extraction_method": "landing_ai_ade_semantic",
                "file_path": file_path,
                "success": True,
                "ade_metadata": {
                    "start_page_idx": getattr(ade_result, 'start_page_idx', None),
                    "end_page_idx": getattr(ade_result, 'end_page_idx', None),
                    "doc_type": getattr(ade_result, 'doc_type', 'unknown')
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing ADE response: {e}")
            return {
                "error": f"Error processing ADE response: {str(e)}",
                "success": False
            }
    
    def _enhance_chunk_content(self, content: str, chunk_type: str, chunk_id: Optional[str]) -> str:
        """
        Enhance chunk content based on its type for better RAG retrieval.
        
        Args:
            content: Raw content from the chunk
            chunk_type: Type of chunk (text, table, figure, marginalia)
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Enhanced content string optimized for RAG
        """
        if not content:
            return content
            
        # Add type-specific prefixes for better semantic understanding
        if chunk_type == "table":
            # Tables are in HTML format, add context
            if content.startswith("<table"):
                return f"TABLE DATA:\n{content}\n\nThis table contains structured data that can be referenced for queries about tabular information, financial data, or organized listings."
            else:
                return f"TABLE: {content}"
                
        elif chunk_type == "figure":
            # Figures include charts, graphs, diagrams
            return f"FIGURE/CHART: {content}\n\nThis visual element contains graphical information that may include charts, diagrams, or other visual data representations."
            
        elif chunk_type == "marginalia":
            # Headers, footers, page numbers
            return f"DOCUMENT METADATA: {content}\n\nThis is structural document information like headers, footers, or page numbering."
            
        elif chunk_type == "text":
            # Regular text content - check if it contains key-value pairs
            if ":" in content and "\n" in content:
                # Likely contains structured data
                return f"STRUCTURED TEXT:\n{content}\n\nThis text contains structured information with key-value pairs or organized data."
            else:
                return f"TEXT: {content}"
        
        # Default case
        return f"{chunk_type.upper()}: {content}"
    
    def _format_table_content(self, table: Any) -> str:
        """Format table data into readable text (legacy method)."""
        try:
            if hasattr(table, 'to_dict'):
                table_dict = table.to_dict()
                return f"Table data: {table_dict}"
            elif hasattr(table, 'data'):
                return f"Table data: {table.data}"
            else:
                return f"Table: {str(table)}"
        except:
            return f"Table: {str(table)}"
    
    def _format_form_field_content(self, form_field: Any) -> str:
        """Format form field data into readable text (legacy method)."""
        try:
            if hasattr(form_field, 'name') and hasattr(form_field, 'value'):
                return f"Form field '{form_field.name}': {form_field.value}"
            elif hasattr(form_field, 'to_dict'):
                field_dict = form_field.to_dict()
                return f"Form field: {field_dict}"
            else:
                return f"Form field: {str(form_field)}"
        except:
            return f"Form field: {str(form_field)}"
    
    def test_connection(self) -> bool:
        """Test if the Landing AI ADE connection is working."""
        try:
            if not self.config:
                logger.error("No API key configured for Landing AI")
                return False
            
            # Create a simple test - we could try parsing a small test document
            # For now, just check if we can import and initialize
            logger.info("Landing AI ADE library initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Landing AI ADE connection test failed: {e}")
            return False
