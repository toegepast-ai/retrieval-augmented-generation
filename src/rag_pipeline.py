import openai
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json

from .document_processor import DocumentProcessor
from .embeddings import MultimodalEmbedder
from .vector_database import VectorDatabase
from .landing_ai_agentic_processor import LandingAIAgenticProcessor

logger = logging.getLogger(__name__)

class MultimodalRAG:
    """Main RAG pipeline that handles document processing, embedding, and question answering."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            output_dir=config.get('image_output_dir', './data/extracted_images')
        )
        
        # Initialize Landing AI processor if enabled
        self.landing_ai_processor = None
        if config.get('use_landing_ai') and config.get('vision_agent_api_key'):
            try:
                self.landing_ai_processor = LandingAIAgenticProcessor(
                    api_key=config['vision_agent_api_key']
                )
                logger.info("Landing AI ADE processor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Landing AI processor: {e}")
        
        self.embedder = MultimodalEmbedder(
            openai_api_key=config['openai_api_key'],
            text_model=config.get('text_embedding_model', 'text-embedding-3-small')
        )
        
        self.vector_db = VectorDatabase(
            persist_directory=config.get('chroma_persist_directory', './data/vector_db'),
            collection_name=config.get('collection_name', 'rag_collection')
        )
        
        self.client = openai.OpenAI(api_key=config['openai_api_key'])
        self.chat_model = config.get('chat_model', 'gpt-4-turbo-preview')
        
        # Store processed documents metadata
        self.documents_metadata = {}
    
    def process_document(self, pdf_path: str, chunk_size: int = None, chunk_overlap: int = None, 
                        save_ade_json: bool = False, ade_json_dir: str = "./ade_results") -> bool:
        """
        Process a PDF document and add it to the vector database.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks (uses config default if None)
            chunk_overlap: Overlap between chunks (uses config default if None)
            save_ade_json: Whether to save detailed Landing AI ADE JSON results
            ade_json_dir: Directory to save ADE JSON results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing document: {pdf_path}")
            
            # Use config defaults if not specified
            chunk_size = chunk_size or self.config.get('chunk_size', 1000)
            chunk_overlap = chunk_overlap or self.config.get('chunk_overlap', 200)
            
            # Choose processing method
            if self.landing_ai_processor:
                logger.info("Using Landing AI ADE for document processing")
                
                # Prepare ADE extraction parameters
                ade_kwargs = {}
                if save_ade_json:
                    # Create JSON save directory
                    json_dir = Path(ade_json_dir)
                    json_dir.mkdir(parents=True, exist_ok=True)
                    ade_kwargs['result_save_dir'] = str(json_dir)
                    logger.info(f"📄 ADE JSON results will be saved to: {json_dir.absolute()}")
                
                # Extract using Landing AI ADE
                ade_result = self.landing_ai_processor.extract_document_data(pdf_path, **ade_kwargs)
                
                if ade_result.get('success'):
                    ade_chunks = ade_result.get('chunks', [])
                    logger.info(f"Landing AI ADE extracted {len(ade_chunks)} elements")
                    
                    # Log JSON save information
                    if ade_result.get('ade_json_saved'):
                        logger.info(f"📄 Detailed ADE results saved to: {ade_result.get('ade_json_path')}")
                    
                    # Convert to embeddings format
                    document_data = {
                        "text_chunks": [chunk for chunk in ade_chunks if chunk['chunk_type'] in ['text', 'table', 'form_field', 'markdown']],
                        "images": [chunk for chunk in ade_chunks if chunk['chunk_type'] == 'image'],
                        "metadata": {"source": pdf_path, "processor": "landing_ai_ade"}
                    }
                    embedded_chunks = self.embedder.embed_document_chunks(document_data)
                else:
                    logger.warning(f"Landing AI ADE extraction failed: {ade_result.get('error')}")
                    logger.warning("Falling back to standard processing")
                    embedded_chunks = self._process_with_standard_method(pdf_path, chunk_size, chunk_overlap)
            else:
                logger.info("Using standard document processing")
                embedded_chunks = self._process_with_standard_method(pdf_path, chunk_size, chunk_overlap)
            
            logger.info(f"Created embeddings for {len(embedded_chunks)} chunks")
            
            # Add to vector database
            success = self.vector_db.add_documents(embedded_chunks)
            
            if success:
                # Store document metadata
                doc_name = Path(pdf_path).stem
                
                # Count different chunk types
                text_chunks = len([c for c in embedded_chunks if c.get('chunk_type') in ['text', 'table', 'form_field']])
                image_chunks = len([c for c in embedded_chunks if c.get('chunk_type') == 'image'])
                
                self.documents_metadata[doc_name] = {
                    'path': pdf_path,
                    'text_chunks': text_chunks,
                    'images': image_chunks,
                    'total_chunks': len(embedded_chunks),
                    'processor': 'landing_ai_ade' if self.landing_ai_processor else 'standard',
                    'processed_at': str(Path(pdf_path).stat().st_mtime)
                }
                
                logger.info(f"Successfully processed document: {pdf_path}")
                return True
            else:
                logger.error(f"Failed to add document to vector database: {pdf_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process document {pdf_path}: {e}")
            return False
    
    def query(self, question: str, top_k: Optional[int] = None, include_images: bool = True, 
              temperature: float = 0.7) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (uses config default if None)
            include_images: Whether to include image-based context
            temperature: Temperature for response generation
            
        Returns:
            Dictionary containing answer and supporting information
        """
        try:
            logger.info(f"Processing query: {question}")
            
            top_k_value = top_k or self.config.get('top_k', 5)
            
            # Get query embedding
            query_embedding = self.embedder.get_query_embedding(question)
            
            # Retrieve relevant chunks
            if include_images:
                text_results, image_results = self.vector_db.search_by_text_and_images(
                    query_embedding, top_k=top_k_value
                )
                all_results = text_results + image_results
                # Sort by similarity
                all_results.sort(key=lambda x: x['similarity'], reverse=True)
                relevant_chunks = all_results[:top_k_value]
            else:
                relevant_chunks = self.vector_db.search(
                    query_embedding, top_k=top_k_value, chunk_type='text'
                )
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Generate response
            response_data = self._generate_response(question, relevant_chunks, temperature)
            
            # Add metadata
            response_data.update({
                'query': question,
                'num_chunks_used': len(relevant_chunks),
                'chunks_metadata': [
                    {
                        'type': chunk['metadata']['chunk_type'],
                        'page': chunk['metadata']['page_number'],
                        'similarity': chunk['similarity']
                    }
                    for chunk in relevant_chunks
                ]
            })
            
            return response_data
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'answer': f"I apologize, but I encountered an error processing your question: {str(e)}",
                'error': True,
                'sources': []
            }
    
    def _generate_response(self, question: str, relevant_chunks: List[Dict[str, Any]], 
                          temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response using retrieved context with enhanced spatial information."""
        
        # Prepare context with spatial information
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(relevant_chunks):
            chunk_type = chunk['metadata']['chunk_type']
            page_num = chunk['metadata']['page_number']
            content = chunk['content']
            
            # Extract spatial information if available
            spatial_info = ""
            bounding_boxes = chunk['metadata'].get('bounding_boxes', [])
            if bounding_boxes:
                # Use first bounding box for primary location
                box = bounding_boxes[0]
                # Convert coordinates to human-readable location
                location_desc = self._describe_location(box)
                spatial_info = f" (located {location_desc})"
            
            if chunk_type == 'text':
                context_parts.append(f"[Text from page {page_num}{spatial_info}]: {content}")
                sources.append({
                    'type': 'text',
                    'page': page_num,
                    'content_preview': content[:100] + "..." if len(content) > 100 else content,
                    'similarity': chunk['similarity'],
                    'spatial_location': spatial_info.strip(),
                    'coordinates': bounding_boxes[0] if bounding_boxes else None
                })
            elif chunk_type == 'table':
                context_parts.append(f"[Table from page {page_num}{spatial_info}]: {content}")
                sources.append({
                    'type': 'table',
                    'page': page_num,
                    'content_preview': content[:150] + "..." if len(content) > 150 else content,
                    'similarity': chunk['similarity'],
                    'spatial_location': spatial_info.strip(),
                    'coordinates': bounding_boxes[0] if bounding_boxes else None
                })
            elif chunk_type == 'figure':
                context_parts.append(f"[Figure/Chart from page {page_num}{spatial_info}]: {content}")
                sources.append({
                    'type': 'figure',
                    'page': page_num,
                    'description': content,
                    'similarity': chunk['similarity'],
                    'spatial_location': spatial_info.strip(),
                    'coordinates': bounding_boxes[0] if bounding_boxes else None
                })
            elif chunk_type == 'image':
                context_parts.append(f"[Image description from page {page_num}{spatial_info}]: {content}")
                sources.append({
                    'type': 'image',
                    'page': page_num,
                    'filename': chunk['metadata'].get('filename', 'unknown'),
                    'description': content,
                    'similarity': chunk['similarity'],
                    'spatial_location': spatial_info.strip(),
                    'coordinates': bounding_boxes[0] if bounding_boxes else None
                })
        
        context = "\n\n".join(context_parts)
        
        # Create enhanced prompt with spatial awareness
        system_prompt = """You are an AI assistant helping users understand a document called "Miljoenen Nota". 
You have access to both text content and descriptions of charts, graphs, and images from the document.

IMPORTANT: You have precise spatial location information for each piece of content. When the context 
includes location information like "(located in upper left)" or "(located in center)", you should 
reference these locations in your response to help users find the information in the original document.

Your task is to answer questions based on the provided context. Be accurate and cite specific sources when possible.

Guidelines:
- Provide clear, comprehensive answers based on the context
- Reference specific pages AND locations when citing information (e.g., "according to the table in the upper right of page 3")
- If information comes from a chart or graph, mention both the chart type and its location
- Use spatial references to help users locate information in the original document
- If you cannot answer based on the provided context, say so clearly
- Be concise but thorough"""
        
        user_prompt = f"""Context from the document:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': sources,
                'context_used': context,
                'error': False
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                'answer': "I apologize, but I couldn't generate a response due to an error.",
                'sources': sources,
                'error': True,
                'error_message': str(e)
            }
    
    def _describe_location(self, bounding_box: Dict[str, float]) -> str:
        """
        Convert bounding box coordinates to human-readable location description.
        
        Args:
            bounding_box: Dict with 'left', 'top', 'right', 'bottom' coordinates (0-1 scale)
            
        Returns:
            Human-readable location description
        """
        left = bounding_box.get('left', 0)
        top = bounding_box.get('top', 0)
        right = bounding_box.get('right', 1)
        bottom = bounding_box.get('bottom', 1)
        
        # Calculate center points
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        
        # Determine horizontal position
        if center_x < 0.33:
            h_pos = "left"
        elif center_x > 0.67:
            h_pos = "right"
        else:
            h_pos = "center"
        
        # Determine vertical position
        if center_y < 0.33:
            v_pos = "upper"
        elif center_y > 0.67:
            v_pos = "lower"
        else:
            v_pos = "middle"
        
        # Combine positions
        if h_pos == "center" and v_pos == "middle":
            return "in the center of the page"
        elif h_pos == "center":
            return f"in the {v_pos} center of the page"
        elif v_pos == "middle":
            return f"on the {h_pos} side of the page"
        else:
            return f"in the {v_pos} {h_pos} of the page"
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get information about the current database state."""
        stats = self.vector_db.get_collection_stats()
        stats['processed_documents'] = self.documents_metadata
        return stats
    
    def clear_database(self):
        """Clear all documents from the database."""
        self.vector_db.clear_collection()
        self.documents_metadata = {}
        logger.info("Cleared database")
    
    def suggest_questions(self, document_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Suggest relevant questions based on the document content."""
        suggestions = [
            "What are the main financial highlights mentioned in this document?",
            "Can you summarize the key budget allocations?",
            "What trends are shown in the charts and graphs?",
            "What are the major policy changes discussed?",
            "How does this year's budget compare to previous years?",
        ]
        
        # TODO: Make this more dynamic based on actual document content
        # Could analyze document metadata to generate more specific questions
        
        return suggestions
    
    def _process_with_standard_method(self, pdf_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """Process document using standard PyMuPDF method as fallback."""
        document_data = self.document_processor.process_document_with_chunking(
            pdf_path, chunk_size, chunk_overlap
        )
        
        logger.info(f"Standard processing extracted {len(document_data['text_chunks'])} text chunks and "
                   f"{len(document_data['images'])} images")
        
        return self.embedder.embed_document_chunks(document_data)
