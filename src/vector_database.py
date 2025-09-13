import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Manages ChromaDB for storing and retrieving document embeddings."""
    
    def __init__(self, persist_directory: str, collection_name: str = "rag_collection"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = collection_name
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Multimodal RAG collection for documents"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, embedded_chunks: List[Dict[str, Any]]) -> bool:
        """
        Add embedded document chunks to the vector database.
        
        Args:
            embedded_chunks: List of chunks with embeddings from MultimodalEmbedder
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for chunk in embedded_chunks:
                # Generate unique ID
                chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))
                ids.append(chunk_id)
                
                # Extract embedding
                embeddings.append(chunk['embedding'])
                
                # Prepare metadata
                metadata = {
                    'chunk_type': chunk['chunk_type'],
                    'page_number': chunk['page_number'],
                    'source': chunk.get('metadata', {}).get('source', 'unknown')
                }
                
                # Add type-specific metadata
                if chunk['chunk_type'] == 'image':
                    metadata.update({
                        'filename': chunk.get('filename', ''),
                        'image_index': chunk.get('image_index', 0),
                        'size': str(chunk.get('size', '')),
                    })
                
                metadatas.append(metadata)
                
                # Use description or content as document text
                document_text = chunk.get('description', chunk.get('content', ''))
                documents.append(document_text)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Added {len(embedded_chunks)} chunks to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector database: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
               chunk_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks based on query embedding.
        
        Args:
            query_embedding: Embedding of the search query
            top_k: Number of results to return
            chunk_type: Filter by chunk type ('text' or 'image')
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Prepare search parameters
            search_kwargs = {
                'query_embeddings': [query_embedding],
                'n_results': top_k,
                'include': ['metadatas', 'documents', 'distances']
            }
            
            # Add filter if specified
            if chunk_type:
                search_kwargs['where'] = {'chunk_type': chunk_type}
            
            # Perform search
            results = self.collection.query(**search_kwargs)
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_text_and_images(self, query_embedding: List[float], top_k: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Search for both text and image chunks separately.
        
        Returns:
            Tuple of (text_results, image_results)
        """
        text_results = self.search(query_embedding, top_k=top_k, chunk_type='text')
        image_results = self.search(query_embedding, top_k=top_k, chunk_type='image')
        
        return text_results, image_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to understand content
            sample_results = self.collection.get(limit=10, include=['metadatas'])
            
            # Count by type
            text_count = 0
            image_count = 0
            for metadata in sample_results.get('metadatas', []):
                if metadata.get('chunk_type') == 'text':
                    text_count += 1
                elif metadata.get('chunk_type') == 'image':
                    image_count += 1
            
            return {
                'total_chunks': count,
                'text_chunks_sample': text_count,
                'image_chunks_sample': image_count,
                'collection_name': self.collection_name
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all data from the collection."""
        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
            logger.info("Cleared collection")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
    
    def update_chunk(self, chunk_id: str, new_embedding: List[float], 
                     new_content: str, new_metadata: Dict[str, Any]):
        """Update a specific chunk in the database."""
        try:
            self.collection.update(
                ids=[chunk_id],
                embeddings=[new_embedding],
                documents=[new_content],
                metadatas=[new_metadata]
            )
            logger.info(f"Updated chunk: {chunk_id}")
        except Exception as e:
            logger.error(f"Failed to update chunk {chunk_id}: {e}")
