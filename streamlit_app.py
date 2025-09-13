import streamlit as st
import os
import sys
from pathlib import Path
import json
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import MultimodalRAG
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_rag():
    """Initialize the RAG pipeline."""
    if 'rag' not in st.session_state:
        config = {
            'openai_api_key': settings.openai_api_key,
            'vision_agent_api_key': settings.vision_agent_api_key,
            'use_landing_ai': settings.use_landing_ai,
            'text_embedding_model': settings.text_embedding_model,
            'chat_model': settings.chat_model,
            'chroma_persist_directory': settings.chroma_persist_directory,
            'collection_name': settings.collection_name,
            'chunk_size': settings.chunk_size,
            'chunk_overlap': settings.chunk_overlap,
            'top_k': settings.top_k,
            'image_output_dir': './data/extracted_images'
        }
        
        try:
            st.session_state.rag = MultimodalRAG(config)
            return True
        except Exception as e:
            st.error(f"Failed to initialize RAG pipeline: {e}")
            return False
    return True

def main():
    st.set_page_config(
        page_title="Miljoenen Nota RAG Demo",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 RAG Demo")
    st.markdown("""
    This demo showcases a Retrieval-Augmented Generation (RAG) system for PDF document processing.
    The system can process both text and visual content (charts, graphs) from PDF documents.
    """)
    
    # Check if OpenAI API key is configured
    if not settings.openai_api_key:
        st.error("⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        st.info("Copy .env.example to .env and add your OpenAI API key.")
        return
    
    # Show Landing AI status
    if settings.use_landing_ai and settings.vision_agent_api_key:
        st.sidebar.success("✅ Landing AI ADE: Enabled")
    elif settings.use_landing_ai and not settings.vision_agent_api_key:
        st.warning("⚠️ Landing AI enabled but API key not configured")
    else:
        st.info("📄 Using standard document processing (Landing AI disabled)")
    
    # Initialize RAG
    if not initialize_rag():
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Miljoenen Nota PDF",
            type=['pdf'],
            help="Upload the PDF document you want to analyze"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            upload_dir = Path("./data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🔄 Process Document"):
                # ADE options for Landing AI users
                save_ade_json = False
                ade_json_dir = "./data/ade_results"
                
                if settings.use_landing_ai and settings.vision_agent_api_key:
                    st.info("🤖 Using Landing AI ADE for enhanced document processing")
                    save_ade_json = True
                    ade_json_dir = "./data/ade_results"
                
                with st.spinner("Processing document... This may take a few minutes."):
                    try:
                        success = st.session_state.rag.process_document(
                            str(file_path),
                            save_ade_json=save_ade_json,
                            ade_json_dir=ade_json_dir
                        )
                        if success:
                            st.success("✅ Document processed successfully!")
                            if save_ade_json:
                                st.info("📄 Detailed ADE JSON results saved for advanced analysis")
                            st.session_state.document_processed = True
                        else:
                            st.error("❌ Failed to process document.")
                    except Exception as e:
                        st.error(f"Error processing document: {e}")
        
        # Database stats
        st.header("📊 Database Stats")
        if st.button("🔍 Show Stats"):
            try:
                stats = st.session_state.rag.get_database_stats()
                st.json(stats)
            except Exception as e:
                st.error(f"Error getting stats: {e}")
        
        # Clear database
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                st.session_state.rag.clear_database()
                st.success("Database cleared!")
                st.session_state.confirm_clear = False
                st.session_state.document_processed = False
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing the database.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Query interface
        question = st.text_input(
            "Enter your question about the document:",
            placeholder="What are the main budget allocations discussed in the document?"
        )
        
        col_query1, col_query2 = st.columns([3, 1])
        with col_query1:
            include_images = st.checkbox("Include charts/graphs in analysis", value=True)
        with col_query2:
            top_k = st.number_input("Results to retrieve", min_value=1, max_value=20, value=5)
        
        if st.button("🔍 Ask Question", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not st.session_state.get('document_processed', False):
                st.warning("Please upload and process a document first.")
            else:
                with st.spinner("Searching and generating answer..."):
                    try:
                        result = st.session_state.rag.query(
                            question, 
                            top_k=top_k, 
                            include_images=include_images
                        )
                        
                        if result.get('error'):
                            st.error(f"Error: {result.get('error_message', 'Unknown error')}")
                        else:
                            # Display answer
                            st.subheader("📝 Answer")
                            st.write(result['answer'])
                            
                            # Display sources
                            with st.expander("📚 Sources Used"):
                                for i, source in enumerate(result['sources']):
                                    st.write(f"**Source {i+1}:** {source['type'].title()} from page {source['page']}")
                                    
                                    # Show spatial information if available (ADE enhancement)
                                    if 'spatial_description' in source and source['spatial_description']:
                                        st.write(f"📍 **Location**: {source['spatial_description']}")
                                    
                                    if 'bounding_boxes' in source and source['bounding_boxes']:
                                        coords = source['bounding_boxes'][0]  # Show first bounding box
                                        st.write(f"🎯 **Coordinates**: ({coords['left']:.2f}, {coords['top']:.2f}) to ({coords['right']:.2f}, {coords['bottom']:.2f})")
                                    
                                    if source['type'] == 'text':
                                        st.write(f"Content preview: {source['content_preview']}")
                                        
                                        # Show chunk type if available (ADE enhancement)
                                        if 'chunk_type' in source:
                                            chunk_type = source['chunk_type']
                                            if hasattr(chunk_type, 'value'):
                                                chunk_type = chunk_type.value
                                            st.write(f"📄 **Type**: {chunk_type}")
                                            
                                    elif source['type'] == 'image':
                                        st.write(f"Description: {source['description']}")
                                        
                                    st.write(f"Similarity: {source['similarity']:.3f}")
                                    st.divider()
                            
                            # Store in session state for history
                            if 'chat_history' not in st.session_state:
                                st.session_state.chat_history = []
                            
                            st.session_state.chat_history.append({
                                'question': question,
                                'answer': result['answer'],
                                'sources': len(result['sources']),
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    
                    except Exception as e:
                        st.error(f"Error processing question: {e}")
    
    with col2:
        st.header("💡 Suggested Questions")
        
        # Get suggested questions
        suggestions = [
            "What are the main financial highlights?",
            "Summarize the key budget allocations",
            "What trends are shown in the charts?",
            "What policy changes are discussed?",
            "How does this compare to previous years?"
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{hash(suggestion)}"):
                st.session_state.suggested_question = suggestion
                # Use JavaScript to set the input value
                st.rerun()
        
        # Chat history
        if st.session_state.get('chat_history'):
            st.header("📝 Recent Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {chat['question'][:50]}..."):
                    st.write(f"**Asked:** {chat['timestamp']}")
                    st.write(f"**Sources used:** {chat['sources']}")
                    st.write(f"**Answer:** {chat['answer'][:200]}...")
    
    # Handle suggested question
    if st.session_state.get('suggested_question'):
        st.session_state.current_question = st.session_state.suggested_question
        del st.session_state.suggested_question
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Features:**
    - 📄 PDF text extraction and processing
    - 🖼️ Chart and graph analysis using GPT-4 Vision
    - 🔍 Semantic search across text and visual content
    - 💬 Natural language question answering
    - 📊 Multimodal retrieval-augmented generation
    """)

if __name__ == "__main__":
    main()
