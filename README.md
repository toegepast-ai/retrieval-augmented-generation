# Spatial-Aware RAG with Landing AI ADE

A sophisticated Retrieval-Augmented Generation (RAG) system that processes PDF documents with **spatial intelligence**. Uses Landing AI's Agentic Document Engine (ADE) to extract not just content, but precise location information for enhanced source attribution.

## ✨ Key Features

- **🎯 Spatial Intelligence**: Exact page locations and bounding box coordinates for every piece of content
- **🤖 Landing AI ADE Integration**: Advanced document parsing with chunk type detection (text, tables, figures)
- **📄 Source Attribution**: Responses include precise references to original document locations
- **💾 JSON Export**: Complete spatial data export for advanced analysis
- **🚀 Streamlit Interface**: Interactive web application for document processing and querying
- **💰 Cost-Effective**: Smart caching system prevents duplicate processing charges

## 🏗️ Architecture

```
PDF Document → Landing AI ADE → Spatial Chunks → Vector Database → Location-Aware Responses
     ↓              ↓               ↓              ↓                    ↓
  Raw PDF    Advanced Parsing   Coordinates    Semantic Search    "Found on page 2,
                                & Metadata      & Retrieval        upper-left corner"
```

## 🚀 Quick Start

### 1. Setup
```bash
git clone <repository-url>
cd retrieval-augmented-generation
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### 2. API Keys Required
```bash
# .env file
OPENAI_API_KEY=your-openai-api-key
VISION_AGENT_API_KEY=your-landing-ai-api-key
USE_LANDING_AI=true
```

### 3. Run Application
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
├── src/                    # Core RAG pipeline
│   ├── rag_pipeline.py     # Main orchestration
│   ├── landing_ai_agentic_processor.py  # ADE integration
│   ├── embeddings.py       # Text embeddings
│   └── vector_database.py  # ChromaDB storage
├── config/                 # Configuration
├── data/                   # Data storage
├── streamlit_app.py        # Web interface
└── requirements.txt        # Dependencies
```

## 🎯 Usage Example

```python
from src.rag_pipeline import MultimodalRAG

config = {
    'openai_api_key': 'your-key',
    'vision_agent_api_key': 'your-landing-ai-key',
    'use_landing_ai': True,
    'chroma_persist_directory': './data/vector_db'
}

rag = MultimodalRAG(config)

# Process document with spatial intelligence
rag.process_document('document.pdf', save_ade_json=True)

# Query with location-aware responses
result = rag.query("What are the key findings?")
print(result['answer'])  # Includes spatial references
```

## 🔍 What Makes This Special

- **Spatial Awareness**: Unlike traditional RAG systems that lose location context, this system preserves and utilizes spatial information
- **Source Traceability**: Every response can tell you exactly where information came from in the original document
- **Advanced Parsing**: Landing AI ADE provides superior document understanding compared to basic text extraction
- **Cost Optimization**: Smart caching prevents reprocessing the same documents

## 📊 Supported Content Types

- ✅ **Text**: Paragraphs, headings, lists
- ✅ **Tables**: Structured data with spatial context
- ✅ **Figures**: Charts, graphs, diagrams
- ✅ **Marginalia**: Footnotes, annotations, side notes

## ⚙️ Configuration

Key settings in `.env`:
```bash
OPENAI_API_KEY=your-openai-key
VISION_AGENT_API_KEY=your-landing-ai-key
USE_LANDING_AI=true
CHUNK_SIZE=1000
TOP_K=5
```

---

**Built with**: Landing AI ADE • OpenAI • ChromaDB • Streamlit
