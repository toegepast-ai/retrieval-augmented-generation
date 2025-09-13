# Ruimtelijk-Bewuste RAG met Landing AI ADE

Een geavanceerd Retrieval-Augmented Generation (RAG) systeem dat PDF-documenten verwerkt met **ruimtelijke intelligentie**. Gebruikt Landing AI's Agentic Document Engine (ADE) om niet alleen inhoud te extraheren, maar ook precieze locatie-informatie voor verbeterde bronverwijzing.

## Hoofdkenmerken

- **Ruimtelijke Intelligentie**: Exacte paginaposities en begrenzingscoördinaten voor elk stuk inhoud
- **Landing AI ADE Integratie**: Geavanceerde documentverwerking met chunk-type detectie (tekst, tabellen, figuren)
- **Bronverwijzing**: Antwoorden bevatten precieze referenties naar originele documentlocaties
- **JSON Export**: Complete ruimtelijke data export voor geavanceerde analyse
- **Streamlit Interface**: Interactieve webapplicatie voor documentverwerking en queries
- **Kosteneffectief**: Slim caching systeem voorkomt dubbele verwerkingskosten

## Architectuur

```
PDF Document → Landing AI ADE → Spatial Chunks → Vector Database → Location-Aware Responses
     ↓              ↓               ↓              ↓                    ↓
  Raw PDF    Advanced Parsing   Coordinates    Semantic Search    "Gevonden op pagina 2,
                                & Metadata      & Retrieval        linkerbovenhoek"
```

## Snelstart

### 1. Setup
```bash
git clone <repository-url>
cd retrieval-augmented-generation
pip install -r requirements.txt
cp .env.example .env
# Bewerk .env met je API keys
```

### 2. Vereiste API Keys
```bash
# .env bestand
OPENAI_API_KEY=jouw-openai-api-key
VISION_AGENT_API_KEY=jouw-landing-ai-api-key
USE_LANDING_AI=true
```

### 3. Applicatie Starten
```bash
streamlit run streamlit_app.py
```

## Projectstructuur

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

## Gebruiksvoorbeeld

```python
from src.rag_pipeline import MultimodalRAG

config = {
    'openai_api_key': 'jouw-key',
    'vision_agent_api_key': 'jouw-landing-ai-key',
    'use_landing_ai': True,
    'chroma_persist_directory': './data/vector_db'
}

rag = MultimodalRAG(config)

# Process document with spatial intelligence
rag.process_document('document.pdf', save_ade_json=True)

# Query with location-aware responses
result = rag.query("Wat zijn de belangrijkste bevindingen?")
print(result['answer'])  # Includes spatial references
```

## Configuratie

Belangrijke instellingen in `.env`:
```bash
OPENAI_API_KEY=jouw-openai-key
VISION_AGENT_API_KEY=jouw-landing-ai-key
USE_LANDING_AI=true
CHUNK_SIZE=1000
TOP_K=5
```

## Roadmap & Toekomstige Verbeteringen

Deze repository dient als basis voor het verkennen van geavanceerde RAG-strategieën en enterprise-grade verbeteringen.

### Geavanceerde RAG Strategieën
- **Hybride Retrieval**: Combinatie van dense + sparse retrieval (BM25 + semantisch)
- **Multi-Stage Retrieval**: Grof-naar-fijn document filtering
- **Retrieval Fusion**: RRF (Reciprocal Rank Fusion) voor resultaatsamenvoeging
- **Agentic RAG**: Multi-agent systemen voor complexe documentanalyse
- **Iteratieve Verfijning**: Zelfverbeterende query expansie en verfijning

### Database & Storage Backends
- **Neo4j Integratie**: Graph-gebaseerde kennisrepresentatie
  - Entity relationship mapping
  - Knowledge graph constructie uit documenten
  - Graph-traversal gebaseerde retrieval
- **Pinecone**: Hoogperformante vector database
- **PostgreSQL + pgvector**: Relationele + vector hybride opslag
- **Elasticsearch**: Full-text search integratie
- **Weaviate**: ML-native vector database

### Infrastructuur & DevOps
- **Containerisatie**
  - Multi-stage Docker builds voor productie optimalisatie
  - Docker Compose voor lokale ontwikkeling
  - Kubernetes manifests voor orkestratie
- **Infrastructure as Code (IaC)**
  - Terraform modules voor AWS/Azure/GCP deployment
  - Pulumi voor cloud-agnostische infrastructuur
  - Helm charts voor Kubernetes deployments
- **CI/CD Pipeline**
  - GitHub Actions voor geautomatiseerd testen
  - Multi-environment deployment (dev/staging/prod)
  - Geautomatiseerde security scanning en dependency updates
  - Performance benchmarking in CI

### Cloud & Hosting Oplossingen
- **AWS Deployment**
  - ECS/Fargate voor serverless containers
  - Lambda functies voor API endpoints
  - S3 + CloudFront voor statische assets
  - RDS voor metadata opslag
- **Azure Integratie**
  - Azure Container Instances
  - Cognitive Services integratie
  - Azure Functions voor event-driven processing
- **GCP Oplossingen**
  - Cloud Run voor serverless deployment
  - Vertex AI voor ML model serving
  - Cloud Storage voor document assets

### Security & Compliance
- **Authenticatie & Autorisatie**
  - OAuth 2.0 / OIDC integratie
  - Role-based access control (RBAC)
  - API key management en rotatie
- **Compliance Features**
  - GDPR-compliant data handling
  - Audit logging en traceerbaarheid
  - Data encryptie at rest en in transit

### Monitoring & Observability
- **Performance Monitoring**
  - Prometheus + Grafana dashboards
  - Application Performance Monitoring (APM)
  - Cost tracking en optimalisatie
- **Logging & Tracing**
  - Gestructureerde logging met ELK stack
  - Distributed tracing met Jaeger
  - Error tracking met Sentry

### Development & Testing
- **Testing Framework**
  - Unit tests met pytest
  - Integratie tests voor API endpoints
  - Load testing met Locust
  - RAG-specifieke evaluatie metrics
- **Documentatie**
  - API documentatie met FastAPI/Swagger
  - Architecture decision records (ADRs)
  - Runbooks voor operations

### Gespecialiseerde Use Cases
- **Multi-Language Support**: Documentverwerking in meerdere talen
- **Audio Integratie**: Speech-to-text en voice queries
- **Mobile Interface**: React Native of Flutter app
- **Samenwerking**: Multi-user documentanalyse en delen

---

**Gebouwd met**: Landing AI ADE • OpenAI • ChromaDB • Streamlit
