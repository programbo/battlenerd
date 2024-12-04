import os
import chromadb
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from pathlib import Path
from typing import Union
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware
from generator import TextGenerator
from cache import QueryCache
from ingest import MarkdownIngester
from process import TextProcessor
from embed import EmbeddingGenerator
from store import VectorStore
from routes.validate_templates import router as validation_router
from template_validator import TemplateValidator
import logging
from datetime import datetime

# Set up logging at the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory (one level up from src)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Initialize with Path object instead of string
ingester = MarkdownIngester(DATA_DIR)

app = FastAPI(title="Military Documents RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize components
processor = TextProcessor()
embedder = EmbeddingGenerator()
vector_store = VectorStore()
generator = TextGenerator()
cache = QueryCache()
validator = TemplateValidator()

# Track online status
is_online = True
llm_endpoint = "https://api.anthropic.com/v1/messages"

def check_online_status():
    try:
        requests.get("https://www.google.com", timeout=1)
        return True
    except:
        return False

class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = Field(default=10, le=50)
    min_confidence: Optional[float] = 0.0
    force_offline: Optional[bool] = False
    use_template: Optional[bool] = True

class QueryResponse(BaseModel):
    results: List[str]
    metadata: List[Dict]

class Locstat(BaseModel):
    call_sign: str = Field(..., example="A12")
    grid_reference: str = Field(...)
    status: str = Field(...)
    additional_remarks: str = Field(...)

class Casevac(BaseModel):
    call_sign: str = Field(...)
    pick_up_point: str = Field(...)
    number_casualties_lying: str = Field(...)
    number_casualties_walking: str = Field(...)
    nature_of_injuries: str = Field(...)
    priority: str = Field(...)
    requirements_for_specialist_equipment: str = Field(...)
    call_sign_and_frequency: str = Field(...)
    hoist_requirements: str = Field(...)
    additional_remarks: str = Field(...)

class Sitrep(BaseModel):
    call_sign: str = Field(...)
    grid_reference: str = Field(...)
    status: str = Field(...)
    strength: str = Field(...)
    morale: str = Field(...)
    rations: str = Field(...)
    water: str = Field(...)
    additional_remarks: str = Field(...)

class Report(BaseModel):
    type: str
    content: Union[Locstat]

class IngestDocument(BaseModel):
    content: str
    filename: str
    source: str

class IngestRequest(BaseModel):
    documents: List[IngestDocument]

class QueryResponseWithText(QueryResponse):
    generated_text: str
    from_cache: bool = False
    generated_by: str = "unknown"  # Can be "claude", "local_llm", or "vector_store"

class RiskScore(BaseModel):
    category: str
    score: float = Field(..., ge=0, le=1)
    rationale: str

class RiskAssessment(BaseModel):
    operation_name: str = "Op Steel Sentinel"
    risks: List[RiskScore]
    timestamp: str
    summary: str

@app.post("/query", response_model=QueryResponseWithText)
async def query_documents(request: QueryRequest):
    """
    Query the vector store for relevant document sections.
    Optionally filter by confidence score and limit number of results.
    """
    try:
        results = vector_store.query(
            query_text=request.query,
            n_results=request.n_results,
            min_confidence=request.min_confidence
        )

        # Check cache first
        context = "\n".join(results['documents'][0])
        cached_response = cache.get(
            query=request.query,
            context=context,
            offline=request.force_offline
        )

        if cached_response:
            logger.info("Returning cached response")
            return QueryResponseWithText(**cached_response)

        # Generate new response if not cached
        global is_online
        is_online = False if request.force_offline else check_online_status()
        generated_by = "unknown"

        if is_online:
            logger.info("System is online, attempting to use Claude API")
            try:
                # Create a relevance-aware context
                structured_context = []

                # Group content by source document and section
                grouped_data = {}
                for content, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    source_key = f"{metadata['source_document']} - {metadata['section_title']}"
                    if source_key not in grouped_data:
                        grouped_data[source_key] = []

                    # Clean up the content
                    cleaned_content = content.replace(f"Confidence: {metadata['confidence']} Source: {metadata['source_document']}", "").strip()

                    grouped_data[source_key].append({
                        'content': cleaned_content,
                        'confidence': metadata['confidence'],
                        'filename': metadata['filename']
                    })

                # Create introduction
                unique_sources = set(meta['source_document'] for meta in results['metadatas'][0])
                relevance_intro = (
                    f"The following information was retrieved from {len(unique_sources)} "
                    f"source(s) based on your query. Each section includes a confidence "
                    f"score indicating its relevance.\n\n"
                )

                # Format each section's data
                for source_key, entries in grouped_data.items():
                    source_doc, section = source_key.split(" - ")
                    section_content = []

                    for entry in entries:
                        section_content.append(
                            f"Confidence: {entry['confidence']}\n"
                            f"{entry['content']}\n"
                        )

                    structured_context.append(
                        f"Source: {source_doc}\nSection: {section}\n---\n" + "\n".join(section_content)
                    )

                # Load all markdown files
                all_markdown = []
                for file in DATA_DIR.glob("**/*.md"):
                    with open(file, 'r') as f:
                        content = f.read()
                        all_markdown.append(f"File: {file.name}\n---\n{content}\n\n")

                full_context = "\n".join(all_markdown)
                print(f"Full context length: {len(full_context)}")
                # Check if context is too large (staying well under 200k to leave room for response)
                if len(full_context) > 150000:
                    logger.warning("Context too large, falling back to vector store results")
                    formatted_context = relevance_intro + "\n\n".join(structured_context)
                else:
                    formatted_context = ("Here are all available documents:\n\n" +
                                       full_context +
                                       "\n\nAnd here are the most relevant sections:\n\n" +
                                       relevance_intro +
                                       "\n\n".join(structured_context))

                response = requests.post(
                    llm_endpoint,
                    headers={
                        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": "claude-3-sonnet-20240229",
                        "max_tokens": 1024,
                        "messages": [{
                            "role": "user",
                            "content": f"Based on this context:\n\n{formatted_context}\n\nPlease answer this question: {request.query}"
                        }]
                    }
                )

                # Add debug logging
                logger.info(f"Claude API Status Code: {response.status_code}")
                logger.info(f"Claude API Response: {response.text}")

                response_json = response.json()
                generated_text = response_json["content"][0]["text"]
                generated_by = "claude"
                logger.info("Successfully generated response using Claude API")
            except Exception as e:
                logger.error(f"Claude API call failed: {str(e)}")
                logger.error(f"Full error: {repr(e)}")
                logger.info("Falling back to local LLM")
                generated_text = generator.generate_response(
                    request.query,
                    results['documents'][0],
                    use_template=request.use_template
                )
                generated_by = "local_llm"
        else:
            logger.info("System is offline, using local LLM")
            generated_text = generator.generate_response(
                request.query,
                results['documents'][0],
                use_template=request.use_template
            )
            generated_by = "local_llm"

        response = QueryResponseWithText(
            results=results['documents'][0],
            metadata=results['metadatas'][0],
            generated_text=generated_text,
            from_cache=False,
            generated_by=generated_by
        )

        # Cache the response
        cache.set(
            query=request.query,
            context=context,
            offline=request.force_offline,
            response=response.dict()
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents(request: Optional[IngestRequest] = None):
    """
    Ingest documents either from request body or data directory.
    If request body is provided, ingest those documents.
    If no request body, load and ingest documents from data directory.
    """
    try:
        # Determine source of documents
        if request and request.documents:
            documents = [{
                "content": doc.content,
                "filename": doc.filename,
                "source": doc.source
            } for doc in request.documents]
            print(f"Processing {len(documents)} documents from request")
        else:
            print(f"Loading documents from: {ingester.data_dir}")
            documents = ingester.load_markdown_files()
            print(f"Found {len(documents)} documents in directory")

            if not documents:
                return {
                    "status": "warning",
                    "message": "No documents found in data directory"
                }

        # Selectively invalidate cache for affected documents
        invalidated_entries = cache.invalidate_documents(documents)

        cleaned_documents = processor.clean_text(documents)
        documents_with_embeddings = embedder.generate_embeddings(cleaned_documents)
        vector_store.store_documents(documents_with_embeddings)

        return {
            "status": "success",
            "message": f"Ingested {len(documents)} documents from {'request' if request else 'files'}. Invalidated {invalidated_entries} cache entries."
        }
    except Exception as e:
        if isinstance(e, ValueError):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/report")
async def ingest_report(report: Report):
    """
    Ingests a new report into chromadb
    """
    try:
        reports_with_embeddings = embedder.generate_report_embeddings([report.model_dump()])
        vector_store.store_report(reports_with_embeddings)
        return {
            "status" : "success",
            "message" : "Successfully ingested report"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    """
    return {"status": "healthy"}

@app.delete("/reset")
async def reset():
    """
    Empties the collection from chromadb
    """
    global vector_store  # Use the global instance
    vector_store.clear()
    return {"status": "success"}

@app.get("/stats")
async def get_stats():
    """
    Returns metadata about the contents of the vector store
    """
    try:
        results = vector_store.collection.get()

        if not results or not results['metadatas']:
            return {
                "status": "empty",
                "message": "No documents in vector store",
                "count": 0
            }

        metadatas = results['metadatas']

        # Debug logging
        print("Confidence scores in metadata:")
        for m in metadatas:
            print(f"confidence_score: {m.get('confidence_score')}, type: {type(m.get('confidence_score'))}")

        # Calculate average confidence score - with additional error handling
        confidence_scores = []
        for m in metadatas:
            score = m.get('confidence_score')
            if score is not None and score != '':  # Skip empty strings
                try:
                    confidence_scores.append(float(score))
                except ValueError as e:
                    print(f"Failed to convert score: {score}, type: {type(score)}")
                    continue

        # Rest of the stats calculation
        stats = {
            "total_sections": len(metadatas),
            "unique_documents": len(set(m['filename'] for m in metadatas)),
            "confidence_levels": Counter(m['confidence'] for m in metadatas if m.get('confidence')),
            "sections_by_document": Counter(m['filename'] for m in metadatas),
            "sections_by_title": Counter(m['section_title'] for m in metadatas if m.get('section_title')),
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            "sources": Counter(m['source_document'] for m in metadatas if m.get('source_document'))
        }

        return {
            "status": "success",
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )

@app.get("/risks", response_model=RiskAssessment)
async def assess_risks():
    """
    Analyzes documents to assess operational risks for Op Steel Sentinel
    """
    try:
        prompt = """Based on the provided operational documents, provide a risk assessment for Op Steel Sentinel.
        Return your response in this exact JSON format:
        {
            "risks": [
                {
                    "category": "Mission Success",
                    "score": 0.7,
                    "rationale": "Brief explanation"
                },
                ... continue for all categories ...
            ],
            "summary": "Brief overall assessment"
        }

        You MUST include scores and rationales for exactly these categories in this order:
        1. Mission Success
        2. Operational Readiness
        3. Cyber Security
        4. Supply Chain
        5. Intelligence
        6. Personnel
        7. Environmental
        8. Political
        9. Financial
        10. Time

        Each score must be between 0 and 1."""

        response = requests.post(
            llm_endpoint,
            headers={
                "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1024,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            }
        )

        response_json = response.json()
        assessment = response_json["content"][0]["text"]

        # Parse the JSON response
        import json
        from datetime import datetime

        parsed = json.loads(assessment)
        return RiskAssessment(
            risks=parsed["risks"],
            timestamp=datetime.now().isoformat(),
            summary=parsed["summary"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount validation router
app.include_router(
    validation_router,
    tags=["validation"]
)

# Pass dependencies to router
validation_router.validator = validator
validation_router.vector_store = vector_store
validation_router.processor = processor


def start_server():
    """
    Start the FastAPI server using uvicorn.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
