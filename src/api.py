from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import os
from pathlib import Path
from typing import Union
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware

from ingest import MarkdownIngester
from process import TextProcessor
from embed import EmbeddingGenerator
from store import VectorStore
import chromadb

# Get the project root directory (one level up from src)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Initialize with Path object instead of string
ingester = MarkdownIngester(DATA_DIR)  # Try without str() conversion

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

class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    min_confidence: Optional[float] = 0.0

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

@app.post("/query", response_model=QueryResponse)
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
        return QueryResponse(
            results=results['documents'][0],  # First query's results
            metadata=results['metadatas'][0]  # First query's metadata
        )
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
            # Use documents from request body
            documents = [{
                "content": doc.content,
                "filename": doc.filename,
                "source": doc.source
            } for doc in request.documents]
            print(f"Processing {len(documents)} documents from request")
        else:
            # Load documents from data directory
            print(f"Loading documents from: {ingester.data_dir}")
            documents = ingester.load_markdown_files()
            print(f"Found {len(documents)} documents in directory")

            if not documents:
                return {
                    "status": "warning",
                    "message": "No documents found in data directory"
                }

        cleaned_documents = processor.clean_text(documents)
        documents_with_embeddings = embedder.generate_embeddings(cleaned_documents)
        vector_store.store_documents(documents_with_embeddings)

        return {
            "status": "success",
            "message": f"Ingested {len(documents)} documents from {'request' if request else 'files'}"
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

def start_server():
    """
    Start the FastAPI server using uvicorn.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
