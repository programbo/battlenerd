from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
import uvicorn
import os
from pathlib import Path
import tempfile
from collections import Counter

from ingest import MarkdownIngester
from process import TextProcessor
from embed import EmbeddingGenerator
from store import VectorStore
import chromadb

# Get the project root directory (one level up from src)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Debug info
print(f"Data directory exists: {DATA_DIR.exists()}")
print(f"Data directory path: {DATA_DIR}")
print(f"Contents of data directory:")
for item in DATA_DIR.glob("*"):
    print(f"- {item.name}")

# Initialize components
ingester = MarkdownIngester(DATA_DIR)
processor = TextProcessor()
embedder = EmbeddingGenerator()
vector_store = VectorStore()

class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    min_confidence: Optional[float] = 0.0

    @validator('min_confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('min_confidence must be between 0 and 1')
        return v

    @validator('n_results')
    def validate_n_results(cls, v):
        if v < 1:
            raise ValueError('n_results must be greater than 0')
        return v

class QueryResponse(BaseModel):
    results: List[str]
    metadata: List[Dict]

class Document(BaseModel):
    content: str
    filename: str
    source: Optional[str] = None

class IngestRequest(BaseModel):
    documents: List[Document] = Field(..., description="List of documents to ingest")

app = FastAPI(title="Military Documents RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the vector store for relevant document sections.
    Optionally filter by confidence score and limit number of results.
    """
    try:
        # Check if vector store is empty
        if vector_store.collection.count() == 0:
            raise HTTPException(
                status_code=404,
                detail="No documents found in vector store. Please ingest documents first."
            )

        results = vector_store.query(
            query_text=request.query,
            n_results=request.n_results,
            min_confidence=request.min_confidence
        )

        if not results['documents'][0]:
            return QueryResponse(results=[], metadata=[])

        return QueryResponse(
            results=results['documents'][0],
            metadata=results['metadatas'][0]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/ingest", response_model=Dict[str, str])
async def ingest_documents(request: IngestRequest):
    """
    Ingest one or more documents provided in the request body.
    """
    try:
        if not request.documents:
            raise HTTPException(
                status_code=400,
                detail="No documents provided in request"
            )

        documents = [{
            "content": doc.content,
            "filename": doc.filename,
            "source": doc.source or f"api_upload/{doc.filename}"
        } for doc in request.documents]

        try:
            cleaned_documents = processor.clean_text(documents)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process documents: {str(e)}"
            )

        documents_with_embeddings = embedder.generate_embeddings(cleaned_documents)
        vector_store.store_documents(documents_with_embeddings)

        return {
            "status": "success",
            "message": f"Successfully ingested {len(documents)} documents"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

@app.post("/ingest/files", response_model=Dict[str, str])
async def ingest_from_files():
    """
    Ingest documents from the data directory on disk.
    """
    try:
        print(f"Starting ingestion from: {ingester.data_dir}")
        documents = ingester.load_markdown_files()
        print(f"Found {len(documents)} documents")

        if not documents:
            print("No documents found. Directory contents:")
            for item in Path(ingester.data_dir).glob("*"):
                print(f"- {item.name}")
            return {
                "status": "warning",
                "message": "No documents found in data directory"
            }

        cleaned_documents = processor.clean_text(documents)
        documents_with_embeddings = embedder.generate_embeddings(cleaned_documents)
        vector_store.store_documents(documents_with_embeddings)

        return {
            "status": "success",
            "message": f"Successfully ingested {len(documents)} documents from files"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/reset")
async def reset():
    """
    Deletes the collection from chromadb
    """
    client = chromadb.PersistentClient(path="chroma_db")
    try:
        client.delete_collection("markdown_documents")
        return {"status": "success"}
    except ValueError:
        return {"status": "error collection does not exist"}

@app.get("/stats")
async def get_stats():
    """
    Returns metadata about the contents of the vector store
    """
    try:
        # Get all documents and metadata
        results = vector_store.collection.get()

        if not results or not results['metadatas']:
            return {
                "status": "empty",
                "message": "No documents in vector store",
                "count": 0
            }

        metadatas = results['metadatas']

        # Collect statistics
        stats = {
            "total_sections": len(metadatas),
            "unique_documents": len(set(m['filename'] for m in metadatas)),
            "confidence_levels": Counter(m['confidence'] for m in metadatas if m.get('confidence')),
            "sections_by_document": Counter(m['filename'] for m in metadatas),
            "sections_by_title": Counter(m['section_title'] for m in metadatas if m.get('section_title')),
            "average_confidence": 0.0,
            "sources": Counter(m['source_document'] for m in metadatas if m.get('source_document'))
        }

        # Calculate average confidence score
        confidence_scores = [m.get('confidence_score', 0) for m in metadatas]
        if confidence_scores:
            stats["average_confidence"] = sum(confidence_scores) / len(confidence_scores)

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
