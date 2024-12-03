from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os
from pathlib import Path

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

# Initialize with Path object instead of string
ingester = MarkdownIngester(DATA_DIR)  # Try without str() conversion

app = FastAPI(title="Military Documents RAG API")

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
async def ingest_documents():
    """
    Trigger reingestion of documents from the data directory.
    """
    try:
        print(f"Starting ingestion from: {ingester.data_dir}")
        documents = ingester.load_markdown_files()
        print(f"Found {len(documents)} documents")
        if not documents:
            print("No documents found. Directory contents:")
            for item in Path(ingester.data_dir).glob("*"):
                print(f"- {item.name}")

        cleaned_documents = processor.clean_text(documents)
        documents_with_embeddings = embedder.generate_embeddings(cleaned_documents)
        vector_store.store_documents(documents_with_embeddings)

        return {
            "status": "success",
            "message": f"Successfully ingested {len(documents)} documents"
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
    
def start_server():
    """
    Start the FastAPI server using uvicorn.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
