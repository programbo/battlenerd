from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

from ingest import MarkdownIngester
from process import TextProcessor
from embed import EmbeddingGenerator
from store import VectorStore

app = FastAPI(title="Military Documents RAG API")

# Initialize components
ingester = MarkdownIngester("data")
processor = TextProcessor()
embedder = EmbeddingGenerator()
vector_store = VectorStore()

class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    min_confidence: Optional[float] = 0.0

class QueryResponse(BaseModel):
    results: List[Dict]
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
        # Pipeline execution
        documents = ingester.load_markdown_files()
        cleaned_documents = processor.clean_text(documents)
        documents_with_embeddings = embedder.generate_embeddings(cleaned_documents)
        vector_store.store_documents(documents_with_embeddings)

        return {"status": "success", "message": "Documents ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    """
    return {"status": "healthy"}

def start_server():
    """
    Start the FastAPI server using uvicorn.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
