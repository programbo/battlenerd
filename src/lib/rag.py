from sentence_transformers import SentenceTransformer
import chromadb
from typing import List
import numpy as np

class RAGProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("markdown_docs")

    def process_markdown_files(self, contents: List[str]):
        # Simple chunking - you might want to make this more sophisticated
        chunks = []
        for i, content in enumerate(contents):
            # Split on double newlines for markdown-aware chunking
            doc_chunks = content.split('\n\n')
            chunks.extend([
                (f"doc_{i}_chunk_{j}", chunk.strip())
                for j, chunk in enumerate(doc_chunks)
                if chunk.strip()
            ])

        # Unzip the chunks into IDs and texts
        ids, texts = zip(*chunks)

        # Generate embeddings and add to ChromaDB
        embeddings = self.model.encode(texts)
        self.collection.add(
            ids=list(ids),
            documents=list(texts),
            embeddings=embeddings.tolist()
        )

    def query(self, query: str, n_results: int = 4):
        query_embedding = self.model.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        return results['documents'][0]  # First list since we only had one query
