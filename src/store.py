import chromadb
from chromadb.config import Settings
from typing import List, Dict
import numpy as np
from datetime import datetime

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="markdown_documents",
            metadata={"hnsw:space": "cosine"}
        )

    def store_documents(self, documents: List[Dict]) -> None:
        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for doc_idx, doc in enumerate(documents):
            for section_idx, section in enumerate(doc["sections"]):
                # Create unique ID for each section
                section_id = f"{doc_idx}_{section_idx}"
                ids.append(section_id)

                # Store section embedding
                embeddings.append(section["embedding"].tolist())

                # Store section text
                texts.append(section["content"])

                # Create metadata
                metadata = {
                    "source": doc["source"],
                    "filename": doc["filename"],
                    "section_title": section["title"],
                    "confidence": section.get("confidence"),
                    "confidence_score": section.get("confidence_score"),
                    "source_document": section.get("source")
                }
                for key, value in metadata.items():
                    if value is None:
                        metadata[key] = ""
                metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def store_report(self, reports: List[Dict]) -> None:
        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for report in reports:
            # create an id based on the timestamp
            report_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
            ids.append(report_id)

            # store section embedding
            embeddings.append(report["embedding"].tolist())

            # store report text
            report_text = f"{report['type']}: {str(report['content'])}"
            texts.append(report_text)

            # create metadata
            metadata = {
                "source" : "received report",
                "confidence" : "5/5"
            }
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def query(self, query_text: str, n_results: int = 5, min_confidence: float = 0.0):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"confidence_score": {"$gte": min_confidence}} if min_confidence > 0 else None
        )
        return results

    def clear(self) -> None:
        """
        Clears all documents from the vector store by deleting and recreating the collection.
        """
        try:
            self.client.delete_collection("markdown_documents")
        except ValueError:
            # Collection might not exist, that's okay
            pass

        # Recreate the collection with the same settings
        self.collection = self.client.create_collection(
            name="markdown_documents",
            metadata={"hnsw:space": "cosine"}
        )
