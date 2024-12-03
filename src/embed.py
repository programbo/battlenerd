from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents: List[Dict[str, str]]) -> List[Dict]:
        for doc in documents:
            # Generate embeddings for each section
            for section in doc["sections"]:
                # Combine title and content for better context
                text = f"{section['title']}: {section['content']}"
                embedding = self.model.encode(text)
                section["embedding"] = embedding

                # Add confidence score if available
                if section.get("confidence"):
                    try:
                        numerator, denominator = map(int, section["confidence"].split("/"))
                        section["confidence_score"] = numerator / denominator
                    except (ValueError, AttributeError):
                        section["confidence_score"] = 0
        return documents
    
    def generate_report_embeddings(self, reports: List[Dict[str, str]]) -> List[Dict]:
        for report in reports:
            # generate embeddings for each report
            # combine report type and content for better context
            text = f"{report['type']}: {report['content']}"
            embedding = self.model.encode(text)
            report["embedding"] = embedding
        return reports
