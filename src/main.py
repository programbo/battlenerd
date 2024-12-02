import os
from ingest import MarkdownIngester
from process import TextProcessor
from embed import EmbeddingGenerator
from store import VectorStore
from api import start_server
import argparse

def ingest_documents():
    # Initialize components
    data_dir = "data"  # Change this to your markdown files directory
    ingester = MarkdownIngester(data_dir)
    processor = TextProcessor()
    embedder = EmbeddingGenerator()
    vector_store = VectorStore()

    # Pipeline execution
    print("Loading markdown files...")
    documents = ingester.load_markdown_files()

    print("Processing text...")
    cleaned_documents = processor.clean_text(documents)

    print("Generating embeddings...")
    documents_with_embeddings = embedder.generate_embeddings(cleaned_documents)

    print("Storing vectors...")
    vector_store.store_documents(documents_with_embeddings)

    print("Pipeline completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Military Documents RAG System')
    parser.add_argument('--api', action='store_true', help='Start the API server')
    args = parser.parse_args()

    if args.api:
        print("Starting API server...")
        start_server()
    else:
        ingest_documents()

if __name__ == "__main__":
    main()
