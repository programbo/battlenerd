import os
from ingest import MarkdownIngester
from process import TextProcessor
from embed import EmbeddingGenerator
from store import VectorStore
from pathlib import Path
from download_model import ensure_model_downloaded
from api import app, start_server
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

def validate_env_vars():
    required_vars = ["ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


def main():
	# parser = argparse.ArgumentParser(description='Military Documents RAG System')
	# parser.add_argument('--init', action='store_true', help='Injest initial training data')
	# args = parser.parse_args()

	# if args.init:
	# 	print("Injesting initial training data...")
	# 	ingest_documents()

	validate_env_vars()

	# Ensure model is downloaded before starting server
	print("Checking model files...")
	ensure_model_downloaded()

if __name__ == "__main__":
	main()
	print("Starting API server...")
	start_server()
