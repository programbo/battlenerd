import os
from pathlib import Path
from lib.rag import RAGProcessor

def load_markdown_files(data_dir: str = "data") -> list[str]:
    markdown_contents = []

    # Walk through all files in data directory
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.md', '.markdown')):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        markdown_contents.append(content)
                        print(f"Loaded: {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    return markdown_contents

def main():
    # Initialize RAG
    rag = RAGProcessor()

    # Load all markdown files
    print("Loading markdown files...")
    contents = load_markdown_files()

    if not contents:
        print("No markdown files found in /data directory!")
        return

    print(f"Found {len(contents)} markdown files")

    # Process files
    print("Processing files and generating embeddings...")
    rag.process_markdown_files(contents)

    print("RAG training completed!")

if __name__ == "__main__":
    main()
