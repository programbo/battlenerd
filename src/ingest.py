import os
from pathlib import Path
from typing import List, Dict

class MarkdownIngester:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_markdown_files(self) -> List[Dict[str, str]]:
        documents = []
        for file_path in self.data_dir.glob("**/*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "source": str(file_path),
                    "filename": file_path.name
                })
        return documents
