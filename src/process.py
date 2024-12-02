import re
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Section:
    title: str
    content: str
    confidence: str = None
    source: str = None

class TextProcessor:
    def __init__(self):
        self.markdown_patterns = {
            "headers": r"#{1,6}\s+",
            "links": r"\[([^\]]+)\]\([^\)]+\)",
            "code_blocks": r"```[\s\S]*?```",
            "inline_code": r"`[^`]+`",
            "confidence": r"Confidence:\s*(\d+/\d+)\s*Source:\s*([^\n]+)",
            "section_header": r"^##\s+(.+)$"
        }

    def extract_sections(self, text: str) -> List[Section]:
        """Split document into logical sections for better context."""
        sections = []
        current_section = None
        current_content = []

        for line in text.split('\n'):
            # Check for section header
            section_match = re.match(self.markdown_patterns["section_header"], line, re.MULTILINE)

            if section_match:
                # Save previous section if exists
                if current_section:
                    sections.append(Section(
                        title=current_section,
                        content='\n'.join(current_content).strip()
                    ))
                current_section = section_match.group(1)
                current_content = []
            elif current_section:
                # Check for confidence ratings
                conf_match = re.match(self.markdown_patterns["confidence"], line)
                if conf_match:
                    sections[-1].confidence = conf_match.group(1)
                    sections[-1].source = conf_match.group(2)
                else:
                    current_content.append(line)

        # Add final section
        if current_section:
            sections.append(Section(
                title=current_section,
                content='\n'.join(current_content).strip()
            ))

        return sections

    def clean_text(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        cleaned_documents = []

        for doc in documents:
            # Extract sections
            sections = self.extract_sections(doc["content"])

            # Process each section
            processed_sections = []
            for section in sections:
                text = section.content

                # Remove markdown formatting
                text = re.sub(self.markdown_patterns["code_blocks"], "", text)
                text = re.sub(self.markdown_patterns["links"], r"\1", text)
                text = re.sub(self.markdown_patterns["headers"], "", text)
                text = re.sub(self.markdown_patterns["inline_code"], "", text)

                # Clean whitespace
                text = " ".join(text.split())

                # Create structured section with metadata
                processed_sections.append({
                    "title": section.title,
                    "content": text,
                    "confidence": section.confidence,
                    "source": section.source
                })

            # Create document with sections
            cleaned_documents.append({
                "sections": processed_sections,
                "source": doc["source"],
                "filename": doc["filename"]
            })

        return cleaned_documents
