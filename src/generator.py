from typing import List, Dict, Optional
from llama_cpp import Llama
import os
from pathlib import Path
from string import Template
import yaml
from download_model import ensure_model_downloaded
from dotenv import load_dotenv

# Load environment variables at the top of your file
load_dotenv()

class TextGenerator:
    def __init__(self):
        model_path = ensure_model_downloaded()
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=4
        )

        # Load templates from files
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict[str, Template]]:
        """Load all templates from the templates directory"""
        templates = {}
        template_dir = Path("templates")

        if not template_dir.exists():
            print(f"Warning: Template directory not found at {template_dir}")
            return {}

        # Load each template type
        for type_dir in template_dir.iterdir():
            if type_dir.is_dir():
                templates[type_dir.name] = {}
                for template_file in type_dir.glob("*.yaml"):
                    # Skip requirement files
                    if template_file.stem.endswith('_requirements'):
                        continue

                    try:
                        with open(template_file) as f:
                            template_data = yaml.safe_load(f)
                            template_name = template_file.stem
                            templates[type_dir.name][template_name] = {
                                'template': Template(template_data['template']),
                                'variables': template_data.get('variables', [])
                            }
                    except Exception as e:
                        print(f"Error loading template {template_file}: {e}")

        return templates

    def detect_document_type(self, context: str) -> Optional[str]:
        """Detect the type of document based on content and metadata"""
        lower_context = context.lower()

        if any(term in lower_context for term in ["capabilities", "military capabilities", "cyber capabilities"]):
            return "capabilities"
        elif any(term in lower_context for term in ["tactics", "procedure", "drill", "formation"]):
            return "tactics"
        elif any(term in lower_context for term in ["sitrep", "situation report", "status report"]):
            return "sitrep"
        elif any(term in lower_context for term in ["imap", "information management", "analysis process"]):
            # Detect IMAP level
            if "strategic" in lower_context:
                return ("imap", "strategic")
            elif "operational" in lower_context:
                return ("imap", "operational")
            elif "tactical" in lower_context:
                return ("imap", "tactical")
            return ("imap", "default")

        return None

    def _get_template(self, doc_type: str, subtype: str = "default") -> Optional[Dict]:
        """Get a specific template by type and subtype"""
        return self.templates.get(doc_type, {}).get(subtype)

    def extract_template_values(self, doc_type: str, context: str) -> Dict[str, str]:
        """Extract values to fill in templates based on document type"""
        template_data = self._get_template(doc_type)
        if not template_data:
            return {}

        required_vars = template_data['variables']
        if doc_type == "capabilities":
            return {
                "entity": context.split("\n")[0],
                "capabilities": "\n".join("- " + line.strip() for line in context.split("\n") if line.strip()),
                "confidence": "High",
                "source": "Military Documentation"
            }
        elif doc_type == "tactics":
            return {
                "name": context.split("\n")[0],
                "steps": "\n".join("- " + line.strip() for line in context.split("\n") if line.strip()),
                "considerations": "Maintain security and communication throughout",
                "source": "Tactical Manual"
            }
        elif doc_type == "imap":
            # Extract sections based on IMAP headers
            sections = {}
            current_section = None
            content_lines = []

            for line in context.split('\n'):
                if line.strip().startswith('1. INFORMATION'):
                    current_section = 'information_summary'
                elif line.strip().startswith('2. MEANING'):
                    current_section = 'meaning'
                elif line.strip().startswith('3. ACTION'):
                    current_section = 'action'
                elif line.strip().startswith('4. PREDICTION'):
                    current_section = 'prediction'
                elif current_section:
                    content_lines.append(line.strip())

            return {
                "information_summary": "\n".join(content_lines),
                "known_facts": "- " + "\n- ".join([l for l in content_lines if l.startswith("KNOWN:")]),
                "assumptions": "- " + "\n- ".join([l for l in content_lines if l.startswith("ASSUMED:")]),
                "gaps": "- " + "\n- ".join([l for l in content_lines if l.startswith("UNKNOWN:")]),
                "required_actions": "- " + "\n- ".join([l for l in content_lines if l.startswith("ACTION:")]),
                "priority": "HIGH",  # Could be extracted from content
                "likely_outcome": next((l for l in content_lines if "LIKELY:" in l), ""),
                "other_possibilities": "- " + "\n- ".join([l for l in content_lines if "POSSIBLE:" in l]),
                "warning_signs": "- " + "\n- ".join([l for l in content_lines if "WARNING:" in l]),
                "info_reliability": "B2",  # Could be extracted from metadata
                "analytical_confidence": "MEDIUM",  # Could be extracted from metadata
                "recommendations": "- " + "\n- ".join([l for l in content_lines if "RECOMMEND:" in l])
            }
        return {}

    def generate_response(self, query: str, context_docs: List[str], use_template: bool = False) -> str:
        context = "\n\n".join(context_docs)

        if use_template:
            doc_type = self.detect_document_type(context)
            template_data = self._get_template(doc_type)
            if template_data:
                template_values = self.extract_template_values(doc_type, context)
                return template_data['template'].safe_substitute(template_values)

        # Fallback to general LLM response
        prompt = f"""Based on the following information, please answer the question.
        If you cannot answer based solely on the provided information, say so.

        Information:
        {context}

        Question: {query}

        Answer:"""

        response = self.model(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            stop=["Question:", "\n\n"]
        )

        return response['choices'][0]['text'].strip()
