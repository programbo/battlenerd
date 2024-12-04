from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ValidationResult:
    completion_percentage: float
    missing_fields: Dict[str, str]  # field: reason needed
    critical_gaps: List[str]        # showstopper missing data
    data_requests: List[Tuple[str, str, int]]  # (data_type, justification, priority 1-5)

class TemplateValidator:
    def validate_template(self, template_name: str, available_context: Dict[str, str]) -> ValidationResult:
        """Validates if we have sufficient context to populate a template"""
        # Load template requirements
        template_reqs = self._load_template_requirements(template_name)

        # Check available data against requirements
        filled_fields = 0
        missing_fields = {}
        critical_gaps = []
        data_requests = []

        for field, requirements in template_reqs.items():
            if not self._check_field_data(field, requirements, available_context):
                missing_fields[field] = requirements.get('reason', 'Required for completeness')
                if requirements.get('critical', False):
                    critical_gaps.append(field)
                data_requests.append(
                    (field,
                     requirements.get('justification', ''),
                     requirements.get('priority', 3))
                )
            else:
                filled_fields += 1

        completion = (filled_fields / len(template_reqs)) * 100

        return ValidationResult(
            completion_percentage=completion,
            missing_fields=missing_fields,
            critical_gaps=critical_gaps,
            data_requests=sorted(data_requests, key=lambda x: x[2])  # Sort by priority
        )

    def _load_template_requirements(self, template_name: str) -> Dict:
        """Loads metadata about template field requirements"""
        # Convert template name to requirements path
        template_path = Path(template_name)
        requirements_path = template_path.parent / f"{template_path.stem}_requirements.yaml"

        try:
            with open(requirements_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ValueError(f"No requirements file found for template: {template_name}")

    def _check_field_data(self, field: str, requirements: Dict, context: Dict) -> bool:
        """Checks if we have sufficient context for a specific field"""
        required_context = requirements.get('required_context', [])
        return all(ctx in context for ctx in required_context)
