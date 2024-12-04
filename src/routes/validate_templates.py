from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
from store import VectorStore
from process import TextProcessor

router = APIRouter()

# Will be set by main app
validator = None
vector_store = None
processor = None

class TemplateValidationRequest(BaseModel):
    template_names: List[str]
    available_context: Dict[str, str]

class DataRequest(BaseModel):
    field: str
    justification: str
    priority: int
    examples: Optional[List[str]] = None

class ValidationResult(BaseModel):
    completion_percentage: float
    can_generate: bool
    missing_critical: List[str]
    data_requests: List[DataRequest]
    warning_message: Optional[str] = None
    info_message: Optional[str] = None
    relevant_sections: List[str] = []  # Sections from vector store that match requirements

class ValidationResponse(BaseModel):
    templates: Dict[str, ValidationResult]

@router.post("/validate-templates")
async def validate_templates(request: TemplateValidationRequest) -> ValidationResponse:
    results = {}

    for template_name in request.template_names:
        # Get template requirements
        result = validator.validate_template(template_name, request.available_context)

        # Query vector store for each required context field
        relevant_sections = []
        for field in result.missing_fields:
            # Search vector store for content relevant to this field
            query = f"information about {field} for {template_name}"
            search_results = vector_store.query(
                query_text=query,
                n_results=3,
                min_confidence=0.7
            )

            if search_results and search_results['documents']:
                # Process and analyze found content
                content = "\n".join(search_results['documents'][0])
                processed_content = processor.clean_text(content)

                # Update completion based on found content
                if processed_content:
                    relevant_sections.extend(search_results['documents'][0])
                    # Remove from missing fields if content is sufficient
                    if field in result.missing_fields:
                        del result.missing_fields[field]
                    if field in result.critical_gaps:
                        result.critical_gaps.remove(field)

        # Recalculate completion percentage based on vector store findings
        total_fields = len(validator._load_template_requirements(template_name))
        filled_fields = total_fields - len(result.missing_fields)
        completion = (filled_fields / total_fields) * 100

        validation_result = ValidationResult(
            completion_percentage=completion,
            can_generate=len(result.critical_gaps) == 0,
            missing_critical=result.critical_gaps,
            data_requests=[
                DataRequest(
                    field=field,
                    justification=justification,
                    priority=priority,
                    examples=get_field_examples(field, template_name)
                )
                for field, justification, priority in result.data_requests
            ],
            relevant_sections=relevant_sections
        )

        if validation_result.completion_percentage < 50:
            validation_result.warning_message = "Insufficient data for meaningful analysis"
        elif validation_result.completion_percentage < 80:
            validation_result.info_message = "Additional data would improve analysis quality"

        results[template_name] = validation_result

    return ValidationResponse(templates=results)

def get_field_examples(field: str, template_name: str) -> List[str]:
    """Returns context-aware examples based on template type and field"""
    # Template-specific examples
    template_examples = {
        "imap_operational": {
            "situation": [
                "NT forces have established defensive positions along coastal areas of Roxas City",
                "Coalition forces are positioned for amphibious operations"
            ],
            "background": [
                "Recent UN resolution 8873 authorized coalition intervention",
                "Previous NT activities in region include..."
            ]
        },
        "opord": {
            "mission": [
                "CJTF 667 will conduct amphibious operations to...",
                "BG XXX will secure key terrain vicinity..."
            ]
        }
        # Add more template-specific examples
    }

    return template_examples.get(template_name, {}).get(field, [])
