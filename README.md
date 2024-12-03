# Document RAG API

## Description

A FastAPI-based application designed to ingest, process, and query documents
stored in markdown format. It leverages a Retrieval-Augmented Generation (RAG)
approach to provide relevant document sections based on user queries.

## Usage

### Querying the API

- **Ingest Documents from Files**:

  ```bash
  curl -X POST "http://localhost:8000/ingest/files"
  ```

- **Ingest Documents Directly**:

  ```bash
  curl -X POST "http://localhost:8000/ingest" \
    -H "Content-Type: application/json" \
    -d '{
      "documents": [
        {
          "content": "# Document Title\n\n## Section 1\nConfidence: 5/5 Source: Primary\n\nThis is the content...",
          "filename": "document1.md",
          "source": "api_upload"
        }
      ]
    }'
  ```

- **Query Documents**:
  ```bash
  curl -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "your query here",
      "n_results": 5,
      "min_confidence": 0.8
    }'
  ```

### API Documentation

Access the interactive API documentation at `http://localhost:8000/docs` after
starting the server.

### API Endpoints

- **POST /ingest**

  - Ingest one or more documents directly via API
  - Accepts JSON payload with document content
  - Documents must include content and filename
  - Optional source field for tracking origin

- **POST /ingest/files**

  - Ingest all markdown files from the data directory
  - No request body needed
  - Returns count of ingested documents

- **POST /query**

  - Search through ingested documents
  - Supports filtering by confidence score
  - Returns most relevant sections

- **GET /health**
  - Basic health check endpoint
  - Returns server status

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:

   ```bash
   # Normal start
   python src/main.py

   # Start with initial data ingestion
   python src/main.py --init
   ```

## Development

### Setting Up

1. Ensure you have a `/data` directory in the project root with markdown files
   to ingest.

2. Use `watchdog` to automatically restart the server on code changes:
   ```bash
   python watchdog_script.py
   ```

### Debugging

- Check the console output for debug information about the data directory and
  file loading.
- Ensure the `data` directory contains markdown files and is correctly
  referenced in the code.
- Monitor the server logs for ingestion and query details
- Use the /health endpoint to verify server status

### Contributing

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

### Query Endpoint (/query)

**Success Response:**

```json
{
  "results": [
    "Olvana's cyber capabilities are sophisticated, known for subversion operations. They target sectors including healthcare, financial services, defense industries, energy, and government systems. Known malicious activities include healthcare system targeting, financial sector attacks, and critical infrastructure targeting.",
    "The cyber warfare program is active and known for limited attacks targeting adversarial nations and institutions. Effectiveness has been demonstrated through multiple operations."
  ],
  "metadata": [
    {
      "source": "data/olvana.md",
      "filename": "olvana.md",
      "section_title": "Cyber Capabilities",
      "confidence": "5/5",
      "confidence_score": 1.0,
      "source_document": "Country Overview Document"
    },
    {
      "source": "data/north_torbia.md",
      "filename": "north_torbia.md",
      "section_title": "Cyber Capabilities",
      "confidence": "5/5",
      "confidence_score": 1.0,
      "source_document": "Country Overview Document"
    }
  ]
}
```

**Error Response (Invalid Confidence Score):**

```json
{
  "detail": "min_confidence must be between 0 and 1"
}
```

**Error Response (Empty Vector Store):**

```json
{
  "detail": "No documents found in vector store. Please ingest documents first."
}
```

### Ingest Endpoint (/ingest)

**Success Response:**

```json
{
  "status": "success",
  "message": "Successfully ingested 3 documents"
}
```

**Error Response (Invalid Document Format):**

```json
{
  "detail": [
    {
      "loc": ["body", "documents", 0, "content"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Error Response (Processing Error):**

```json
{
  "detail": "Failed to process document: Invalid markdown structure"
}
```

### Ingest Files Endpoint (/ingest/files)

**Success Response:**

```json
{
  "status": "success",
  "message": "Successfully ingested 11 documents from files"
}
```

**Warning Response (No Files):**

```json
{
  "status": "warning",
  "message": "No documents found in data directory"
}
```

**Error Response (Directory Access):**

```json
{
  "detail": "Failed to access data directory: Permission denied"
}
```

### Health Check Endpoint (/health)

**Success Response:**

```json
{
  "status": "healthy"
}
```
