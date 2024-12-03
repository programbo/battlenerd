# Document RAG API

## Description

A FastAPI-based application designed to ingest, process, and query documents
stored in markdown format. It leverages a Retrieval-Augmented Generation (RAG)
approach to provide relevant document sections based on user queries.

## Usage

### Querying the API

- **Ingest Documents**:

  ```bash
  curl -X POST "http://localhost:8000/ingest"
  ```

- **Query Documents**:
  ```bash
  curl -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "your query here"}'
  ```

### API Documentation

Access the interactive API documentation at `http://localhost:8000/docs` after
starting the server.

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

### Contributing

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.
