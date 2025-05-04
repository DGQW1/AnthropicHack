# Concept-Based RAG System

This backend system processes educational materials from various folders, extracts key concepts, and provides an API for retrieving content by concept.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export VOYAGE_API_KEY="your_voyage_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

## Usage

### Running the API

Start the FastAPI server:
```bash
cd rag_backend
python api.py
```

The API will be available at http://localhost:8000

### API Endpoints

- `GET /concepts` - Retrieve all extracted concepts
- `GET /concept/{concept}` - Get content related to a specific concept
- `POST /process` - Process folders to build the index (request body: `{"folders": ["folder1", "folder2", ...]}`)

### Direct Usage

You can also use the RAG system directly in Python:

```python
from backend import RAGSystem

rag_system = RAGSystem(voyage_api_key, anthropic_api_key)

# Process folders
rag_system.process_all_folders(["270handout", "270questions", "270slides"])

# Get all concepts
concepts = rag_system.get_all_concepts()

# Get content for a specific concept
concept_content = rag_system.get_content_for_concept("example_concept")
```

## Frontend Integration

This backend is designed to be integrated with a frontend that displays:
1. A list of all extracted concepts
2. Content related to selected concepts
3. Questions related to selected concepts

The API provides all necessary endpoints for this interaction.

## Note on API Keys

This system requires:
- A Voyage AI API key for embeddings
- An Anthropic API key for concept extraction

Please obtain these keys and set them as environment variables before running. 