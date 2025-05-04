# CS Course Material Concept Extraction

This project uses Claude API to extract key concepts and generate summaries from computer science course materials.

## Overview

The system analyzes PDF files from course materials and extracts:
1. A one-sentence summary of each document
2. The single most important computer science concept for each document

## Components

- **RAG Backend**: The core system for document processing and concept extraction
  - `backend.py`: Core library for extraction and RAG functionality
  - `api.py`: FastAPI server exposing endpoints for the system
  - `view_concepts.html`: Simple web interface to browse extracted concepts

- **Analysis Tools**:
  - `extract_all_concepts.py`: Process all files and extract concepts using Claude
  - `convert_to_csv.py`: Convert JSON results to CSV for easier analysis
  - `analyze_concepts.py`: Generate statistics and visualizations of concept frequencies
  - `view_extracted_concepts.html`: Interactive web viewer for concepts and summaries

## Key Findings

The analysis identified 22 unique concepts across 39 documents, with the most common being:
- Dynamic Programming (7 occurrences)
- NP-Completeness (7 occurrences)
- Network Flow (3 occurrences)
- Greedy Algorithms (3 occurrences)

## Directory Structure

- `270handout/`: Course lecture handouts (PDFs)
- `270questions/`: Homework and problem sets (PDFs)
- `270slides/`: Lecture slides (PPT/PPTX)
- `rag_backend/`: Backend code for the RAG system
- `data/`: Storage for processed data and indices

## Usage

### Running the Concept Extraction

```bash
python extract_all_concepts.py
```

This will process all files in the configured folders and save results to `extracted_concepts.json`.

### Viewing the Results

Open `view_extracted_concepts.html` in a web browser to interactively explore the extracted concepts.

### Converting and Analyzing Results

```bash
python convert_to_csv.py  # Convert to CSV format
python analyze_concepts.py  # Generate statistics and visualizations
```

## Implementation Details

The concept extraction uses Claude 3.5 Haiku to:
1. Read the entire document 
2. Generate a one-sentence summary focused on the document's main topic
3. Extract the single most important computer science concept

If the Claude API is unavailable, the system falls back to a basic frequency-based approach.

## Requirements

- Python 3.8+
- Anthropic API key (for Claude)
- Required packages: `pymupdf`, `anthropic`, `fastapi`, `uvicorn`, `matplotlib`
