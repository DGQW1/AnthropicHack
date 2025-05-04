import os
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json

from backend import RAGSystem

app = FastAPI(title="Concept RAG API")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Initialize RAG system
rag_system = RAGSystem(VOYAGE_API_KEY, ANTHROPIC_API_KEY)

# Models
class ConceptContent(BaseModel):
    content: List[Dict[str, Any]]
    questions: List[Dict[str, Any]]

class ProcessFoldersRequest(BaseModel):
    folders: List[str]

class ProcessFoldersResponse(BaseModel):
    success: bool
    message: str
    chunks_processed: int
    concepts_extracted: int

# API Routes
@app.get("/")
async def root():
    return {"message": "Concept RAG API is running"}

@app.get("/concepts", response_model=List[str])
async def get_all_concepts():
    """Get all extracted concepts for frontend display"""
    try:
        return rag_system.get_all_concepts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting concepts: {str(e)}")

@app.get("/concept/{concept}", response_model=ConceptContent)
async def get_concept_content(concept: str):
    """Get content related to a specific concept"""
    try:
        return rag_system.get_content_for_concept(concept)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting content for concept: {str(e)}")

@app.post("/process", response_model=ProcessFoldersResponse)
async def process_folders(request: ProcessFoldersRequest):
    """Process folders to build the index (admin operation)"""
    try:
        # Get the base directory (parent of the current working directory)
        base_dir = os.path.dirname(os.path.abspath(os.getcwd()))
        
        # Convert relative paths to absolute paths
        full_paths = []
        for folder in request.folders:
            # Handle paths that start with ../ or ./
            if folder.startswith("../") or folder.startswith("./"):
                # Resolve from current directory
                full_path = os.path.abspath(os.path.join(os.getcwd(), folder))
            elif os.path.isabs(folder):
                # Already absolute path
                full_path = folder
            else:
                # Treat as relative to parent directory
                full_path = os.path.join(base_dir, folder)
            
            # Verify folder exists
            if not os.path.exists(full_path):
                print(f"Warning: Folder does not exist: {full_path}")
            else:
                print(f"Folder exists: {full_path}")
                
            full_paths.append(full_path)
        
        # Process folders
        _, chunks = rag_system.process_all_folders(full_paths)
        
        # Get number of concepts
        concepts = rag_system.get_all_concepts()
        
        return {
            "success": True,
            "message": "Folders processed successfully",
            "chunks_processed": len(chunks),
            "concepts_extracted": len(concepts)
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing folders: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 