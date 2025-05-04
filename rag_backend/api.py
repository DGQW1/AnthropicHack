import os
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import fitz

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

@app.get("/summarize/{file_path:path}")
async def summarize_file(file_path: str):
    """
    Summarize a file and extract its key concept using Claude 3.5 Haiku
    """
    try:
        # Convert to absolute path if necessary
        if not os.path.isabs(file_path):
            abs_path = os.path.abspath(file_path)
            if not os.path.exists(abs_path) and file_path.startswith(".."):
                # Try relative to the current directory
                abs_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
        else:
            abs_path = file_path
            
        if not os.path.exists(abs_path):
            return {"error": f"File not found: {file_path}"}
            
        # Use the new extract_file_concept method
        result = rag_system.concept_extractor.extract_file_concept(abs_path)
        return result
        
    except Exception as e:
        return {"error": f"Error summarizing file: {str(e)}"}

@app.get("/summarize_direct/{file_path:path}")
async def summarize_direct(file_path: str):
    """
    Summarize a file directly by reading its contents without using the extraction methods.
    Uses Claude 3.5 Haiku to generate a one-sentence summary and extract a key concept.
    """
    try:
        # Convert to absolute path if necessary
        if not os.path.isabs(file_path):
            abs_path = os.path.abspath(file_path)
            if not os.path.exists(abs_path) and file_path.startswith(".."):
                # Try relative to the current directory
                abs_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
        else:
            abs_path = file_path
            
        if not os.path.exists(abs_path):
            return {"error": f"File not found: {file_path}"}
            
        print(f"Reading file: {abs_path}")
        # Read the file directly
        try:
            if abs_path.lower().endswith('.pdf'):
                # Handle PDF files
                doc = fitz.open(abs_path)
                content = ""
                for page in doc:
                    content += "\n\n" + page.get_text().strip()
            else:
                # Handle text files
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}
        
        # Truncate if text is too long
        max_length = 20000  # Claude's context limit is higher, but we'll be conservative
        if len(content) > max_length:
            print(f"Text too long ({len(content)} chars), truncating to {max_length} chars")
            content = content[:max_length]
        
        # Extract concepts
        if not rag_system.concept_extractor.client:
            # Fallback if Claude API is not available
            concepts = rag_system.concept_extractor.extract_concepts_basic(content)
            
            # Extract basic summary
            paragraphs = content.split('\n\n')
            summary = paragraphs[0][:200] + "..." if paragraphs and len(paragraphs[0]) > 200 else paragraphs[0] if paragraphs else "Summary not available"
            
            return {
                "summary": summary,
                "key_concept": concepts[0] if concepts else ""
            }
        
        # Generate summary using Claude
        try:
            # First call: Get a one-sentence summary
            summary_prompt = f"""
            This is content from a computer science education document.
            
            Content: {content}
            
            Please summarize this document in exactly one sentence. Focus on the main topic being discussed.
            """
            
            summary_response = rag_system.concept_extractor.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=200,
                temperature=0,
                system="You are a computer science education expert who creates concise, accurate summaries.",
                messages=[
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ]
            )
            
            summary = summary_response.content[0].text.strip()
            # Remove quotes if Claude returned the summary in quotes
            summary = summary.strip('"\'')
            
            # Second call: Extract one key concept
            concept_prompt = f"""
            This is content from a computer science education document.
            
            Content: {content}
            
            What is the single most important computer science concept discussed in this document?
            Return ONLY the concept name, with no additional text or explanation.
            """
            
            concept_response = rag_system.concept_extractor.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=50,
                temperature=0,
                system="You are a computer science education expert who identifies the most important technical concepts.",
                messages=[
                    {
                        "role": "user",
                        "content": concept_prompt
                    }
                ]
            )
            
            key_concept = concept_response.content[0].text.strip()
            # Remove quotes if Claude returned the concept in quotes
            key_concept = key_concept.strip('"\'')
            
            return {
                "summary": summary,
                "key_concept": key_concept
            }
        except Exception as e:
            print(f"Error in Claude summarization: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "key_concept": ""
            }
            
    except Exception as e:
        print(f"Error in direct summarization: {e}")
        return {"error": f"Error summarizing file: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 