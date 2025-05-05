import os
import sys
import json
import time
from pathlib import Path
from anthropic import Anthropic

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_backend.backend import ConceptExtractor

# Output file to save results
RESULTS_FILE = "extracted_concepts.json"

# API keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    print("Warning: ANTHROPIC_API_KEY environment variable not set")
    print("The extraction will be less accurate without Claude API access")

# Initialize Anthropic client
anthropic_client = None
if ANTHROPIC_API_KEY:
    try:
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        print(f"Could not initialize Anthropic client: {e}")

# Initialize concept extractor
concept_extractor = ConceptExtractor(anthropic_client)

# Folder paths
FOLDERS = [
    "270handout",
    "270questions",
    "270slides"
]

def get_file_paths(folder_path):
    """Get all PDF file paths in a folder"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return []
    
    # Get only PDF files (skip PPTX as they're causing issues)
    return [str(f) for f in folder.glob("*.pdf")]

def extract_concepts(file_path):
    """Extract concepts from a file"""
    print(f"Processing: {file_path}")
    try:
        # Use our new extract_file_concept method
        result = concept_extractor.extract_file_concept(file_path)
        
        # Add filename to the result
        #result["file"] = os.path.basename(file_path)
        result["path"] = file_path
        # Add category
        # from the folder name, if folder = 270slides, category = slides
        # if folder = 270handout, category = handout
        # if folder = 270questions, category = questions
        category = ""
        if "270slides" in file_path:
            category = "slide"
        elif "270handout" in file_path:
            category = "handout"
        elif "270questions" in file_path:
            category = "question"
        result["category"] = category
        
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # category
        category = ""
        if "270slides" in file_path:
            category = "slide"
        elif "270handout" in file_path:
            category = "handout"
        elif "270questions" in file_path:
            category = "question"
        # Return a result with an error message

        return {
            "path": file_path,
            "summary": f"Error: {str(e)}",
            "key_concept": "",
            "category": category
        }

def main():
    all_results = {}
    
    # Process each folder
    for folder in FOLDERS:
        print(f"\nProcessing folder: {folder}")
        file_paths = get_file_paths(folder)
        print(f"Found {len(file_paths)} files")
        
        folder_results = []
        for file_path in file_paths:
            # Extract concepts and add to results
            result = extract_concepts(file_path)
            folder_results.append(result)
            
            # Save intermediate results after each file (in case of crashes)
            all_results[folder] = folder_results
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            
            # Sleep briefly to avoid rate limits
            time.sleep(1)
        
        # Save this folder's results
        all_results[folder] = folder_results
    
    # Final save
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Print summary
    total_files = sum(len(results) for results in all_results.values())
    print(f"Processed {total_files} files across {len(FOLDERS)} folders")

if __name__ == "__main__":
    main() 