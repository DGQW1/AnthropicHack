import os
from backend import RAGSystem
from pathlib import Path
import glob
import json

# Use environment variables for API keys
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def main():
    # Initialize the RAG system
    rag_system = RAGSystem(VOYAGE_API_KEY, ANTHROPIC_API_KEY)
    
    # Define the test folder
    base_dir = os.path.dirname(os.path.abspath(os.getcwd()))
    test_folder = os.path.join(base_dir, "270handout")
    
    # Verify folder exists
    if not os.path.exists(test_folder):
        print(f"ERROR: Folder does not exist: {test_folder}")
        # Try with a relative path
        test_folder = "../270handout"
        if not os.path.exists(test_folder):
            print(f"ERROR: Folder also does not exist at relative path: {test_folder}")
            test_folder = os.path.abspath("../270handout")
            print(f"Trying absolute path: {test_folder}")
            if not os.path.exists(test_folder):
                print(f"ERROR: Folder also does not exist at absolute path")
                return
    
    print(f"SUCCESS: Folder exists at {test_folder}")
    
    # List files in the folder
    files = os.listdir(test_folder)
    print(f"Found {len(files)} files in the folder")
    print(f"Sample files: {files[:5]}")
    
    # Process the folder
    chunks = rag_system.process_folder(test_folder)
    print(f"Processed {len(chunks)} chunks from the folder")
    
    # Build index
    faiss_index, indexed_chunks = rag_system.build_index(chunks)
    print(f"Built index with {len(indexed_chunks)} chunks")
    
    # Check extracted concepts
    concepts = rag_system.get_all_concepts()
    print(f"Extracted {len(concepts)} concepts")
    if concepts:
        print(f"Sample concepts: {concepts[:10]}")
    
    # Check data files created
    storage_dir = Path("./data/storage")
    if os.path.exists(storage_dir / "all_chunks.json"):
        print("SUCCESS: all_chunks.json was created")
    else:
        print("ERROR: all_chunks.json was not created")
    
    if os.path.exists(storage_dir / "all_concepts.json"):
        print("SUCCESS: all_concepts.json was created")
    else:
        print("ERROR: all_concepts.json was not created")
        
    if os.path.exists(storage_dir / "concept_to_chunks.json"):
        print("SUCCESS: concept_to_chunks.json was created")
    else:
        print("ERROR: concept_to_chunks.json was not created")

if __name__ == "__main__":
    main() 