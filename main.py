#!/usr/bin/env python3
# main.py - Generate metadata from PDFs and convert to LlamaIndex documents

import os
import json
import argparse
from pathlib import Path
from llama_index.core import Document
from typing import List, Dict, Any

# Import from our backend
from rag_backend.backend import MetadataGenerator, ConceptExtractor, RAGSystem
from anthropic import Anthropic

def get_api_keys() -> tuple:
    """Get API keys from environment variables"""
    voyage_api_key = os.environ.get("VOYAGE_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not voyage_api_key:
        print("Warning: VOYAGE_API_KEY environment variable not set")
    
    if not anthropic_api_key:
        print("Warning: ANTHROPIC_API_KEY environment variable not set")
    
    return voyage_api_key, anthropic_api_key

def find_pdf_files(folders: List[str]) -> List[str]:
    """Find all PDF files in the specified folders"""
    pdf_files = []
    
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Folder {folder} does not exist")
            continue
            
        # Walk through the directory and find all PDF files
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    
    return pdf_files

def generate_metadata(pdf_files: List[str], anthropic_client) -> Dict[str, Any]:
    """Generate metadata for each PDF file"""
    metadata_generator = MetadataGenerator(anthropic_client)
    
    all_metadata = {}
    for pdf_file in pdf_files:
        print(f"Generating metadata for {pdf_file}")
        try:
            # First extract text from PDF
            pdf_text = PDFtoText.extract_text_with_limit(pdf_file)
            
            # Generate metadata using the extracted text
            metadata = metadata_generator.generate_metadata(pdf_file)
            all_metadata[pdf_file] = metadata
            print(f"  Summary: {metadata.get('summary', 'N/A')}")
            print(f"  Key concept: {metadata.get('key_concept', 'N/A')}")
            print(f"  Category: {metadata.get('category', 'N/A')}")
        except Exception as e:
            print(f"  Error generating metadata: {e}")
            all_metadata[pdf_file] = {
                "path": pdf_file,
                "filename": os.path.basename(pdf_file),
                "summary": f"Error: {str(e)}",
                "key_concept": "",
                "category": "unknown"
            }
    
    return all_metadata

def convert_to_llamaindex_documents(metadata_dict: Dict[str, Any]) -> List[Document]:
    """Convert the PDFs with their metadata to LlamaIndex Document objects"""
    documents = []
    
    for file_path, metadata in metadata_dict.items():
        try:
            # Read the PDF content
            from rag_backend.backend import ConceptExtractor
            extractor = ConceptExtractor(None)  # No need for Anthropic client here
            
            # The ConceptExtractor can read a file and get its full text
            try:
                if file_path.lower().endswith('.pdf'):
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    full_text = ""
                    for page in doc:
                        full_text += "\n\n" + page.get_text().strip()
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
            
            # Convert metadata to LlamaIndex metadata format (all keys as strings)
            doc_metadata = {
                "filename": metadata.get("filename", ""),
                "path": metadata.get("path", ""),
                "category": metadata.get("category", ""),
                "summary": metadata.get("summary", ""),
                "key_concept": metadata.get("key_concept", ""),
                "topics": json.dumps(metadata.get("topics", [])),  # Convert list to JSON string
                "difficulty": metadata.get("difficulty", "medium"),
                "prerequisites": json.dumps(metadata.get("prerequisites", [])),  # Convert list to JSON string
            }
            
            # Create a LlamaIndex Document
            document = Document(
                text=full_text,
                metadata=doc_metadata,
                excluded_embed_metadata_keys=["path"],  # Don't include the path in the embedding
                excluded_llm_metadata_keys=["path"],    # Don't include the path in LLM context
            )
            
            documents.append(document)
            print(f"Created LlamaIndex document for {file_path}")
            
        except Exception as e:
            print(f"Error creating document for {file_path}: {e}")
    
    return documents

def save_metadata_to_json(metadata_dict: Dict[str, Any], output_file: str):
    """Save the metadata to a JSON file"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create a simplified version for storage that isn't nested with full paths
        simplified_metadata = {}
        for file_path, metadata in metadata_dict.items():
            filename = metadata.get("filename", os.path.basename(file_path))
            simplified_metadata[filename] = metadata
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_metadata, f, indent=2)
            
        print(f"Metadata saved to {output_file}")
    except Exception as e:
        print(f"Error saving metadata to {output_file}: {e}")

def main():
    """Main function that orchestrates the metadata extraction and document creation"""
    parser = argparse.ArgumentParser(description="Generate metadata for PDFs and convert to LlamaIndex documents")
    parser.add_argument("--folders", nargs="+", default=["270handout", "270questions", "270slides"],
                      help="List of folders to process")
    parser.add_argument("--output", type=str, default="metadata_output.json",
                      help="Output JSON file to save metadata")
    parser.add_argument("--index_output", type=str, default="./data/document_index",
                      help="Output directory for the LlamaIndex VectorStore")
    
    args = parser.parse_args()
    
    # Get API keys
    voyage_api_key, anthropic_api_key = get_api_keys()
    
    # Initialize Anthropic client if API key is available
    anthropic_client = None
    if anthropic_api_key:
        try:
            anthropic_client = Anthropic(api_key=anthropic_api_key)
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")
    
    # Find all PDF files
    pdf_files = find_pdf_files(args.folders)
    print(f"Found {len(pdf_files)} PDF files in the specified folders")
    
    # Generate metadata
    metadata_dict = generate_metadata(pdf_files, anthropic_client)
    
    # Save metadata to JSON
    save_metadata_to_json(metadata_dict, args.output)
    
    # Convert to LlamaIndex documents
    documents = convert_to_llamaindex_documents(metadata_dict)
    print(f"Created {len(documents)} LlamaIndex documents")
    
    # Optionally, create a VectorStore and save it
    if len(documents) > 0:
        try:
            from llama_index.core import VectorStoreIndex, StorageContext
            from llama_index.vector_stores.faiss import FaissVectorStore
            import numpy as np
            
            # Check if voyage_api_key is available
            if voyage_api_key:
                # Initialize RAG system to use its embed_texts method
                rag_system = RAGSystem(voyage_api_key, anthropic_api_key)
                
                # Extract texts from documents
                texts = [doc.text for doc in documents]
                
                # Get embeddings
                print(f"Getting embeddings for {len(texts)} documents...")
                embeddings = rag_system.embed_texts(texts)
                
                # Create Faiss vector store
                vector_store = FaissVectorStore(dim=len(embeddings[0]))
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Create VectorStoreIndex
                index = VectorStoreIndex(documents, storage_context=storage_context)
                
                # Save the index
                index.storage_context.persist(persist_dir=args.index_output)
                print(f"Saved document index to {args.index_output}")
            else:
                print("Skipping vector index creation because VOYAGE_API_KEY is not available")
        except Exception as e:
            print(f"Error creating vector index: {e}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()