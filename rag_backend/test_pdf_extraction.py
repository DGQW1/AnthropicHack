from backend import RAGSystem
import os
import json

# Initialize the RAG system (with dummy keys since we're just testing extraction)
rag_system = RAGSystem("dummy_voyage_key")

# Define path to a sample PDF file
pdf_path = "../270handout/270handout1.pdf"  # This is a sample path, adjust as needed

def test_pdf_extraction(pdf_path):
    print(f"\n=== Testing PDF extraction on {pdf_path} ===\n")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return
    
    print(f"File exists: {pdf_path}")
    
    # Extract PDF chunks
    try:
        chunks = rag_system.extract_pdf_chunks(pdf_path)
        print(f"Successfully extracted {len(chunks)} chunks from the PDF")
        
        # Print a summary of each chunk
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            text_preview = chunk["text"][:150] + "..." if len(chunk["text"]) > 150 else chunk["text"]
            print(f"Text preview: {text_preview}")
            print(f"Metadata: {chunk['metadata']}")
            
            # Extract concepts from each chunk
            concepts = rag_system.concept_extractor.extract_concepts(chunk["text"])
            print(f"Extracted {len(concepts)} concepts:")
            for j, concept in enumerate(concepts):
                print(f"  {j+1}. {concept}")
    
    except Exception as e:
        print(f"Error processing PDF file: {e}")
        import traceback
        print(traceback.format_exc())

# Try to find an available PDF file from the possible folders
def find_sample_pdf():
    possible_paths = [
        "../270handout/270handout1.pdf",
        "../270handout/270handout7 (1).pdf",
        "/Users/victor/Documents/USC/AnthropicHack/270handout/270handout1.pdf"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If no files found, search in parent directory for PDF files
    parent_dir = os.path.abspath("..")
    for folder in ["270handout", "270questions", "270slides"]:
        folder_path = os.path.join(parent_dir, folder)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(".pdf"):
                    return os.path.join(folder_path, file)
    
    return None

# Execute test
pdf_path = find_sample_pdf()
if pdf_path:
    test_pdf_extraction(pdf_path)
else:
    print("No sample PDF files found. Please adjust the paths in the script.") 