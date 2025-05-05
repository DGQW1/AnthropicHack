"""
PDF loading and processing module for the RAG application
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import re
from anthropic import Anthropic

# Import using relative paths
from ..config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_TEXT_LENGTH, ANTHROPIC_API_KEY
from ..summary_extractor.summary_extractor import SummaryExtractor


class PDFLoader:
    """
    Handles loading PDFs, extracting text, and creating chunks with metadata
    """
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: The extracted text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {file_path}")
            
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                full_text += "\n\n" + page.get_text().strip()
            return full_text
        except Exception as e:
            raise IOError(f"Error extracting text from PDF: {str(e)}")
    
    @staticmethod
    def extract_text_with_limit(file_path: str, max_length: int = MAX_TEXT_LENGTH) -> str:
        """
        Extract text from a PDF file with a length limit
        
        Args:
            file_path: Path to the PDF file
            max_length: Maximum number of characters to extract
            
        Returns:
            str: The extracted text content, truncated if necessary
        """
        text = PDFLoader.extract_text(file_path)
        if len(text) > max_length:
            print(f"Text too long ({len(text)} chars), truncating to {max_length} chars")
            return text[:max_length]
        return text
    
    @staticmethod
    def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, 
                      metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            metadata: Optional metadata to associate with each chunk
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        chunks = []
        start = 0
        
        if metadata is None:
            metadata = {}
            
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Create a copy of metadata and add position
            chunk_metadata = {**metadata, "start_pos": start}
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
            start += chunk_size - chunk_overlap
            
        return chunks
    
    @staticmethod
    def process_file(file_path: str, chunk_size: int = CHUNK_SIZE, 
                     chunk_overlap: int = CHUNK_OVERLAP, 
                     anthropic_client: Optional[Anthropic] = None) -> List[Dict[str, Any]]:
        """
        Process a PDF file to extract text and create chunks with metadata
        
        Args:
            file_path: Path to the PDF file
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            anthropic_client: Optional Anthropic client for summary generation
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        # Extract text from PDF
        try:
            text = PDFLoader.extract_text_with_limit(file_path)
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return []
        
        # Create basic metadata
        metadata = {
            "source": Path(file_path).name,
            "source_type": "pdf",
            "path": file_path,
            "category": PDFLoader.determine_category(file_path)
        }
        
        # Extract summary and key concept if text is available
        if text:
            # Initialize SummaryExtractor
            if anthropic_client is None and ANTHROPIC_API_KEY:
                try:
                    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
                except Exception as e:
                    print(f"Could not initialize Anthropic client: {e}")
            
            summary_extractor = SummaryExtractor(anthropic_client)
            
            # Extract summary and key concept
            summary_result = summary_extractor.extract_summary_and_concept(text)
            
            # Add to metadata
            metadata["summary"] = summary_result.get("summary", "")
            metadata["key_concept"] = summary_result.get("key_concept", "")
            
            print(f"Generated summary for {file_path}: {metadata['summary'][:50]}...")
            print(f"Extracted key concept: {metadata['key_concept']}")
        
        # Create chunks with metadata
        chunks = PDFLoader.create_chunks(text, chunk_size, chunk_overlap, metadata)
        print(f"Extracted {len(chunks)} chunks from {file_path}")
        
        return chunks
    
    @staticmethod
    def determine_category(file_path: str) -> str:
        """
        Determine the category of a file based on its path
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: The category (slide, handout, question, or other)
        """
        if "270slides" in file_path:
            return "slide"
        elif "270handout" in file_path:
            return "handout"
        elif "270questions" in file_path:
            return "question"
        else:
            return "other"
    
    @staticmethod
    def process_folder(folder_path: str, chunk_size: int = CHUNK_SIZE,
                       chunk_overlap: int = CHUNK_OVERLAP,
                       anthropic_client: Optional[Anthropic] = None) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a folder
        
        Args:
            folder_path: Path to the folder containing PDF files
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            anthropic_client: Optional Anthropic client for summary generation
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        folder_path = os.path.abspath(folder_path)
        print(f"Processing folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            return []
        
        # Initialize Anthropic client if not provided
        if anthropic_client is None and ANTHROPIC_API_KEY:
            try:
                anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            except Exception as e:
                print(f"Could not initialize Anthropic client: {e}")
        
        # Find all PDF files
        all_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    all_files.append(os.path.join(root, file))
        
        print(f"Found {len(all_files)} PDF files in {folder_path}")
        
        # Process each PDF file
        all_chunks = []
        for file_path in all_files:
            print(f"Processing file: {file_path}")
            file_chunks = PDFLoader.process_file(
                file_path, 
                chunk_size, 
                chunk_overlap,
                anthropic_client
            )
            all_chunks.extend(file_chunks)
        
        print(f"Total chunks extracted from {folder_path}: {len(all_chunks)}")
        return all_chunks