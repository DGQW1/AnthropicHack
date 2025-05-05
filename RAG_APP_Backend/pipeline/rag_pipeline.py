"""
Main RAG pipeline module that orchestrates the entire RAG process
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from anthropic import Anthropic
from pathlib import Path

from ..config import (
    ANTHROPIC_API_KEY, 
    VOYAGE_API_KEY, 
    DEFAULT_CONTENT_FOLDERS,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from ..loaders.pdf_loader import PDFLoader
from ..embeddings.embedding_service import EmbeddingService
from ..store.vector_store import VectorStore
from ..retrieval.retriever import Retriever
from .concept_extractor import ConceptExtractor
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates loading, indexing, and querying
    """
    
    def __init__(self):
        """Initialize the RAG pipeline"""
        # Initialize clients
        self.anthropic_client = None
        if ANTHROPIC_API_KEY:
            try:
                self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            except Exception as e:
                logger.error(f"Could not initialize Anthropic client: {e}")
        
        # Initialize components
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.embedding_service, self.vector_store)
        self.concept_extractor = ConceptExtractor(self.anthropic_client)
        
        # Load index if it exists
        self.vector_store.load_index()
    
    def index_folders(self, folders: List[str] = DEFAULT_CONTENT_FOLDERS) -> bool:
        """
        Index content in the specified folders
        
        Args:
            folders: List of folder paths to index
            
        Returns:
            bool: True if indexing was successful
        """
        try:
            # Process all folders to get chunks
            logger.info(f"Indexing folders: {folders}")
            all_chunks = []
            for folder in folders:
                logger.info(f"Processing folder: {folder}")
                folder_chunks = PDFLoader.process_folder(
                    folder, 
                    CHUNK_SIZE, 
                    CHUNK_OVERLAP,
                    self.anthropic_client
                )
                all_chunks.extend(folder_chunks)
                logger.info(f"Extracted {len(folder_chunks)} chunks from {folder}")
            
            # If no chunks were found, return False
            if not all_chunks:
                logger.warning("No chunks found to index")
                return False
            
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = self.embedding_service.embed_texts(texts)
            
            # Build vector index
            logger.info("Building vector index")
            self.vector_store.build_index(all_chunks, embeddings)
            
            logger.info(f"Indexed {len(all_chunks)} chunks successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return False
    
    def process_query(self, query: str, k: int = 5, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query
        
        Args:
            query: Query string
            k: Number of results to return
            category_filter: Optional category to filter results
            
        Returns:
            Dictionary with query results
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Apply category filter if specified
            filter_func = None
            if category_filter:
                filter_func = self.retriever.filter_by_category(category_filter)
                logger.info(f"Applied category filter: {category_filter}")
            
            # Retrieve relevant documents
            results = self.retriever.retrieve(query, k=k, filter_func=filter_func)
            
            # Generate LLM response if Anthropic client is available
            response = None
            sources = []
            if self.anthropic_client:
                # Format context for LLM
                context = "Context information:\n\n"
                for i, result in enumerate(results):
                    context += f"[Document {i+1}]\n{result['text']}\n\n"
                    # Track sources
                    metadata = result.get("metadata", {})
                    sources.append({
                        "source": metadata.get("source", "Unknown"),
                        "category": metadata.get("category", "Unknown"),
                        "score": result.get("score", 0.0)
                    })
                
                # Generate response
                prompt = f"""
                {context}
                
                Given the context information above, please answer the following question:
                {query}
                
                Your answer should:
                1. Be comprehensive but concise
                2. Use only information from the context
                3. Say "I don't know" if the answer isn't in the context
                """
                
                try:
                    llm_response = self.anthropic_client.messages.create(
                        model="claude-3-5-haiku-20241022",
                        max_tokens=500,
                        temperature=0,
                        system="You are a helpful assistant for computer science education. You answer questions based on the context information provided.",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    response = llm_response.content[0].text.strip()
                except Exception as e:
                    logger.error(f"Error generating LLM response: {e}")
                    response = f"Error generating response: {str(e)}"
            
            # Return query results
            return {
                "query": query,
                "results": results,
                "response": response,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "response": None,
                "sources": []
            }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file to extract concepts and generate summary
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary with file metadata
        """
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Extract text from file
            if file_path.lower().endswith('.pdf'):
                file_text = PDFLoader.extract_text_with_limit(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_text = f.read()
            
            # Extract concepts and summary
            result = self.concept_extractor.extract_file_concept(file_text=file_text)
            
            # Add file information
            result["path"] = file_path
            result["filename"] = os.path.basename(file_path)
            result["category"] = self._determine_category(file_path)
            
            logger.info(f"Generated metadata for: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return {
                "path": file_path,
                "filename": os.path.basename(file_path),
                "summary": f"Error: {str(e)}",
                "key_concept": "",
                "category": self._determine_category(file_path) 
            }
    
    def get_all_concepts(self) -> List[str]:
        """
        Get all concepts in the index
        
        Returns:
            List of concept strings
        """
        return self.retriever.list_concepts()
    
    def get_concept_content(self, concept: str) -> Dict[str, List]:
        """
        Get content related to a specific concept
        
        Args:
            concept: Concept to get content for
            
        Returns:
            Dictionary with content and questions lists
        """
        return self.retriever.retrieve_by_concept(concept)
    
    def _determine_category(self, file_path: str) -> str:
        """Helper method to determine file category"""
        if "270slides" in file_path:
            return "slide"
        elif "270handout" in file_path:
            return "handout"
        elif "270questions" in file_path:
            return "question"
        else:
            return "other"