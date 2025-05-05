"""
Retrieval module for finding relevant content based on queries
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from ..embeddings.embedding_service import EmbeddingService
from ..store.vector_store import VectorStore


class Retriever:
    """
    Handles retrieval of relevant documents based on queries
    """
    
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        """
        Initialize retriever
        
        Args:
            embedding_service: Service for creating embeddings
            vector_store: Store for vectors and metadata
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = 5, 
                 filter_func: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            k: Number of results to return
            filter_func: Optional function to filter results
            
        Returns:
            List of documents with text, metadata, and score
        """
        # Get query embedding
        query_embedding = self.embedding_service.embed_texts([query])[0]
        
        # Search vector store
        results = self.vector_store.search(np.array([query_embedding]), k=k*2)  # Fetch extra to allow for filtering
        
        # Apply filter if provided
        if filter_func is not None:
            results = [r for r in results if filter_func(r)]
            results = results[:k]  # Limit to k after filtering
        else:
            results = results[:k]
        
        return results
    
    def retrieve_by_concept(self, concept: str) -> Dict[str, List]:
        """
        Retrieve content related to a specific concept
        
        Args:
            concept: Concept to retrieve content for
            
        Returns:
            Dictionary with 'content' and 'questions' lists
        """
        return self.vector_store.get_content_for_concept(concept)
    
    def list_concepts(self) -> List[str]:
        """
        List all available concepts
        
        Returns:
            List of concept strings
        """
        return self.vector_store.get_all_concepts()
    
    def filter_by_category(self, category: str) -> callable:
        """
        Create a filter function for a specific category
        
        Args:
            category: Category to filter for (e.g. 'slide', 'handout', 'question')
            
        Returns:
            Function that filters results by category
        """
        def filter_func(result: Dict[str, Any]) -> bool:
            metadata = result.get("metadata", {})
            return metadata.get("category") == category
        
        return filter_func
        
    def filter_by_source(self, source_pattern: str) -> callable:
        """
        Create a filter function for a specific source pattern
        
        Args:
            source_pattern: String pattern to match in source field
            
        Returns:
            Function that filters results by source
        """
        def filter_func(result: Dict[str, Any]) -> bool:
            metadata = result.get("metadata", {})
            source = metadata.get("source", "")
            return source_pattern.lower() in source.lower()
        
        return filter_func