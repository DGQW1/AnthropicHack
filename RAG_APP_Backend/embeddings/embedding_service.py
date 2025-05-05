"""
Embedding service for generating vector representations of text
"""

import numpy as np
from typing import List, Union, Optional
import warnings

from voyageai import Client as VoyageClient

from ..config import VOYAGE_API_KEY, EMBEDDING_MODEL


class EmbeddingService:
    """
    Service for generating embeddings from text using the Voyage AI API
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        """
        Initialize the embedding service
        
        Args:
            api_key: Voyage AI API key. If None, uses the key from config
            model: Model to use for embeddings
        """
        self.api_key = api_key or VOYAGE_API_KEY
        self.model = model
        
        if not self.api_key:
            warnings.warn("No Voyage API key provided. Embeddings will be random.")
            self.client = None
        else:
            try:
                self.client = VoyageClient(api_key=self.api_key)
            except Exception as e:
                warnings.warn(f"Failed to initialize Voyage client: {e}")
                self.client = None
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy.ndarray: Array of embeddings, one for each input text
        """
        if not texts:
            return np.array([])
        
        if not self.client:
            # Return random embeddings for testing/fallback
            warnings.warn("Using random embeddings (Voyage client not available)")
            return np.random.random((len(texts), 768)).astype('float32')
        
        try:
            # Voyage API has two different parameter naming conventions
            # depending on the version, so we handle both
            try:
                resp = self.client.embed(model=self.model, input=texts)
                embeddings = [item["embedding"] for item in resp["data"]]
            except TypeError:
                # Try with 'texts' parameter instead of 'input'
                resp = self.client.embed(model=self.model, texts=texts)
                embeddings = [item["embedding"] for item in resp["data"]]
                
            return np.array(embeddings, dtype='float32')
            
        except Exception as e:
            warnings.warn(f"Error with Voyage API: {e}")
            # Return random embeddings as fallback
            return np.random.random((len(texts), 768)).astype('float32')
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length
        
        Args:
            embeddings: Array of embeddings to normalize
            
        Returns:
            numpy.ndarray: Normalized embeddings
        """
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return embeddings / norms