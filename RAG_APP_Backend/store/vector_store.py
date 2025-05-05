"""
Vector store for storing and retrieving embeddings and metadata
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict

from ..config import FAISS_DIR, STORAGE_DIR


class VectorStore:
    """
    Vector store for storing and retrieving document embeddings and metadata
    """
    
    def __init__(self, faiss_dir: Path = FAISS_DIR, storage_dir: Path = STORAGE_DIR):
        """
        Initialize the vector store
        
        Args:
            faiss_dir: Directory for storing FAISS index
            storage_dir: Directory for storing associated metadata
        """
        self.faiss_dir = Path(faiss_dir)
        self.storage_dir = Path(storage_dir)
        
        # Ensure directories exist
        self.faiss_dir.mkdir(exist_ok=True, parents=True)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # Index and data structures
        self.faiss_index = None
        self.chunks = []
        self.concept_to_chunks = defaultdict(list)
        self.concept_to_questions = defaultdict(list)
        self.all_concepts = set()
    
    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Build a FAISS index from chunks and embeddings
        
        Args:
            chunks: List of text chunks with metadata
            embeddings: Numpy array of embeddings corresponding to chunks
            
        Returns:
            Tuple of (faiss_index, chunks)
        """
        # Handle empty chunks case
        if not chunks or len(embeddings) == 0:
            print("Warning: No chunks to index")
            # Create empty index files to avoid future errors
            faiss_index = faiss.IndexFlatIP(768)  # Default dimension
            faiss.write_index(faiss_index, str(self.faiss_dir / "content.index"))
            
            # Save empty mappings
            self._save_mappings({}, {}, [], [])
            
            return faiss_index, chunks
        
        # Initialize FAISS
        d = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(d)  # inner-product index
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to the index
        faiss_index.add(embeddings)
        
        # Store index and chunks
        self.faiss_index = faiss_index
        self.chunks = chunks
        
        # Persist index to disk
        faiss.write_index(faiss_index, str(self.faiss_dir / "content.index"))
        print("FAISS index saved.")
        
        # Extract concepts from each chunk and build concept mapping
        self._build_concept_mappings(chunks)
        
        # Save mappings
        self._save_mappings(
            self.concept_to_chunks,
            self.concept_to_questions,
            list(self.all_concepts),
            chunks
        )
        
        return faiss_index, chunks
    
    def _build_concept_mappings(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build mappings between concepts and chunks
        
        Args:
            chunks: List of text chunks with metadata
        """
        from ..loaders.pdf_loader import PDFLoader
        
        # Reset mappings
        self.concept_to_chunks = defaultdict(list)
        self.concept_to_questions = defaultdict(list)
        self.all_concepts = set()
        
        # Extract concepts from each chunk
        for i, chunk in enumerate(chunks):
            # Get metadata
            metadata = chunk.get("metadata", {})
            
            # Basic concept extraction
            from ..pipeline.concept_extractor import ConceptExtractor
            extractor = ConceptExtractor(None)  # No need for Anthropic client here
            concepts = extractor.extract_concepts_basic(chunk["text"])
            
            # Add to concept mappings
            for concept in concepts:
                self.concept_to_chunks[concept].append(i)  # Store chunk index
                self.all_concepts.add(concept)
                
                # Add to questions mapping if from questions folder
                if metadata.get("category") == "question":
                    self.concept_to_questions[concept].append({
                        "text": chunk["text"],
                        "source": metadata.get("source", "")
                    })
    
    def _save_mappings(self, concept_to_chunks: Dict, concept_to_questions: Dict, 
                       all_concepts: List[str], chunks: List[Dict[str, Any]]) -> None:
        """
        Save mappings to disk
        
        Args:
            concept_to_chunks: Dictionary mapping concepts to chunk indices
            concept_to_questions: Dictionary mapping concepts to question chunks
            all_concepts: List of all concepts
            chunks: List of text chunks with metadata
        """
        # Save concept mappings
        with open(self.storage_dir / "concept_to_chunks.json", "w") as f:
            # Convert defaultdict to regular dict for serialization
            json.dump({k: v for k, v in concept_to_chunks.items()}, f)
        
        with open(self.storage_dir / "concept_to_questions.json", "w") as f:
            json.dump({k: v for k, v in concept_to_questions.items()}, f)
        
        with open(self.storage_dir / "all_concepts.json", "w") as f:
            json.dump(all_concepts, f)
        
        # Save all chunks for retrieval
        with open(self.storage_dir / "all_chunks.json", "w") as f:
            json.dump(chunks, f)
    
    def load_index(self) -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
        """
        Load FAISS index and chunks from disk
        
        Returns:
            Tuple of (faiss_index, chunks)
        """
        faiss_index_path = self.faiss_dir / "content.index"
        chunks_path = self.storage_dir / "all_chunks.json"
        
        # Check if files exist
        if not faiss_index_path.exists() or not chunks_path.exists():
            print("Warning: Index files do not exist")
            return None, []
        
        try:
            # Load FAISS index
            faiss_index = faiss.read_index(str(faiss_index_path))
            
            # Load chunks
            with open(chunks_path, "r") as f:
                chunks = json.load(f)
            
            # Load concept mappings
            self._load_mappings()
            
            # Store in instance
            self.faiss_index = faiss_index
            self.chunks = chunks
            
            print(f"Loaded index with {faiss_index.ntotal} vectors and {len(chunks)} chunks")
            return faiss_index, chunks
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return None, []
    
    def _load_mappings(self) -> None:
        """
        Load concept mappings from disk
        """
        # Load concept to chunks mapping
        if (self.storage_dir / "concept_to_chunks.json").exists():
            with open(self.storage_dir / "concept_to_chunks.json", "r") as f:
                self.concept_to_chunks = defaultdict(list, json.load(f))
        
        # Load concept to questions mapping
        if (self.storage_dir / "concept_to_questions.json").exists():
            with open(self.storage_dir / "concept_to_questions.json", "r") as f:
                self.concept_to_questions = defaultdict(list, json.load(f))
        
        # Load all concepts
        if (self.storage_dir / "all_concepts.json").exists():
            with open(self.storage_dir / "all_concepts.json", "r") as f:
                self.all_concepts = set(json.load(f))
    
    def get_all_concepts(self) -> List[str]:
        """
        Get all concepts in the index
        
        Returns:
            List of concept strings
        """
        # If we haven't loaded yet, load mappings
        if not self.all_concepts and (self.storage_dir / "all_concepts.json").exists():
            self._load_mappings()
            
        return list(self.all_concepts)
    
    def get_content_for_concept(self, concept: str) -> Dict[str, List]:
        """
        Get all content related to a specific concept
        
        Args:
            concept: Concept to get content for
            
        Returns:
            Dictionary with 'content' and 'questions' lists
        """
        # Create empty results to return if files don't exist
        empty_result = {
            "content": [],
            "questions": []
        }
        
        # Load mappings if needed
        if not self.concept_to_chunks:
            self._load_mappings()
        
        # Load chunks if needed
        if not self.chunks and (self.storage_dir / "all_chunks.json").exists():
            with open(self.storage_dir / "all_chunks.json", "r") as f:
                self.chunks = json.load(f)
        
        # Return empty if no chunks loaded
        if not self.chunks:
            return empty_result
        
        # Get chunk indices for this concept
        chunk_indices = self.concept_to_chunks.get(concept, [])
        
        # Get content from those chunks
        content = [self.chunks[i] for i in chunk_indices if i < len(self.chunks)]
        
        # Get related questions
        questions = self.concept_to_questions.get(concept, [])
        
        return {
            "content": content,
            "questions": questions
        }
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of dictionaries with 'text', 'metadata', and 'score' keys
        """
        # Make sure we have an index
        if self.faiss_index is None:
            self.faiss_index, self.chunks = self.load_index()
            
        if self.faiss_index is None:
            print("No index available for search")
            return []
            
        # Ensure query embedding is 2D and normalized
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue  # Invalid index
                
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(score)
            })
            
        return results