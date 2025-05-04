import os
import fitz  # PyMuPDF for PDF parsing
import numpy as np
import faiss
from pathlib import Path
import json
from collections import defaultdict
import re

# We'll need these packages, following the pipeline
# pip install pymupdf faiss-cpu llama_index voyageai anthropic

from voyageai import Client as VoyageClient
from anthropic import Anthropic  # Claude client

from llama_index.core import Document
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine

class ConceptExtractor:
    """Extracts key concepts from documents using LLM"""
    
    def __init__(self, anthropic_client):
        self.client = anthropic_client
    
    def extract_file_concept(self, file_path):
        """
        Process a file to generate a one-sentence summary and extract a single key concept.
        This function reads the entire file and uses Claude 3.5 Haiku to generate the summary and concept.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            dict: Contains 'summary' (one sentence) and 'key_concept' (single concept)
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "summary": f"File not found: {file_path}",
                    "key_concept": ""
                }
            
            # Read the entire file directly
            try:
                if file_path.lower().endswith('.pdf'):
                    # Handle PDF files
                    doc = fitz.open(file_path)
                    full_text = ""
                    for page in doc:
                        full_text += "\n\n" + page.get_text().strip()
                else:
                    # Handle text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
            except Exception as e:
                return {
                    "summary": f"Error reading file: {str(e)}",
                    "key_concept": ""
                }
            
            # Truncate if text is too long
            max_length = 20000  # Claude's context limit is higher, but we'll be conservative
            if len(full_text) > max_length:
                print(f"Text too long ({len(full_text)} chars), truncating to {max_length} chars")
                full_text = full_text[:max_length]
            
            # If Claude API is not available, fall back to basic extraction
            if not self.client:
                # Extract a basic summary from first paragraph
                paragraphs = full_text.split('\n\n')
                basic_summary = paragraphs[0][:200] + "..." if paragraphs and len(paragraphs[0]) > 200 else paragraphs[0] if paragraphs else "Summary not available"
                
                # Extract a basic concept using our frequency approach
                basic_concept = self.extract_concepts_basic(full_text)[0] if full_text else "concept not available"
                
                return {
                    "summary": basic_summary,
                    "key_concept": basic_concept
                }
            
            # First call: Summarize the entire file in one sentence
            try:
                summary_prompt = f"""
                This is content from a computer science education document.
                
                Content: {full_text}
                
                Please summarize this document in exactly one sentence. Focus on the main topic being discussed.
                """
                
                summary_response = self.client.messages.create(
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
                
                Content: {full_text}
                
                What is the single most important computer science concept discussed in this document?
                Return ONLY the concept name, with no additional text or explanation.
                """
                
                concept_response = self.client.messages.create(
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
                print(f"Error using Claude API: {e}")
                return {
                    "summary": f"Error generating summary with Claude: {str(e)}",
                    "key_concept": ""
                }
                
        except Exception as e:
            print(f"Error in file processing: {e}")
            return {
                "summary": f"Error processing file: {str(e)}",
                "key_concept": ""
            }
    
    def extract_concepts_basic(self, text):
        """Basic frequency-based concept extraction as fallback"""
        # Convert text to lowercase and remove punctuation
        text_lower = text.lower()
        for char in ".,;:!?()[]{}\"\"''-—":
            text_lower = text_lower.replace(char, ' ')
        
        # Get all words and phrases (1-3 words)
        words = text_lower.split()
        phrases = []
        
        # Add single words (excluding common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 
                     'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                     'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
                     'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                     'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
                     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                     'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
                     'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'}
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                phrases.append(word)
        
        # Add 2-3 word phrases
        for i in range(len(words) - 1):
            if words[i] not in stop_words or words[i+1] not in stop_words:
                phrases.append(words[i] + ' ' + words[i+1])
        
        for i in range(len(words) - 2):
            if (words[i] not in stop_words or 
                words[i+1] not in stop_words or 
                words[i+2] not in stop_words):
                phrases.append(words[i] + ' ' + words[i+1] + ' ' + words[i+2])
        
        # Count phrase frequencies
        phrase_counts = {}
        for phrase in phrases:
            if phrase in phrase_counts:
                phrase_counts[phrase] += 1
            else:
                phrase_counts[phrase] = 1
        
        # Sort by frequency and length
        sorted_phrases = sorted(phrase_counts.items(), 
                               key=lambda x: (x[1], len(x[0])), 
                               reverse=True)
        
        # Get top phrases as candidates (up to 20)
        candidate_concepts = []
        for phrase, count in sorted_phrases[:30]:
            # Only add if it appears at least twice
            if count >= 2:
                candidate_concepts.append(phrase)
            if len(candidate_concepts) >= 20:
                break
        
        # If we didn't find any candidates, add some generic ones
        if not candidate_concepts:
            # Find any capitalized terms (likely important)
            for word in text.split():
                original_word = word.strip().strip(".,;:!?()[]{}\"\"'")
                if original_word and original_word[0].isupper() and len(original_word) > 3:
                    candidate_concepts.append(original_word.lower())
                    if len(candidate_concepts) >= 5:
                        break
            
            # Still nothing? Add "computer science" as default
            if not candidate_concepts:
                candidate_concepts.append("computer science")
        
        # Limit to top 10
        final_concepts = candidate_concepts[:10]
        print(f"Extracted concepts via frequency: {final_concepts}")
        return final_concepts

class RAGSystem:
    """Main RAG system for processing documents and retrieving by concept"""
    
    def __init__(self, voyage_api_key, anthropic_api_key=None):
        self.voyage_client = VoyageClient(api_key=voyage_api_key)
        self.anthropic_client = None
        if anthropic_api_key:
            try:
                self.anthropic_client = Anthropic(api_key=anthropic_api_key)
            except Exception as e:
                print(f"Could not initialize Anthropic client: {e}")
        
        # Initialize concept extractor with or without Anthropic client
        self.concept_extractor = ConceptExtractor(self.anthropic_client)
        
        # Storage paths
        self.faiss_dir = Path("./data/faiss_store")
        self.faiss_dir.mkdir(exist_ok=True, parents=True)
        self.storage_dir = Path("./data/storage")
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # Concept mapping
        self.concept_to_chunks = defaultdict(list)
        self.concept_to_questions = defaultdict(list)
        self.all_concepts = set()
    
    def embed_texts(self, texts):
        """Embed texts using Voyage AI"""
        # Check the correct parameter name based on API version
        try:
            resp = self.voyage_client.embed(model="voyage-3", input=texts)
            return [item["embedding"] for item in resp["data"]]
        except TypeError:
            # Try with 'texts' parameter instead of 'input'
            try:
                resp = self.voyage_client.embed(model="voyage-3", texts=texts)
                return [item["embedding"] for item in resp["data"]]
            except Exception as e:
                print(f"Error with Voyage API: {e}")
                # Return dummy embeddings for testing (all zeros)
                return [np.zeros(768) for _ in texts]
    
    def extract_pdf_chunks(self, path, chunk_size=2000, overlap=200):
        """Extract text chunks from PDF files"""
        doc = fitz.open(path)
        full_text = ""
        for page in doc:
            full_text += "\n\n" + page.get_text().strip()
        
        chunks = []
        start = 0
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunk_text = full_text[start:end]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": Path(path).name,
                    "source_type": path.split("/")[-2] if "/" in path else "",
                    "start_pos": start,
                }
            })
            start += chunk_size - overlap
        
        return chunks
    
    def process_text_file(self, path, chunk_size=2000, overlap=200):
        """Process text files (non-PDF)"""
        with open(path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        chunks = []
        start = 0
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunk_text = full_text[start:end]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": Path(path).name,
                    "source_type": path.split("/")[-2] if "/" in path else "",
                    "start_pos": start,
                }
            })
            start += chunk_size - overlap
        
        return chunks
    
    def process_folder(self, folder_path):
        """Process all files in a folder"""
        # Convert to absolute path for clarity in debugging
        folder_path = os.path.abspath(folder_path)
        print(f"Processing folder: {folder_path}")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            return []
            
        # List all files to verify we can see them
        all_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                all_files.append(os.path.join(root, file))
        
        print(f"Found {len(all_files)} files in {folder_path}")
        if len(all_files) > 0:
            print(f"Sample files: {all_files[:3]}")
            
        chunks = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                path = os.path.join(root, file)
                try:
                    print(f"Processing file: {path}")
                    if file.lower().endswith('.pdf'):
                        file_chunks = self.extract_pdf_chunks(path)
                        print(f"  Extracted {len(file_chunks)} chunks from PDF")
                    elif file.lower().endswith(('.txt', '.md')):
                        file_chunks = self.process_text_file(path)
                        print(f"  Extracted {len(file_chunks)} chunks from text file")
                    else:
                        print(f"  Skipping unsupported file type: {file}")
                        continue  # Skip unsupported file types
                    
                    chunks.extend(file_chunks)
                    
                    # If this is from the questions folder, extract concepts and build mapping
                    if "270questions" in path:
                        for chunk in file_chunks:
                            concepts = self.concept_extractor.extract_concepts_basic(chunk["text"])
                            print(f"  Extracted concepts: {concepts}")
                            for concept in concepts:
                                self.concept_to_questions[concept].append({
                                    "text": chunk["text"],
                                    "source": chunk["metadata"]["source"]
                                })
                                self.all_concepts.add(concept)
                except Exception as e:
                    print(f"Error processing file {path}: {e}")
                    import traceback
                    print(traceback.format_exc())
        
        print(f"Total chunks extracted from {folder_path}: {len(chunks)}")
        return chunks
    
    def build_index(self, chunks):
        """Build vector index from chunks"""
        # Handle empty chunks case
        if not chunks:
            print("Warning: No chunks to index")
            # Create empty index files to avoid future errors
            faiss_index = faiss.IndexFlatIP(768)  # Default dimension
            faiss.write_index(faiss_index, str(self.faiss_dir / "content.index"))
            
            # Save empty mappings
            with open(self.storage_dir / "concept_to_chunks.json", "w") as f:
                json.dump({}, f)
            
            with open(self.storage_dir / "concept_to_questions.json", "w") as f:
                json.dump({}, f)
            
            with open(self.storage_dir / "all_concepts.json", "w") as f:
                json.dump(list(self.all_concepts), f)
            
            # Save empty chunks
            with open(self.storage_dir / "all_chunks.json", "w") as f:
                json.dump([], f)
            
            return faiss_index, chunks
        
        # Convert chunks to texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Get embeddings
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embed_texts(texts)
        embeddings = np.array(embeddings, dtype="float32")
        
        # Initialize FAISS
        d = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(d)  # inner-product index
        faiss.normalize_L2(embeddings)      # normalize vectors
        faiss_index.add(embeddings)         # add embeddings
        
        # Persist index to disk
        faiss.write_index(faiss_index, str(self.faiss_dir / "content.index"))
        print("FAISS index saved.")
        
        # Extract concepts from each chunk and build concept mapping
        for i, chunk in enumerate(chunks):
            concepts = self.concept_extractor.extract_concepts_basic(chunk["text"])
            for concept in concepts:
                self.concept_to_chunks[concept].append(i)  # Store chunk index
                self.all_concepts.add(concept)
        
        # Save concept mappings
        with open(self.storage_dir / "concept_to_chunks.json", "w") as f:
            json.dump({k: v for k, v in self.concept_to_chunks.items()}, f)
        
        with open(self.storage_dir / "concept_to_questions.json", "w") as f:
            json.dump({k: v for k, v in self.concept_to_questions.items()}, f)
        
        with open(self.storage_dir / "all_concepts.json", "w") as f:
            json.dump(list(self.all_concepts), f)
        
        # Save all chunks for retrieval
        with open(self.storage_dir / "all_chunks.json", "w") as f:
            json.dump(chunks, f)
        
        return faiss_index, chunks
    
    def process_all_folders(self, folders):
        """Process multiple folders and build index"""
        all_chunks = []
        for folder in folders:
            print(f"Processing folder: {folder}")
            folder_chunks = self.process_folder(folder)
            all_chunks.extend(folder_chunks)
            print(f"Extracted {len(folder_chunks)} chunks from {folder}")
        
        # Build the index with all chunks
        faiss_index, chunks = self.build_index(all_chunks)
        return faiss_index, chunks
    
    def get_all_concepts(self):
        """Return all extracted concepts"""
        # If we have the file already, load from there
        if os.path.exists(self.storage_dir / "all_concepts.json"):
            with open(self.storage_dir / "all_concepts.json", "r") as f:
                return json.load(f)
        return list(self.all_concepts)
    
    def get_content_for_concept(self, concept):
        """Get all content related to a specific concept"""
        # Create empty results to return if files don't exist
        empty_result = {
            "content": [],
            "questions": []
        }
        
        # Check if files exist
        if not os.path.exists(self.storage_dir / "concept_to_chunks.json") or \
           not os.path.exists(self.storage_dir / "concept_to_questions.json") or \
           not os.path.exists(self.storage_dir / "all_chunks.json"):
            print("Warning: Index files do not exist. Run process_all_folders first.")
            return empty_result
        
        # Load mappings if not already in memory
        if not self.concept_to_chunks and os.path.exists(self.storage_dir / "concept_to_chunks.json"):
            with open(self.storage_dir / "concept_to_chunks.json", "r") as f:
                self.concept_to_chunks = defaultdict(list, json.load(f))
        
        if not self.concept_to_questions and os.path.exists(self.storage_dir / "concept_to_questions.json"):
            with open(self.storage_dir / "concept_to_questions.json", "r") as f:
                self.concept_to_questions = defaultdict(list, json.load(f))
        
        # Load all chunks
        try:
            with open(self.storage_dir / "all_chunks.json", "r") as f:
                all_chunks = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Warning: all_chunks.json not found or invalid")
            return empty_result
        
        # Get chunk indices for this concept
        chunk_indices = self.concept_to_chunks.get(concept, [])
        
        # Get content from those chunks
        content = [all_chunks[i] for i in chunk_indices if i < len(all_chunks)]
        
        # Get related questions
        questions = self.concept_to_questions.get(concept, [])
        
        return {
            "content": content,
            "questions": questions
        }
    
    def summarize_file(self, file_path):
        """Summarize an entire file and extract its key concept"""
        try:
            # Determine file type and extract chunks appropriately
            chunks = []
            if file_path.lower().endswith('.pdf'):
                chunks = self.extract_pdf_chunks(file_path)
            elif file_path.lower().endswith(('.txt', '.md')):
                chunks = self.process_text_file(file_path)
            else:
                return {"summary": f"Unsupported file type: {file_path}", "key_concept": ""}
                
            if not chunks:
                return {"summary": "Could not extract content from file.", "key_concept": ""}
            
            # Combine chunk texts, limiting to prevent token overflow
            all_text = ""
            for chunk in chunks:
                # Handle both dictionary format and object format
                if isinstance(chunk, dict):
                    chunk_text = chunk.get("text", "")
                else:
                    chunk_text = getattr(chunk, "text", "")
                    
                all_text += chunk_text + "\n\n"
                if len(all_text) > 12000:  # Keep under Claude's context window
                    break
            
            # Use Claude to generate summary and key concept
            if self.concept_extractor.client:
                try:
                    prompt = f"""
                    This is content from a computer science education document.
                    
                    Content: {all_text[:12000]}
                    
                    Please provide:
                    1. A concise one-sentence summary of what this document discusses
                    2. The single most important computer science concept in this document
                    
                    Format your response as JSON with "summary" and "key_concept" fields.
                    """
                    
                    response = self.concept_extractor.client.messages.create(
                        model="claude-3-5-haiku-20241022",
                        max_tokens=200,
                        temperature=0,
                        system="You are a computer science education expert who creates concise, accurate summaries.",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    
                    # Parse the JSON response
                    import json
                    import re
                    
                    # Extract JSON from the response
                    response_text = response.content[0].text
                    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                    if not json_match:
                        json_match = re.search(r'{.*}', response_text, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1) if '```json' in response_text else json_match.group(0)
                        result = json.loads(json_str)
                        return result
                    
                    # Fallback if JSON parsing fails
                    return {
                        "summary": "Content extraction failed. Try processing the file again.",
                        "key_concept": ""
                    }
                    
                except Exception as e:
                    print(f"Error using Claude API for file summarization: {e}")
                    return {
                        "summary": "Failed to summarize content using AI.",
                        "key_concept": ""
                    }
            else:
                # Fallback without Claude API
                concepts = self.concept_extractor.extract_concepts_basic(all_text[:5000])
                return {
                    "summary": "Claude API not available for summarization.",
                    "key_concept": concepts[0] if concepts else ""
                }
                
        except Exception as e:
            print(f"Error in file summarization: {e}")
            return {
                "summary": f"Error processing file: {str(e)}",
                "key_concept": ""
            }

# Example usage
if __name__ == "__main__":
    # Initialize with API keys (replace these with actual API keys)
    VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "your_voyage_api_key")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your_anthropic_api_key")
    
    rag_system = RAGSystem(VOYAGE_API_KEY, ANTHROPIC_API_KEY)
    
    # Process all folders
    folders = [
        "270handout",
        "270questions",
        "270slides"
    ]
    
    # Process and build index
    faiss_index, chunks = rag_system.process_all_folders(folders)
    
    # Get all concepts (this would be used by the frontend)
    all_concepts = rag_system.get_all_concepts()
    print(f"Extracted {len(all_concepts)} concepts")
    
    # Example: Get content for a specific concept
    if all_concepts:
        example_concept = all_concepts[0]
        concept_content = rag_system.get_content_for_concept(example_concept)
        print(f"Content for concept '{example_concept}':")
        print(f"Found {len(concept_content['content'])} content chunks")
        print(f"Found {len(concept_content['questions'])} related questions") 