"""
Summary and key concept extraction module
"""

import re
from typing import List, Dict, Any, Optional, Union
from anthropic import Anthropic

from ..config import (
    ANTHROPIC_API_KEY, 
    LLM_MODEL, 
    LLM_TEMPERATURE, 
    SUMMARY_MAX_TOKENS, 
    CONCEPT_MAX_TOKENS,
    MAX_TEXT_LENGTH
)


class SummaryExtractor:
    """
    Extracts concise summaries and key concepts from text using Claude LLM
    """
    
    def __init__(self, anthropic_client: Optional[Anthropic] = None):
        """
        Initialize the summary extractor
        
        Args:
            anthropic_client: Optional Anthropic client. If None, will initialize with API key from config
        """
        self.client = anthropic_client
        if self.client is None and ANTHROPIC_API_KEY:
            try:
                self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
            except Exception as e:
                print(f"Could not initialize Anthropic client: {e}")
                self.client = None
    
    def extract_summary_and_concept(self, text: str) -> Dict[str, str]:
        """
        Generate a one-sentence summary and extract key concept from text
        
        Args:
            text: Text content to analyze
            
        Returns:
            dict: Contains 'summary' (one sentence) and 'key_concept' (single concept)
        """
        try:
            # Ensure we have content to process
            if not text:
                return {
                    "summary": "No content provided",
                    "key_concept": ""
                }
            
            # Check if text needs truncation
            if len(text) > MAX_TEXT_LENGTH:
                print(f"Text too long ({len(text)} chars), truncating to {MAX_TEXT_LENGTH} chars")
                text = text[:MAX_TEXT_LENGTH]
            
            # If Claude API is not available, fall back to basic extraction
            if not self.client:
                return self._basic_extraction(text)
            
            # Generate summary and extract key concept
            try:
                summary = self._generate_summary(text)
                key_concept = self._extract_key_concept(summary)
                
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
            print(f"Error in content processing: {e}")
            return {
                "summary": f"Error processing content: {str(e)}",
                "key_concept": ""
            }
    
    def _generate_summary(self, text: str) -> str:
        """
        Generate a one-sentence summary of the text
        
        Args:
            text: Text to summarize
            
        Returns:
            str: One-sentence summary
        """
        summary_prompt = f"""
        This is content from a computer science education document.
        
        Content: {text}
        
        Please summarize this document in exactly one sentence less than 20 words. Focus on the main topic being discussed.
        """
        
        summary_response = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
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
        return summary.strip('"\'')
    
    def _extract_key_concept(self, text: str) -> str:
        """
        Extract the single most important concept from the text
        
        Args:
            text: Text to analyze
            
        Returns:
            str: Key concept
        """
        concept_prompt = f"""
        This is content from a computer science education document.
        
        Content: {text}
        
        What is the single most important computer science concept discussed in this document?
        Return ONLY the concept name, with no additional text or explanation.
        """
        
        concept_response = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=CONCEPT_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
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
        return key_concept.strip('"\'')
    
    def _basic_extraction(self, text: str) -> Dict[str, str]:
        """
        Basic extraction of summary and concept when Claude API is not available
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Contains 'summary' and 'key_concept'
        """
        # Extract a basic summary from first paragraph
        paragraphs = text.split('\n\n')
        basic_summary = paragraphs[0][:200] + "..." if paragraphs and len(paragraphs[0]) > 200 else paragraphs[0] if paragraphs else "Summary not available"
        
        # Extract a basic concept using frequency approach
        basic_concept = self._extract_concepts_basic(text)[0] if text else "concept not available"
        
        return {
            "summary": basic_summary,
            "key_concept": basic_concept
        }
    
    def _extract_concepts_basic(self, text: str) -> List[str]:
        """
        Basic frequency-based concept extraction as fallback
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            List of concept strings
        """
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
        return final_concepts