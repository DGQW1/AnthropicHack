import json
import sys
import os
from typing import List, Dict, Any

# Input JSON file containing extracted concepts
JSON_FILE = 'extracted_concepts.json'

def load_data() -> Dict[str, List[Dict[str, Any]]]:
    """Load the extracted concepts from JSON file"""
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found")
        return {}
    
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}

def search_concepts(query: str, data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Search for concepts matching the query
    
    Args:
        query: The search term to look for
        data: The extracted concept data
        
    Returns:
        List of matching documents with their information
    """
    query = query.lower()
    results = []
    
    for folder, items in data.items():
        for item in items:
            # Skip items with errors or missing content
            if not item.get('summary') or item.get('summary', '').startswith('Error'):
                continue
            
            # Search in key concept, summary, and filename
            key_concept = item.get('key_concept', '').lower()
            summary = item.get('summary', '').lower()
            filename = item.get('file', '').lower()
            
            if (query in key_concept or 
                query in summary or 
                query in filename):
                
                results.append({
                    'folder': folder,
                    'filename': item.get('file', ''),
                    'key_concept': item.get('key_concept', ''),
                    'summary': item.get('summary', ''),
                    'path': item.get('path', '')
                })
    
    return results

def print_results(results: List[Dict[str, Any]], query: str):
    """Print search results in a readable format"""
    if not results:
        print(f"No results found for query: '{query}'")
        return
    
    print(f"Found {len(results)} results for query: '{query}'\n")
    
    for i, result in enumerate(results, 1):
        print(f"#{i}: {result['filename']} ({result['folder']})")
        print(f"Key Concept: {result['key_concept']}")
        print(f"Summary: {result['summary']}")
        print(f"Path: {result['path']}")
        print("-" * 80)

def main():
    """Main search function"""
    # Check for command line arguments
    if len(sys.argv) < 2:
        print("Usage: python search_concepts.py <search_query>")
        return
    
    # Extract search query from arguments
    query = ' '.join(sys.argv[1:])
    
    # Load data
    data = load_data()
    if not data:
        return
    
    # Search for the query
    results = search_concepts(query, data)
    
    # Print results
    print_results(results, query)

if __name__ == "__main__":
    main() 