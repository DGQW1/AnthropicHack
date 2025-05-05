#!/usr/bin/env python3
"""
Main entry point for the RAG application
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional

from pipeline.rag_pipeline import RAGPipeline
from utils.logging import get_logger

logger = get_logger(__name__)


def index_command(args):
    """Handler for the index command"""
    pipeline = RAGPipeline()
    
    # Parse folders to index
    folders = args.folders.split(',') if args.folders else None
    
    logger.info(f"Indexing folders: {folders}")
    success = pipeline.index_folders(folders)
    
    if success:
        print("Indexing completed successfully")
        return 0
    else:
        print("Indexing failed")
        return 1


def query_command(args):
    """Handler for the query command"""
    pipeline = RAGPipeline()
    
    # Process query
    results = pipeline.process_query(args.query, k=args.num_results, category_filter=args.category)
    
    # Display results
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    print("\n" + "="*80)
    print(f"QUERY: {results['query']}")
    print("="*80 + "\n")
    
    if results.get("response"):
        print("RESPONSE:")
        print(results["response"])
        print("\n" + "-"*80 + "\n")
    
    print("TOP RESULTS:")
    for i, result in enumerate(results["results"]):
        metadata = result.get("metadata", {})
        print(f"\n[Result {i+1}] - Score: {result.get('score', 0):.4f}")
        print(f"Source: {metadata.get('source', 'Unknown')} | Category: {metadata.get('category', 'Unknown')}")
        print("-"*40)
        
        # Truncate text for display
        text = result.get("text", "")
        if len(text) > 200:
            text = text[:200] + "..."
        print(text)
    
    print("\n" + "="*80 + "\n")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return 0


def process_file_command(args):
    """Handler for the process-file command"""
    pipeline = RAGPipeline()
    
    # Process file
    metadata = pipeline.process_file(args.file_path)
    
    # Display results
    print("\n" + "="*80)
    print(f"FILE: {metadata.get('filename', args.file_path)}")
    print("="*80 + "\n")
    
    print(f"Category: {metadata.get('category', 'Unknown')}")
    print(f"Key Concept: {metadata.get('key_concept', 'Not extracted')}")
    print(f"Summary: {metadata.get('summary', 'Not available')}")
    
    print("\n" + "="*80 + "\n")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {args.output}")
    
    return 0


def concepts_command(args):
    """Handler for the concepts command"""
    pipeline = RAGPipeline()
    
    if args.concept:
        # Get content for a specific concept
        concept_content = pipeline.get_concept_content(args.concept)
        
        print("\n" + "="*80)
        print(f"CONCEPT: {args.concept}")
        print("="*80 + "\n")
        
        print(f"Found {len(concept_content.get('content', []))} content chunks")
        print(f"Found {len(concept_content.get('questions', []))} related questions")
        
        # Display content samples
        if concept_content.get('content'):
            print("\nCONTENT SAMPLES:")
            for i, content in enumerate(concept_content['content'][:3]):  # Show first 3
                metadata = content.get("metadata", {})
                print(f"\n[Content {i+1}]")
                print(f"Source: {metadata.get('source', 'Unknown')} | Category: {metadata.get('category', 'Unknown')}")
                print("-"*40)
                
                # Truncate text for display
                text = content.get("text", "")
                if len(text) > 200:
                    text = text[:200] + "..."
                print(text)
        
        # Display question samples
        if concept_content.get('questions'):
            print("\nQUESTION SAMPLES:")
            for i, question in enumerate(concept_content['questions'][:3]):  # Show first 3
                print(f"\n[Question {i+1}]")
                print(f"Source: {question.get('source', 'Unknown')}")
                print("-"*40)
                
                # Truncate text for display
                text = question.get("text", "")
                if len(text) > 200:
                    text = text[:200] + "..."
                print(text)
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(concept_content, f, indent=2)
            print(f"\nConcept content saved to {args.output}")
        
    else:
        # List all concepts
        concepts = pipeline.get_all_concepts()
        concepts.sort()  # Sort alphabetically
        
        print("\n" + "="*80)
        print(f"FOUND {len(concepts)} CONCEPTS")
        print("="*80 + "\n")
        
        # Display in columns
        col_width = max(len(c) for c in concepts) + 4
        cols = 3
        
        for i in range(0, len(concepts), cols):
            row = concepts[i:i+cols]
            print("".join(concept.ljust(col_width) for concept in row))
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(concepts, f, indent=2)
            print(f"\nConcepts saved to {args.output}")
    
    return 0


def main():
    """Main entry point of the application"""
    parser = argparse.ArgumentParser(description="RAG Application for Computer Science Education")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index content in specified folders")
    index_parser.add_argument("--folders", type=str, help="Comma-separated list of folders to index")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the indexed content")
    query_parser.add_argument("query", type=str, help="Query string")
    query_parser.add_argument("--num-results", "-n", type=int, default=5, help="Number of results to return")
    query_parser.add_argument("--category", "-c", type=str, help="Filter results by category (slide, handout, question)")
    query_parser.add_argument("--output", "-o", type=str, help="Save results to this file")
    
    # Process-file command
    process_file_parser = subparsers.add_parser("process-file", help="Process a single file")
    process_file_parser.add_argument("file_path", type=str, help="Path to the file to process")
    process_file_parser.add_argument("--output", "-o", type=str, help="Save metadata to this file")
    
    # Concepts command
    concepts_parser = subparsers.add_parser("concepts", help="List all concepts or get content for a specific concept")
    concepts_parser.add_argument("--concept", "-c", type=str, help="Specific concept to get content for")
    concepts_parser.add_argument("--output", "-o", type=str, help="Save results to this file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "index":
        return index_command(args)
    elif args.command == "query":
        return query_command(args)
    elif args.command == "process-file":
        return process_file_command(args)
    elif args.command == "concepts":
        return concepts_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())