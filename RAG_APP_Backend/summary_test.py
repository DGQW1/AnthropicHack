#!/usr/bin/env python3
"""
Test script for the SummaryExtractor on PDF files in the 270materials directory.
This script processes all PDFs in the 270handout, 270questions, and 270slides folders,
extracts summaries and key concepts, and saves the results to a JSON file.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from anthropic import Anthropic
import concurrent.futures

# Add the current directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our app
from config import ANTHROPIC_API_KEY, MAX_TEXT_LENGTH
from loaders.pdf_loader import PDFLoader
from summary_extractor.summary_extractor import SummaryExtractor


def find_all_pdfs(base_dir: str) -> Dict[str, List[str]]:
    """
    Find all PDF files in the 270handout, 270questions, and 270slides subdirectories
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        Dictionary with category keys and lists of PDF paths
    """
    pdf_files = {
        "handout": [],
        "questions": [],
        "slides": []
    }
    
    # Check if base directory exists
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return pdf_files
    
    # Look for PDF files in the 3 subdirectories
    subdirs = ["270handout", "270questions", "270slides"]
    for subdir in subdirs:
        subdir_path = base_path / subdir
        if not subdir_path.exists():
            print(f"Warning: Subdirectory {subdir} not found in {base_dir}")
            continue
            
        # Find all PDFs in this subdirectory
        for root, _, files in os.walk(subdir_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    
                    # Categorize by directory
                    if "270handout" in file_path:
                        pdf_files["handout"].append(file_path)
                    elif "270questions" in file_path:
                        pdf_files["question"].append(file_path)
                    elif "270slides" in file_path:
                        pdf_files["slide"].append(file_path)
    
    return pdf_files


def process_pdf(file_path: str, anthropic_client: Anthropic) -> Dict[str, Any]:
    """
    Process a single PDF file to extract text and generate summary
    
    Args:
        file_path: Path to the PDF file
        anthropic_client: Anthropic client
        
    Returns:
        Dictionary with file info, summary and key concept
    """
    print(f"Processing: {file_path}")
    start_time = time.time()
    
    try:
        # Extract text from PDF
        pdf_text = PDFLoader.extract_text_with_limit(file_path)
        
        # Create summary extractor
        summary_extractor = SummaryExtractor(anthropic_client)
        
        # Extract summary and key concept
        result = summary_extractor.extract_summary_and_concept(pdf_text)
        
        # Add file info
        result["file_path"] = file_path
        result["filename"] = os.path.basename(file_path)
        result["category"] = PDFLoader.determine_category(file_path)
        result["processing_time"] = time.time() - start_time
        
        print(f"✓ Completed {result['filename']} ({result['processing_time']:.2f}s)")
        print(f"  Summary: {result['summary']}")
        print(f"  Key concept: {result['key_concept']}")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"✗ Error processing {file_path}: {e}")
        
        return {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "category": PDFLoader.determine_category(file_path),
            "summary": f"Error: {str(e)}",
            "key_concept": "",
            "error": str(e),
            "processing_time": processing_time
        }


def process_pdfs_sequential(pdf_files: Dict[str, List[str]], anthropic_client: Anthropic) -> List[Dict[str, Any]]:
    """
    Process PDF files sequentially
    
    Args:
        pdf_files: Dictionary with category keys and lists of PDF paths
        anthropic_client: Anthropic client
        
    Returns:
        List of dictionaries with results
    """
    all_results = []
    total_files = sum(len(files) for files in pdf_files.values())
    processed = 0
    
    # Process each category
    for category, files in pdf_files.items():
        print(f"\nProcessing {len(files)} {category} files:")
        
        for file_path in files:
            # Process this file
            result = process_pdf(file_path, anthropic_client)
            all_results.append(result)
            
            # Update progress
            processed += 1
            print(f"Progress: {processed}/{total_files} ({processed/total_files*100:.1f}%)")
    
    return all_results


def process_pdfs_parallel(pdf_files: Dict[str, List[str]], anthropic_client: Anthropic, max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    Process PDF files in parallel
    
    Args:
        pdf_files: Dictionary with category keys and lists of PDF paths
        anthropic_client: Anthropic client
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of dictionaries with results
    """
    all_results = []
    all_files = []
    
    # Flatten the files list
    for category, files in pdf_files.items():
        print(f"Found {len(files)} {category} files")
        all_files.extend(files)
    
    total_files = len(all_files)
    print(f"\nProcessing {total_files} PDF files with {max_workers} workers")
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dict of {future: file_path} to track which future is for which file
        future_to_file = {
            executor.submit(process_pdf, file_path, anthropic_client): file_path
            for file_path in all_files
        }
        
        # Process completed futures as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error in worker for {file_path}: {e}")
                all_results.append({
                    "file_path": file_path,
                    "filename": os.path.basename(file_path),
                    "category": PDFLoader.determine_category(file_path),
                    "summary": f"Worker error: {str(e)}",
                    "key_concept": "",
                    "error": str(e)
                })
            
            # Update progress
            print(f"Progress: {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)")
    
    return all_results


def main():
    """Main entry point of the script"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test SummaryExtractor on PDF files")
    parser.add_argument("--dir", type=str, default="../270materials", 
                        help="Base directory containing the PDF folders")
    parser.add_argument("--output", type=str, default="summary_results.json",
                        help="Output JSON file path")
    parser.add_argument("--parallel", action="store_true", 
                        help="Process files in parallel")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel workers (only with --parallel)")
    args = parser.parse_args()
    
    # Initialize Anthropic client
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY is not set in the environment variables")
        return 1
    
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Find all PDF files
    print(f"Scanning directory: {args.dir}")
    pdf_files = find_all_pdfs(args.dir)
    
    # Count total files
    total_files = sum(len(files) for files in pdf_files.values())
    print(f"Found {total_files} PDF files to process:")
    for category, files in pdf_files.items():
        print(f"  {category}: {len(files)} files")
    
    if total_files == 0:
        print("No PDF files found. Exiting.")
        return 0
    
    # Process all files
    start_time = time.time()
    
    if args.parallel:
        print(f"Processing files in parallel with {args.workers} workers")
        results = process_pdfs_parallel(pdf_files, anthropic_client, args.workers)
    else:
        print("Processing files sequentially")
        results = process_pdfs_sequential(pdf_files, anthropic_client)
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    success_count = sum(1 for r in results if "error" not in r)
    error_count = len(results) - success_count
    
    print(f"\nProcessing complete in {total_time:.2f} seconds")
    print(f"Successfully processed: {success_count}/{len(results)} files")
    print(f"Failed: {error_count}/{len(results)} files")
    
    # Save results to JSON
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())