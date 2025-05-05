#!/usr/bin/env python3
"""
PDF Splitter for CS270 Course Materials
Splits homework PDFs into individual question files.
"""

import os
import sys
import re
import argparse
from PyPDF2 import PdfReader, PdfWriter

def detect_questions(pdf_path):
    """
    Detect question boundaries in a PDF file.
    Returns a list of (page_num, y_position) tuples for each question start.
    """
    print(f"Analyzing PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    
    # Question markers list
    question_markers = []
    
    # For each page in the PDF
    for page_num, page in enumerate(reader.pages):
        # Extract text from the page
        text = page.extract_text()
        
        # Look for question patterns: "1.", "2.", etc.
        for match in re.finditer(r'(?m)^(\d+)\.\s', text):
            question_num = int(match.group(1))
            position = match.start()
            print(f"  Found Question {question_num} on page {page_num+1} at position {position}")
            question_markers.append((page_num, question_num, position))
    
    # Sort by page and position
    question_markers.sort(key=lambda x: (x[0], x[2]))
    return question_markers

def split_pdf(pdf_path, output_dir=None):
    """
    Split a PDF file into separate files for each question.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return []
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pdf_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Find question markers
    question_markers = detect_questions(pdf_path)
    if not question_markers:
        print("No questions detected in the PDF.")
        return []
    
    # Open the PDF
    reader = PdfReader(pdf_path)
    output_files = []
    
    # For each question
    for i, (page_num, question_num, _) in enumerate(question_markers):
        # Create a new PDF for this question
        writer = PdfWriter()
        
        # Determine page range for this question
        start_page = page_num
        if i < len(question_markers) - 1:
            end_page = question_markers[i+1][0]  # Next question's page
        else:
            end_page = len(reader.pages) - 1  # Last page
        
        # Add pages to the new PDF
        for p in range(start_page, end_page + 1):
            writer.add_page(reader.pages[p])
        
        # Save the new PDF
        output_filename = f"{base_name}_question{question_num}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        
        print(f"Created: {output_filename}")
        output_files.append(output_path)
    
    return output_files

def main():
    parser = argparse.ArgumentParser(description="Split PDF files into questions")
    parser.add_argument("pdf_file", help="Path to the PDF file to split")
    parser.add_argument("--output-dir", help="Directory to save split PDFs")
    args = parser.parse_args()
    
    split_pdf(args.pdf_file, args.output_dir)

if __name__ == "__main__":
    main() 