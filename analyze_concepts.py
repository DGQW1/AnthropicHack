import csv
from collections import Counter
import matplotlib.pyplot as plt
import os

# Input CSV file
CSV_FILE = 'extracted_concepts.csv'

def analyze_concepts():
    """Analyze the frequency of key concepts in our extracted data"""
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found")
        return
    
    # Load the CSV data
    concepts = []
    folders = set()
    folder_concepts = {}
    
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                concept = row.get('key_concept', '').strip()
                folder = row.get('folder', '').strip()
                
                if concept:
                    concepts.append(concept)
                    folders.add(folder)
                    
                    # Track concepts by folder
                    if folder not in folder_concepts:
                        folder_concepts[folder] = []
                    folder_concepts[folder].append(concept)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if not concepts:
        print("No concepts found in the CSV file")
        return
    
    # Count concept frequencies
    concept_counts = Counter(concepts)
    top_concepts = concept_counts.most_common()
    
    # Print overall statistics
    print(f"Found {len(concept_counts)} unique concepts across {len(concepts)} documents")
    print("\nTop 10 Most Common Concepts:")
    for concept, count in top_concepts[:10]:
        print(f"  {concept}: {count} occurrences")
    
    # Print stats by folder
    print("\nConcepts by Folder:")
    for folder in sorted(folders):
        if folder in folder_concepts:
            folder_concept_counts = Counter(folder_concepts[folder])
            top_folder_concepts = folder_concept_counts.most_common(5)
            
            print(f"\n{folder} ({len(folder_concepts[folder])} documents):")
            for concept, count in top_folder_concepts:
                print(f"  {concept}: {count} occurrences")
    
    # Save results to text file
    with open('concept_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(f"Concept Analysis Results\n")
        f.write(f"======================\n\n")
        f.write(f"Found {len(concept_counts)} unique concepts across {len(concepts)} documents\n\n")
        
        f.write("All Concepts by Frequency:\n")
        for concept, count in top_concepts:
            f.write(f"  {concept}: {count} occurrences\n")
        
        f.write("\nConcepts by Folder:\n")
        for folder in sorted(folders):
            if folder in folder_concepts:
                folder_concept_counts = Counter(folder_concepts[folder])
                top_folder_concepts = folder_concept_counts.most_common()
                
                f.write(f"\n{folder} ({len(folder_concepts[folder])} documents):\n")
                for concept, count in top_folder_concepts:
                    f.write(f"  {concept}: {count} occurrences\n")
    
    print(f"\nFull analysis saved to concept_analysis.txt")
    
    # Try to create a visualization if matplotlib is available
    try:
        # Plot top 10 concepts
        top_10_concepts = dict(top_concepts[:10])
        plt.figure(figsize=(12, 6))
        plt.bar(top_10_concepts.keys(), top_10_concepts.values())
        plt.title('Top 10 Key Concepts')
        plt.xlabel('Concept')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_concepts.png')
        print("Created visualization: top_concepts.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")

if __name__ == "__main__":
    analyze_concepts() 