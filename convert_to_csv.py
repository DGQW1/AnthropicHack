import json
import csv
import os

# Input JSON file
JSON_FILE = 'extracted_concepts.json'
# Output CSV file
CSV_FILE = 'extracted_concepts.csv'

def convert_json_to_csv():
    """Convert the extracted concepts JSON to CSV format"""
    # Check if JSON file exists
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found")
        return
    
    # Load the JSON data
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
    
    # Count documents
    total_docs = sum(len(folder_data) for folder_data in data.values())
    print(f"Found {total_docs} documents in {len(data)} folders")
    
    # Prepare CSV rows
    rows = []
    for folder, items in data.items():
        for item in items:
            # Skip items with errors or missing content
            if not item.get('summary') or item.get('summary', '').startswith('Error'):
                continue
                
            # Add the row
            rows.append({
                'folder': folder,
                'filename': item.get('file', ''),
                'filepath': item.get('path', ''),
                'key_concept': item.get('key_concept', ''),
                'summary': item.get('summary', '')
            })
    
    if not rows:
        print("No valid data to write to CSV")
        return
    
    # Write to CSV
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['folder', 'filename', 'filepath', 'key_concept', 'summary'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Successfully wrote {len(rows)} entries to {CSV_FILE}")

if __name__ == "__main__":
    convert_json_to_csv() 