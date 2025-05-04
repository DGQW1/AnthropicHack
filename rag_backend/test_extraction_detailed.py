from backend import RAGSystem, ConceptExtractor
import re

# Sample text with various features to demonstrate extraction
sample_text = """
Network Flow and Max-Flow Min-Cut Theorem

The Max-Flow Min-Cut theorem states that in a flow network, the maximum amount of flow passing from the source to the sink is equal to the minimum capacity of a cut that separates the source and the sink.

Ford-Fulkerson algorithm can solve the max-flow problem in O(E|f|) time, where |f| is the maximum flow value. For networks with integer capacities, this algorithm is guaranteed to terminate.

The residual network Gf represents the remaining capacity in the network. An augmenting path is a path from source to sink in the residual network.

Ford-Fulkerson keeps finding augmenting paths and pushing flow along these paths until no more augmenting paths exist.

Applications of network flow include:
- Bipartite matching
- Circulation problems
- Image segmentation
- Baseball elimination
"""

# Initialize system
rag_system = RAGSystem("dummy_voyage_key")
extractor = rag_system.concept_extractor

# Function to demonstrate step-by-step extraction
def demonstrate_extraction(text):
    print("\n=== DETAILED CONCEPT EXTRACTION PROCESS ===\n")
    print("Original text:")
    print(text)
    print("\n--- STEP 1: Convert to lowercase and remove punctuation ---")
    
    # Simulate Step 1: Convert to lowercase and remove punctuation
    text_lower = text.lower()
    for char in ".,;:!?()[]{}\"\"''-—":
        text_lower = text_lower.replace(char, ' ')
    print(text_lower)
    
    # Simulate Step 2: Extract words and filter stop words
    print("\n--- STEP 2: Extract words and filter stop words ---")
    words = text_lower.split()
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 
                 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
                 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
                 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
                 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'}
    
    filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
    print(f"After removing stop words ({len(words) - len(filtered_words)} words removed):")
    print(filtered_words[:15], "... and more")
    
    # Simulate Step 3: Generate phrases
    print("\n--- STEP 3: Generate 1, 2, and 3-word phrases ---")
    phrases = []
    
    # Add single words
    phrases.extend([word for word in words if len(word) > 3 and word not in stop_words])
    
    # Add 2-word phrases
    for i in range(len(words) - 1):
        if words[i] not in stop_words or words[i+1] not in stop_words:
            phrases.append(words[i] + ' ' + words[i+1])
    
    # Add 3-word phrases
    for i in range(len(words) - 2):
        if (words[i] not in stop_words or 
            words[i+1] not in stop_words or 
            words[i+2] not in stop_words):
            phrases.append(words[i] + ' ' + words[i+1] + ' ' + words[i+2])
    
    print(f"Generated {len(phrases)} total phrases")
    print("Sample phrases:")
    print(phrases[:15], "... and more")
    
    # Simulate Step 4: Count frequencies
    print("\n--- STEP 4: Count phrase frequencies ---")
    phrase_counts = {}
    for phrase in phrases:
        if phrase in phrase_counts:
            phrase_counts[phrase] += 1
        else:
            phrase_counts[phrase] = 1
    
    # Show the most frequent phrases
    sorted_phrases = sorted(phrase_counts.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    print("Top 15 phrases by frequency:")
    for i, (phrase, count) in enumerate(sorted_phrases[:15]):
        print(f"  {i+1}. '{phrase}' appears {count} times")
    
    # Simulate Step 5-7: Select concepts
    print("\n--- STEP 5-7: Select final concepts ---")
    final_concepts = []
    for phrase, count in sorted_phrases:
        if count >= 2:  # Only add if it appears at least twice
            final_concepts.append(phrase)
        if len(final_concepts) >= 10:
            break
    
    print(f"Final {len(final_concepts)} concepts:")
    for i, concept in enumerate(final_concepts):
        print(f"  {i+1}. {concept}")
    
    # Compare with actual extractor output
    print("\n--- ACTUAL EXTRACTOR OUTPUT ---")
    actual_concepts = extractor.extract_concepts(text)
    print(f"Extracted {len(actual_concepts)} concepts:")
    for i, concept in enumerate(actual_concepts):
        print(f"  {i+1}. {concept}")

# Run the demonstration
demonstrate_extraction(sample_text) 