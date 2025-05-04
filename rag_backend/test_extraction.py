from backend import RAGSystem, ConceptExtractor

# Sample text to test concept extraction
test_texts = [
    """
    Greedy algorithms are easy to design, easy to write, and very efficient. 
    The tradeoff is that they also tend to be unclear. Why does the algorithm work? 
    How can you convince someone that the algorithm is correct?
    
    Greedy algorithms often require a proof of correctness. It is not clear at all that our proposed
    algorithm for Interval Scheduling is actually correct.
    """,
    
    """
    Dynamic Programming is a powerful technique for solving optimization problems.
    The key idea is to break down a complex problem into simpler subproblems and 
    solve each subproblem only once, storing the solutions to avoid redundant calculations.
    This approach is particularly useful for problems with overlapping subproblems and optimal substructure.
    """,
    
    """
    A graph is bipartite if nodes can be colored red or blue such that all edges have one red
    end and one blue end. Bipartite Graph Questions: What was wrong with the second example? 
    Why was it impossible to color correctly? Is the converse of that statement necessarily true?
    How could one write an algorithm to test whether a graph is bipartite?
    """
]

# Initialize RAG system (without actual API keys since we're just testing extraction)
rag_system = RAGSystem("dummy_voyage_key")

# Test the concept extractor
extractor = rag_system.concept_extractor

print("Testing concept extraction on sample texts:\n")

# Process each test text
for i, text in enumerate(test_texts):
    print(f"=== Sample Text {i+1} ===")
    print(f"{text[:100]}...\n")
    
    # Extract concepts
    concepts = extractor.extract_concepts(text)
    
    print(f"Extracted {len(concepts)} concepts:")
    for j, concept in enumerate(concepts):
        print(f"  {j+1}. {concept}")
    print("\n")

print("Extraction process explanation:")
print("1. The text is converted to lowercase and punctuation is removed")
print("2. Common stop words are filtered out")
print("3. The system identifies single words, 2-word phrases, and 3-word phrases")
print("4. These are counted for frequency")
print("5. The most frequent terms (appearing at least twice) are selected as concepts")
print("6. If no concepts are found, capitalized terms are identified as potential concepts")
print("7. The top 10 concepts (at most) are returned") 