// Types for our concept data
export type MaterialFile = {
  id: string;
  name: string;
  type: 'slides' | 'notes' | 'questions';
  url: string;
};

export type QuestionDifficulty = 'easy' | 'medium' | 'hard';

export type Question = {
  id: string;
  title: string;
  difficulty: QuestionDifficulty;
  isViewed?: boolean;
  url: string;
};

export type ConceptData = {
  name: string;
  materials: MaterialFile[];
  questions: Question[];
  // Additional fields for expanded content
  overview: string;
  keyCharacteristics?: string[];
  commonApplications?: string[];
};

// Mock data for demonstration
const conceptsData: Record<string, ConceptData> = {
  'dynamic-programming': {
    name: 'Dynamic Programming',
    overview: `<p>Dynamic Programming is a method for solving complex problems by breaking them down into simpler subproblems. It is applicable to problems that have overlapping subproblems and optimal substructure. Dynamic Programming stores the solutions of overlapping subproblems to avoid redundant calculations, which is the key optimization compared to naive recursion. This technique was pioneered by Richard Bellman in the 1950s.</p>`,
    keyCharacteristics: [
      'Optimal substructure - optimal solutions to subproblems can be used to find the optimal solution',
      'Overlapping subproblems - same subproblems are solved multiple times',
      'Memoization or tabulation are used to store solutions to subproblems'
    ],
    commonApplications: [
      'Fibonacci sequence',
      'Shortest path problems',
      'Knapsack problem',
      'Sequence alignment',
      'Matrix chain multiplication'
    ],
    materials: [
      { id: '1', name: 'Introduction to Dynamic Programming', type: 'slides', url: '/pdfs/dynamic-programming/dp_slides.pdf' },
      { id: '2', name: 'Dynamic Programming Example Problems', type: 'notes', url: '/pdfs/dynamic-programming/dp_notes.pdf' },
      { id: '3', name: 'DP Practice Problems Set 1', type: 'questions', url: '/pdfs/dynamic-programming/dp_questions.pdf' },
    ],
    questions: [
      { id: '1', title: 'Binary Search Tree Problem', difficulty: 'medium', url: '/pdfs/dynamic-programming/binary_search_tree.pdf' },
      { id: '2', title: 'Powder Mixing Problem', difficulty: 'hard', url: '/pdfs/dynamic-programming/powder_problem.pdf' },
      { id: '3', title: 'Gas Station Problem', difficulty: 'easy', url: '/pdfs/dynamic-programming/gas_station_problem.pdf' },
    ],
  },
  'divide-and-conquer': {
    name: 'Divide and Conquer',
    overview: `<p>Divide and Conquer is an algorithmic paradigm that solves a problem by recursively breaking it down into simpler subproblems, solving those subproblems, and combining their solutions. This technique is the basis of efficient algorithms for many problems, such as sorting (merge sort, quicksort), multiplying large numbers, and computing the discrete Fourier transform. Unlike dynamic programming, divide and conquer approaches typically don't store intermediate results.</p>`,
    keyCharacteristics: [
      'Divide the problem into smaller subproblems',
      'Conquer the subproblems by solving them recursively',
      'Combine the solutions to create a solution to the original problem'
    ],
    commonApplications: [
      'Merge sort',
      'Quick sort',
      'Binary search',
      'Fast Fourier Transform',
      'Strassen\'s matrix multiplication'
    ],
    materials: [
      { id: '1', name: 'Divide & Conquer Principles', type: 'slides', url: '/pdfs/dynamic-programming/dp_slides.pdf' },
      { id: '2', name: 'Sorting Algorithms Implementation', type: 'notes', url: '/pdfs/dynamic-programming/dp_notes.pdf' },
      { id: '3', name: 'Divide & Conquer Problem Set', type: 'questions', url: '/pdfs/dynamic-programming/dp_questions.pdf' },
    ],
    questions: [
      { id: '1', title: 'Binary Search Tree Problem', difficulty: 'medium', url: '/pdfs/dynamic-programming/binary_search_tree.pdf' },
      { id: '2', title: 'Powder Mixing Problem', difficulty: 'hard', url: '/pdfs/dynamic-programming/powder_problem.pdf' },
      { id: '3', title: 'Gas Station Problem', difficulty: 'easy', url: '/pdfs/dynamic-programming/gas_station_problem.pdf' },
    ],
  },
  'greedy': {
    name: 'Greedy Algorithms',
    overview: `<p>Greedy algorithms make locally optimal choices at each stage with the hope of finding a global optimum solution. While this approach doesn't always produce an optimal solution, greedy heuristics often yield locally optimal solutions that approximate a globally optimal solution in a reasonable amount of time. Greedy algorithms work best for problems with "greedy choice property" (making locally optimal choices leads to a global optimal solution) and "optimal substructure" (optimal solution contains optimal solutions to subproblems).</p>`,
    keyCharacteristics: [
      'Makes locally optimal choices at each step',
      'Never reconsiders previous choices',
      'Generally more efficient than dynamic programming',
      'Does not always yield optimal solutions'
    ],
    commonApplications: [
      'Huffman coding',
      'Dijkstra\'s algorithm for shortest path',
      'Prim\'s and Kruskal\'s algorithms for minimum spanning tree',
      'Activity selection problem',
      'Fractional knapsack problem'
    ],
    materials: [
      { id: '1', name: 'Greedy Algorithm Fundamentals', type: 'slides', url: '/pdfs/dynamic-programming/dp_slides.pdf' },
      { id: '2', name: 'Greedy vs Dynamic Programming', type: 'notes', url: '/pdfs/dynamic-programming/dp_notes.pdf' },
      { id: '3', name: 'Greedy Algorithms Practice', type: 'questions', url: '/pdfs/dynamic-programming/dp_questions.pdf' },
    ],
    questions: [
      { id: '1', title: 'Binary Search Tree Problem', difficulty: 'medium', url: '/pdfs/dynamic-programming/binary_search_tree.pdf' },
      { id: '2', title: 'Powder Mixing Problem', difficulty: 'hard', url: '/pdfs/dynamic-programming/powder_problem.pdf' },
      { id: '3', title: 'Gas Station Problem', difficulty: 'easy', url: '/pdfs/dynamic-programming/gas_station_problem.pdf' },
    ],
  },
  'network-flow': {
    name: 'Network Flow',
    overview: `<p>Network Flow is a specialized area of graph theory focused on finding the maximum flow through a flow network. Problems involve a graph with nodes connected by edges, where each edge has a capacity that limits flow. Key algorithms include Ford-Fulkerson, Edmonds-Karp, and Push-Relabel, which find augmenting paths or use other techniques to maximize flow. Network flow algorithms are used in transportation planning, resource allocation, bipartite matching, and many other optimization problems.</p>`,
    keyCharacteristics: [
      'Source and sink nodes with flow conservation at intermediary nodes',
      'Capacity constraints on edges limiting flow',
      'Maximum flow equals minimum cut (Max-Flow Min-Cut Theorem)',
      'Residual networks used to find augmenting paths'
    ],
    commonApplications: [
      'Maximum bipartite matching',
      'Airline scheduling',
      'Image segmentation',
      'Baseball elimination',
      'Network reliability'
    ],
    materials: [
      { id: '1', name: 'Introduction to Dynamic Programming', type: 'slides', url: '/pdfs/dynamic-programming/dp_slides.pdf' },
      { id: '2', name: 'Dynamic Programming Example Problems', type: 'notes', url: '/pdfs/dynamic-programming/dp_notes.pdf' },
      { id: '3', name: 'DP Practice Problems Set 1', type: 'questions', url: '/pdfs/dynamic-programming/dp_questions.pdf' },
    ],
    questions: [
      { id: '1', title: 'Fibonacci Sequence', difficulty: 'easy', url: '#' },
      { id: '2', title: 'Longest Common Subsequence', difficulty: 'medium', url: '#' },
      { id: '3', title: 'Coin Change Problem', difficulty: 'medium', url: '#' },
      { id: '4', title: 'Knapsack Problem', difficulty: 'hard', url: '#' },
    ],
  },
};

// Function to simulate fetching data from a database
export async function getConceptData(conceptSlug: string): Promise<ConceptData | null> {
  // In a real implementation, this would be a database query
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      const data = conceptsData[conceptSlug];
      resolve(data || null);
    }, 300);
  });
}

// Function to get all available concepts
export async function getAllConcepts(): Promise<{ slug: string; name: string }[]> {
  // In a real implementation, this would be a database query
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      // Get the exact slugs as they appear in the conceptsData object keys
      const concepts = Object.keys(conceptsData).map(slug => ({
        slug,
        name: conceptsData[slug].name
      }));
      resolve(concepts);
    }, 300);
  });
} 