'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import MainLayout from '@/components/layout/MainLayout';
import ChatBox from '@/components/chat/ChatBox';
import { getAllConcepts, getConceptData, ConceptData } from '@/services/conceptService';

// Mock data for demonstration
const recentConcepts = [
  { name: 'Dynamic Programming', slug: 'dynamic-programming', icon: '📊' },
  { name: 'Divide and Conquer', slug: 'divide-and-conquer', icon: '✂️' },
  { name: 'Greedy Algorithms', slug: 'greedy', icon: '🤑' },
];

const recommendedProblems = [
  { id: '1', title: 'Fibonacci Sequence', concept: 'Dynamic Programming', difficulty: 'easy', url: '/concepts/dynamic-programming' },
  { id: '2', title: 'Merge Sort Implementation', concept: 'Divide and Conquer', difficulty: 'medium', url: '/concepts/divide-and-conquer' },
  { id: '3', title: 'Activity Selection Problem', concept: 'Greedy Algorithms', difficulty: 'easy', url: '/concepts/greedy' },
];

export default function Home() {
  const [conceptsData, setConceptsData] = useState<ConceptData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showAllProgress, setShowAllProgress] = useState(false);
  const [showAllProblems, setShowAllProblems] = useState(false);

  // Fetch concepts data
  useEffect(() => {
    async function fetchData() {
      setIsLoading(true);
      try {
        const concepts = await getAllConcepts();
        const conceptsWithData = await Promise.all(
          concepts.map(async (concept) => {
            const data = await getConceptData(concept.slug);
            return data;
          })
        );
        setConceptsData(conceptsWithData.filter(Boolean) as ConceptData[]);
      } catch (error) {
        console.error('Error fetching concepts data:', error);
      } finally {
        setIsLoading(false);
      }
    }
    
    fetchData();
  }, []);

  // Calculate how many questions have been viewed for a concept
  // For demo purposes, we'll simulate viewed questions with random data
  const getViewedCount = (totalQuestions: number) => {
    // In a real app, this would use actual view data
    // For demo, just return a random number of viewed questions
    return Math.floor(Math.random() * (totalQuestions + 1));
  };

  // Determine which concepts to display
  const progressConcepts = showAllProgress 
    ? conceptsData 
    : conceptsData.slice(0, 4);

  // Get all problems from all concepts
  const allProblems = conceptsData.flatMap(concept => {
    // Find the slug that corresponds to this concept
    const conceptSlug = recentConcepts.find(c => c.name === concept.name)?.slug || 
                      concept.name.toLowerCase().replace(/\s+/g, '-');
    
    return concept.questions.map(question => ({
      id: question.id,
      title: question.title,
      difficulty: question.difficulty,
      concept: concept.name,
      url: `/concepts/${conceptSlug}`
    }));
  });

  // Determine which problems to display
  const displayedProblems = showAllProblems
    ? allProblems
    : recommendedProblems;

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Welcome Section */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <div className="flex items-center space-x-4">
            <div className="h-12 w-12 rounded-full bg-[#DE7356] flex items-center justify-center text-white">
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
              </svg>
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Welcome to Clarity</h2>
              <p className="text-gray-500">Your interactive platform for learning algorithms and data structures</p>
            </div>
          </div>

          <div className="mt-6">
            <h3 className="text-sm font-medium text-gray-700">Start your learning journey</h3>
            <div className="mt-2 grid grid-cols-1 gap-4 sm:grid-cols-3">
              {recentConcepts.map((concept) => (
                <Link
                  key={concept.slug}
                  href={`/concepts/${concept.slug}`}
                  className="relative rounded-lg border border-gray-300 bg-white px-6 py-5 shadow-sm flex items-center space-x-3 hover:border-gray-400 focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500"
                >
                  <div className="flex-shrink-0 text-2xl">{concept.icon}</div>
                  <div className="flex-1 min-w-0">
                    <span className="absolute inset-0" aria-hidden="true" />
                    <p className="text-sm font-medium text-gray-900">{concept.name}</p>
                    <p className="text-sm text-gray-500 truncate">Explore concept</p>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Middle Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Recommended Problems */}
          <div className="bg-white shadow-sm rounded-lg p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-medium text-gray-900">
                {showAllProblems ? 'All Problems' : 'Recommended Problems'}
              </h2>
              <button 
                onClick={() => setShowAllProblems(!showAllProblems)}
                className="text-sm font-medium text-[#DE7356] hover:text-[#B05A44]"
              >
                {showAllProblems ? 'Show Recommended' : 'View All'}
              </button>
            </div>
            <div className="mt-4">
              {isLoading && showAllProblems ? (
                <div className="text-center py-4 text-gray-500">Loading problems...</div>
              ) : (
                <ul className="divide-y divide-gray-200">
                  {displayedProblems.length === 0 ? (
                    <div className="text-center py-4 text-gray-500">No problems available</div>
                  ) : (
                    displayedProblems.map((problem, index) => (
                      <li key={`${problem.id}-${problem.concept}-${index}`} className="py-4">
                        <Link href={problem.url} className="block hover:bg-gray-50">
                          <div className="flex items-center space-x-4">
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium text-gray-900 truncate">{problem.title}</p>
                              <p className="text-sm text-gray-500 truncate">{problem.concept}</p>
                            </div>
                            <span
                              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                problem.difficulty === 'easy'
                                  ? 'bg-green-100 text-green-800'
                                  : problem.difficulty === 'medium'
                                  ? 'bg-yellow-100 text-yellow-800'
                                  : 'bg-red-100 text-red-800'
                              }`}
                            >
                              {problem.difficulty}
                            </span>
                          </div>
                        </Link>
                      </li>
                    ))
                  )}
                </ul>
              )}
            </div>
          </div>

          {/* Progress Tracking */}
          <div className="bg-white shadow-sm rounded-lg p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-medium text-gray-900">Your Progress</h2>
              <button 
                onClick={() => setShowAllProgress(!showAllProgress)}
                className="text-sm font-medium text-[#DE7356] hover:text-[#B05A44]"
              >
                {showAllProgress ? 'Show Less' : 'View All'}
              </button>
            </div>
            
            <div className="mt-4 space-y-4">
              {isLoading ? (
                <div className="text-center py-4 text-gray-500">Loading progress data...</div>
              ) : (
                <>
                  {progressConcepts.length === 0 ? (
                    <div className="text-center py-4 text-gray-500">No concepts available</div>
                  ) : (
                    progressConcepts.map((concept) => {
                      const totalQuestions = concept.questions.length;
                      const viewedCount = getViewedCount(totalQuestions);
                      const percentageViewed = totalQuestions > 0 
                        ? Math.round((viewedCount / totalQuestions) * 100)
                        : 0;
                        
                      return (
                        <div key={`progress-${concept.name}-${concept.questions.length}`}>
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-medium text-gray-700">{concept.name}</p>
                            <p className="text-sm font-medium text-gray-700">{viewedCount}/{totalQuestions} viewed</p>
                          </div>
                          <div className="mt-2 w-full bg-gray-200 rounded-full h-2.5">
                            <div 
                              className="bg-[#DE7356] h-2.5 rounded-full" 
                              style={{ width: `${percentageViewed}%` }}
                            ></div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Bottom section with chat */}
        <ChatBox conceptContext="general algorithms" />
      </div>
    </MainLayout>
  );
}
