'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import MainLayout from '@/components/layout/MainLayout';
import ConceptMaterial from '@/components/concept/ConceptMaterial';
import ConceptQuestions from '@/components/questions/ConceptQuestions';
import ChatBox from '@/components/chat/ChatBox';
import { getConceptData, ConceptData, Question } from '@/services/conceptService';

export default function ConceptClient({ conceptSlug }: { conceptSlug: string }) {
  const router = useRouter();
  
  const [conceptData, setConceptData] = useState<ConceptData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedQuestion, setSelectedQuestion] = useState<Question | null>(null);
  const [chatContext, setChatContext] = useState<string | undefined>(undefined);
  
  // Fetch concept data from "database"
  useEffect(() => {
    async function fetchData() {
      setIsLoading(true);
      try {
        const data = await getConceptData(conceptSlug);
        setConceptData(data);
        // Reset selected question when changing concepts
        setSelectedQuestion(null);
      } catch (error) {
        console.error('Error fetching concept data:', error);
      } finally {
        setIsLoading(false);
      }
    }
    
    if (conceptSlug) {
      fetchData();
    }
  }, [conceptSlug]);

  // Fix scrolling issue - make sure this only runs on the client
  useEffect(() => {
    // Use the browser's history API to replace the current state and scroll to top
    if (typeof window !== 'undefined') {
      window.history.scrollRestoration = 'manual';
      window.scrollTo(0, 0);
    }
  }, [conceptSlug]);

  // Set chat context when concept or question changes
  useEffect(() => {
    if (!conceptData) return;
    
    if (selectedQuestion) {
      // Create a more detailed context that includes the question
      setChatContext(`${conceptData.name}: ${selectedQuestion.title} (${selectedQuestion.difficulty})`);
      
      // Auto-scroll to chat when a question is selected
      setTimeout(() => {
        const chatElement = document.getElementById('chat-section');
        if (chatElement) {
          chatElement.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    } else {
      setChatContext(conceptData.name);
    }
  }, [conceptData, selectedQuestion]);

  const handleSelectQuestion = (question: Question) => {
    setSelectedQuestion(question);
  };

  // Show loading state while fetching data
  if (isLoading) {
    return (
      <MainLayout>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#DE7356]"></div>
        </div>
      </MainLayout>
    );
  }

  // Show error state if concept not found
  if (!conceptData) {
    return (
      <MainLayout>
        <div className="bg-white shadow-sm rounded-xl p-6">
          <h2 className="text-xl font-semibold text-gray-900">Concept Not Found</h2>
          <p className="mt-2 text-gray-500">The requested concept could not be found.</p>
          <button 
            onClick={() => router.push('/')}
            className="mt-4 px-4 py-2 bg-[#DE7356] text-white rounded hover:bg-[#C26B56]"
          >
            Return to Dashboard
          </button>
        </div>
      </MainLayout>
    );
  }

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Top section with concept materials and questions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ConceptMaterial 
            conceptName={conceptData.name} 
            materials={conceptData.materials}
            overview={conceptData.overview}
            keyCharacteristics={conceptData.keyCharacteristics}
            commonApplications={conceptData.commonApplications}
          />
          
          <ConceptQuestions 
            conceptName={conceptData.name}
            questions={conceptData.questions}
            onSelectQuestion={handleSelectQuestion}
          />
        </div>
        
        {/* Bottom section with chat */}
        <div id="chat-section" className="scroll-mt-8">
          <ChatBox 
            conceptContext={chatContext} 
            questionContext={selectedQuestion ? `I need help with the "${selectedQuestion.title}" problem.` : undefined}
          />
        </div>
        
        {/* Information about selected question, if any */}
        {selectedQuestion && (
          <div className="bg-[#DE7356]/10 p-4 rounded-lg border border-[#DE7356]/20 text-sm text-[#B05A44]">
            <p className="font-medium">Currently discussing:</p>
            <p>
              <span className="font-semibold">{selectedQuestion.title}</span> - 
              {' '}
              <span className={`inline-block px-2 py-0.5 rounded-full text-xs ${
                selectedQuestion.difficulty === 'easy' 
                  ? 'bg-green-100 text-green-800' 
                  : selectedQuestion.difficulty === 'medium'
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-red-100 text-red-800'
              }`}>
                {selectedQuestion.difficulty}
              </span>
              {' '}
              <button 
                onClick={() => setSelectedQuestion(null)}
                className="ml-2 text-[#DE7356] hover:text-[#B05A44] text-xs underline"
              >
                Clear
              </button>
            </p>
          </div>
        )}
      </div>
    </MainLayout>
  );
} 