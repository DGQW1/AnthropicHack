import { useState } from 'react';
import { Question, QuestionDifficulty } from '@/services/conceptService';
import QuestionViewer from './QuestionViewer';
import PDFViewer from '../concept/PDFViewer';

type ConceptQuestionsProps = {
  conceptName: string;
  questions: Question[];
  onSelectQuestion: (question: Question) => void;
};

export default function ConceptQuestions({
  conceptName,
  questions,
  onSelectQuestion
}: ConceptQuestionsProps) {
  const [filter, setFilter] = useState<'all' | QuestionDifficulty>('all');
  const [viewingQuestion, setViewingQuestion] = useState<Question | null>(null);
  const [viewingPdf, setViewingPdf] = useState<Question | null>(null);
  const [viewedQuestions, setViewedQuestions] = useState<Record<string, boolean>>({});
  
  const filteredQuestions = filter === 'all' 
    ? questions 
    : questions.filter(q => q.difficulty === filter);

  // Handle viewing a question
  const handleViewQuestion = (question: Question) => {
    setViewingQuestion(question);
    // Mark question as viewed
    setViewedQuestions(prev => ({
      ...prev,
      [question.id]: true
    }));
  };

  // Handle viewing the PDF directly
  const handleViewPdf = (question: Question) => {
    setViewingPdf(question);
    // Mark question as viewed
    setViewedQuestions(prev => ({
      ...prev,
      [question.id]: true
    }));
  };

  // Check if a question has been viewed
  const isQuestionViewed = (questionId: string) => {
    return viewedQuestions[questionId] || false;
  };

  // Handle asking about a question
  const handleAskQuestion = (question: Question) => {
    onSelectQuestion(question);
    // Close the viewer if it's open
    setViewingQuestion(null);
    setViewingPdf(null);
  };

  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden h-full flex flex-col">
      <div className="p-4 bg-[#DE7356] text-white">
        <h2 className="text-xl font-semibold">Practice Problems</h2>
        <p className="text-sm text-white opacity-80">{conceptName}</p>
      </div>

      <div className="border-b border-gray-200 p-3 flex justify-between items-center">
        <div className="text-sm text-gray-500">
          {filteredQuestions.length} problems available
        </div>
        <div className="flex space-x-1">
          <button 
            onClick={() => setFilter('all')}
            className={`px-2 py-1 text-xs rounded ${
              filter === 'all' 
                ? 'bg-gray-200 text-gray-800' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            All
          </button>
          <button 
            onClick={() => setFilter('easy')}
            className={`px-2 py-1 text-xs rounded ${
              filter === 'easy' 
                ? 'bg-green-200 text-green-800' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            Easy
          </button>
          <button 
            onClick={() => setFilter('medium')}
            className={`px-2 py-1 text-xs rounded ${
              filter === 'medium' 
                ? 'bg-yellow-200 text-yellow-800' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            Medium
          </button>
          <button 
            onClick={() => setFilter('hard')}
            className={`px-2 py-1 text-xs rounded ${
              filter === 'hard' 
                ? 'bg-red-200 text-red-800' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            Hard
          </button>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        {filteredQuestions.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            No questions available with the selected filter.
          </div>
        ) : (
          <ul className="divide-y divide-gray-200">
            {filteredQuestions.map((question) => (
              <li key={question.id} className="hover:bg-gray-50">
                <div className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center flex-1">
                      <div className={`w-2 h-2 rounded-full mr-3 ${
                        'bg-gray-300'
                      }`} />
                      <div>
                        <h3 className="font-medium text-gray-900">{question.title}</h3>
                        <div className="flex items-center mt-1">
                          <span 
                            className={`text-xs px-2 py-1 rounded ${
                              question.difficulty === 'easy' 
                                ? 'bg-green-100 text-green-800' 
                                : question.difficulty === 'medium'
                                  ? 'bg-yellow-100 text-yellow-800'
                                  : 'bg-red-100 text-red-800'
                            }`}
                          >
                            {question.difficulty}
                          </span>
                          {isQuestionViewed(question.id) && (
                            <span className="ml-2 text-xs text-blue-600 flex items-center">
                              <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                              </svg>
                              Viewed
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex space-x-2 ml-4">
                      <button
                        onClick={() => handleViewPdf(question)}
                        className={`px-3 py-1 text-xs border rounded-md flex items-center ${
                          isQuestionViewed(question.id) 
                            ? 'border-[#DE7356]/30 text-[#DE7356] bg-[#DE7356]/5 hover:bg-[#DE7356]/10' 
                            : 'border-gray-300 hover:bg-gray-50'
                        }`}
                      >
                        <svg className={`h-3 w-3 mr-1 ${isQuestionViewed(question.id) ? 'text-blue-500' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                        {isQuestionViewed(question.id) ? 'View Again' : 'View'}
                      </button>
                      <button
                        onClick={() => handleAskQuestion(question)}
                        className="px-3 py-1 text-xs bg-[#DE7356] text-white rounded-md hover:bg-[#C26B56] flex items-center"
                      >
                        <svg className="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                        Ask
                      </button>
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Question Info Viewer Modal */}
      {viewingQuestion && (
        <QuestionViewer 
          question={viewingQuestion}
          onClose={() => setViewingQuestion(null)}
          onAsk={handleAskQuestion}
        />
      )}

      {/* PDF Viewer Modal */}
      {viewingPdf && (
        <PDFViewer 
          url={viewingPdf.url} 
          onClose={() => setViewingPdf(null)} 
        />
      )}
    </div>
  );
} 