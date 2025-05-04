import { Question } from '@/services/conceptService';

type QuestionViewerProps = {
  question: Question;
  onClose: () => void;
  onAsk: (question: Question) => void;
};

export default function QuestionViewer({ question, onClose, onAsk }: QuestionViewerProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        <div className="flex justify-between items-center p-4 border-b border-gray-200">
          <h3 className="text-lg font-medium">Question Details</h3>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500 focus:outline-none"
          >
            <span className="sr-only">Close</span>
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="p-6 overflow-auto flex-grow">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">{question.title}</h2>
              <div className="mt-2 flex items-center">
                <span 
                  className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    question.difficulty === 'easy'
                      ? 'bg-green-100 text-green-800'
                      : question.difficulty === 'medium'
                      ? 'bg-yellow-100 text-yellow-800'
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {question.difficulty}
                </span>
              </div>
            </div>
          </div>

          <div className="prose max-w-none bg-gray-50 p-4 rounded-lg">
            <p className="text-sm italic text-gray-500 mb-4">
              In a real implementation, this would display the full question details.
              For this demo, we're showing placeholder content.
            </p>
            
            <h3>Problem Statement</h3>
            <p>
              This is a placeholder for the {question.title} problem description. 
              In a real implementation, this would contain the full problem statement, 
              constraints, and examples for the {question.difficulty} level problem.
            </p>
            
            <h3>Example Input/Output</h3>
            <pre className="bg-gray-100 p-3 rounded">
              {`Input: [example input for ${question.title}]
Output: [example output for ${question.title}]`}
            </pre>
            
            <h3>Constraints</h3>
            <ul>
              <li>Constraint 1 for {question.title}</li>
              <li>Constraint 2 for {question.title}</li>
              <li>Time complexity should be optimized</li>
            </ul>
          </div>
        </div>
        
        <div className="px-4 py-3 bg-gray-50 text-right rounded-b-lg flex justify-between">
          <button
            type="button"
            onClick={() => onAsk(question)}
            className="py-2 px-4 bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Ask About This Problem
          </button>
          
          <button
            type="button"
            onClick={onClose}
            className="py-2 px-4 border border-gray-300 bg-white text-gray-700 rounded hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
} 