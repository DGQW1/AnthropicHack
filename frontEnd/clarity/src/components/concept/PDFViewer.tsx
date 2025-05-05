import { useState } from 'react';

type PDFViewerProps = {
  url: string;
  onClose: () => void;
};

export default function PDFViewer({ url, onClose }: PDFViewerProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        <div className="flex justify-between items-center p-4 border-b border-gray-200">
          <h3 className="text-lg font-medium">PDF Viewer</h3>
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
          <div className="w-full h-[70vh]">
            <object 
              data={url} 
              type="application/pdf" 
              width="100%" 
              height="100%"
              className="border rounded"
            >
              <p>Your browser does not support embedded PDFs. 
                <a href={url} target="_blank" rel="noopener noreferrer">Click here to download the PDF</a>
              </p>
            </object>
          </div>
        </div>
        
        <div className="px-4 py-3 bg-gray-50 text-right rounded-b-lg">
          <button
            type="button"
            onClick={onClose}
            className="py-2 px-4 bg-[#DE7356] text-white rounded hover:bg-[#C26B56] focus:outline-none focus:ring-2 focus:ring-[#DE7356]"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
} 