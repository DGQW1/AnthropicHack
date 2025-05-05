import { useState } from 'react';
import PDFViewer from './PDFViewer';
import { MaterialFile } from '@/services/conceptService';

type ConceptMaterialProps = {
  conceptName: string;
  materials: MaterialFile[];
  overview?: string;
  keyCharacteristics?: string[];
  commonApplications?: string[];
};

export default function ConceptMaterial({ 
  conceptName, 
  materials,
  overview,
  keyCharacteristics,
  commonApplications
}: ConceptMaterialProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'materials'>('overview');
  const [selectedFile, setSelectedFile] = useState<MaterialFile | null>(null);

  // Handle PDF file opening
  const handleFileClick = (e: React.MouseEvent<HTMLAnchorElement>, file: MaterialFile) => {
    e.preventDefault();
    setSelectedFile(file);
  };

  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden h-full flex flex-col">
      <div className="p-4 bg-[#DE7356] text-white">
        <h2 className="text-xl font-semibold">{conceptName}</h2>
      </div>
      
      <div className="border-b border-gray-200">
        <nav className="flex">
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'overview'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'materials'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('materials')}
          >
            Materials
          </button>
        </nav>
      </div>
      
      <div className="p-4 flex-1 overflow-y-auto">
        {activeTab === 'overview' ? (
          <div className="prose max-w-none">
            {/* Description section - removed */}
            
            {/* Overview section - simplified to a single paragraph */}
            {overview && (
              <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm mb-6">
                <h3 className="text-lg font-medium text-gray-900 mb-3">Concept Overview</h3>
                <div 
                  className="text-gray-700" 
                  dangerouslySetInnerHTML={{ __html: overview }} 
                />
              </div>
            )}
            
            {/* Key Characteristics section */}
            {keyCharacteristics && keyCharacteristics.length > 0 && (
              <div className="mt-6 bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                <h3 className="text-lg font-medium text-gray-900 mb-3">Key Characteristics</h3>
                <ul className="space-y-2 text-gray-700">
                  {keyCharacteristics.map((item, index) => (
                    <li key={index} className="flex items-start">
                      <span className="mr-2 mt-1 text-blue-500">•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* Common Applications section */}
            {commonApplications && commonApplications.length > 0 && (
              <div className="mt-6 bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                <h3 className="text-lg font-medium text-gray-900 mb-3">Common Applications</h3>
                <ul className="space-y-2 text-gray-700">
                  {commonApplications.map((item, index) => (
                    <li key={index} className="flex items-start">
                      <span className="mr-2 mt-1 text-blue-500">•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <h3 className="font-medium text-gray-700">Course Materials</h3>
            
            {materials.length === 0 ? (
              <p className="text-gray-500 italic">No materials available for this concept yet.</p>
            ) : (
              <ul className="divide-y divide-gray-200">
                {materials.map((file) => (
                  <li key={file.id} className="py-3">
                    <a 
                      href={file.url} 
                      className="flex items-center hover:bg-gray-50 p-2 rounded"
                      onClick={(e) => handleFileClick(e, file)}
                    >
                      <svg className="h-6 w-6 text-red-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9 2a2 2 0 00-2 2v8a2 2 0 002 2h6a2 2 0 002-2V6.414A2 2 0 0016.414 5L14 2.586A2 2 0 0012.586 2H9z" />
                        <path d="M3 8a2 2 0 012-2h2a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
                      </svg>
                      <div>
                        <p className="text-sm font-medium text-gray-900">{file.name}</p>
                        <p className="text-xs text-gray-500 capitalize">{file.type} (PDF)</p>
                      </div>
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* PDF Viewer Modal */}
      {selectedFile && (
        <PDFViewer 
          url={selectedFile.url} 
          onClose={() => setSelectedFile(null)} 
        />
      )}
    </div>
  );
} 