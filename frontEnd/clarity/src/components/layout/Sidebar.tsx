import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { getAllConcepts } from '@/services/conceptService';

type Concept = {
  slug: string;
  name: string;
};

export default function Sidebar() {
  const pathname = usePathname();
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchConcepts() {
      try {
        const conceptsData = await getAllConcepts();
        setConcepts(conceptsData);
      } catch (error) {
        console.error('Error fetching concepts:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchConcepts();
  }, []);

  return (
    <div className="h-screen bg-[#F8F8F8] text-[#333333] w-64 fixed left-0 top-0 overflow-y-auto border-r border-[#EBEBEB]">
      <div className="h-16 flex items-center px-4 border-b border-[#E0E0E0] bg-[#F0F0F0]">
        <div className="flex items-center">
          <img src="/clarity-logo.png" alt="Clarity Logo" className="h-13 w-auto mr-2" />
          <h1 className="text-2xl font-semibold text-[#333333] font-sans">Clarity</h1>
        </div>
      </div>
      
      <nav className="mt-6">
        <div className="px-4 mb-3">
          <h2 className="text-[#777777] uppercase text-xs font-medium tracking-wider">DASHBOARD</h2>
        </div>
        <Link href="/" className={`flex items-center px-4 py-2 ${pathname === '/' ? 'bg-[#EFEFEF] text-[#333333]' : 'hover:bg-[#F2F2F2] text-[#444444]'}`}>
          <svg className="h-5 w-5 mr-3 text-[#666666]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
          </svg>
          Dashboard
        </Link>
        
        <div className="px-4 mb-3 mt-6">
          <h2 className="text-[#777777] uppercase text-xs font-medium tracking-wider">ALGORITHM CONCEPTS</h2>
        </div>
        
        {isLoading ? (
          <div className="px-4 py-2 text-[#999999]">Loading concepts...</div>
        ) : (
          concepts.map((concept) => (
            <Link 
              key={concept.slug} 
              href={`/concepts/${concept.slug}`}
              className={`flex items-center px-4 py-2 ${pathname === `/concepts/${concept.slug}` ? 'bg-[#EFEFEF] text-[#333333]' : 'hover:bg-[#F2F2F2] text-[#444444]'}`}
            >
              <svg className="h-5 w-5 mr-3 text-[#666666]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              {concept.name}
            </Link>
          ))
        )}
      </nav>
    </div>
  );
} 