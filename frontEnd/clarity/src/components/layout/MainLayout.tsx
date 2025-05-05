import { ReactNode } from 'react';
import Sidebar from './Sidebar';

type MainLayoutProps = {
  children: ReactNode;
};

export default function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="flex min-h-screen bg-[#FAFAFA]">
      <Sidebar />
      
      <div className="flex-1 ml-64">
        <header className="bg-[#F0F0F0] border-b border-[#E0E0E0] z-10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16 items-center">
              <div className="flex items-center">
                {/* <img src="/clarity-logo.png" alt="Clarity Logo" className="h-8 w-auto mr-3" /> */}
                <h1 className="text-xl font-semibold text-[#333333] font-sans tracking-tight">CSCI270</h1>
              </div>
              
              <div className="flex items-center">
                <button
                  type="button"
                  className="p-1 rounded-full text-[#666666] hover:text-[#333333] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#DE7356]"
                >
                  <span className="sr-only">View notifications</span>
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                  </svg>
                </button>
                
                <div className="ml-3 relative">
                  <div>
                    <button
                      type="button"
                      className="max-w-xs flex items-center text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#DE7356]"
                    >
                      <span className="sr-only">Open user menu</span>
                      <div className="h-8 w-8 rounded-full bg-[#DE7356] flex items-center justify-center text-white">
                        U
                      </div>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </header>
        
        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {children}
        </main>
      </div>
    </div>
  );
} 