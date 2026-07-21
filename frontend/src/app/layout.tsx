import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import MouseGlow from "@/components/MouseGlow";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Agentic Disease Finder | Virtual CMO Diagnostics Suite",
  description: "Advanced multi-modal deep learning platform routing EEG signals and brain MRI scans across neural ensembles for automated clinical diagnostics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${spaceGrotesk.variable} h-full w-full antialiased`}>
      <body className="min-h-full w-full bg-[#09090e] text-slate-100 font-sans relative flex flex-col">
        {/* Background Ambient Radial Glows */}
        <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
          <div className="absolute top-0 left-0 w-[500px] h-[500px] rounded-full bg-primary-purple/10 blur-[150px] -translate-x-1/3 -translate-y-1/3" />
          <div className="absolute top-0 right-0 w-[500px] h-[500px] rounded-full bg-accent-cyan/8 blur-[150px] translate-x-1/3 -translate-y-1/3" />
          
          {/* Subtle grid lines background */}
          <div 
            className="absolute inset-0 opacity-[0.02]" 
            style={{
              backgroundImage: 'linear-gradient(to right, #ffffff 1px, transparent 1px), linear-gradient(to bottom, #ffffff 1px, transparent 1px)',
              backgroundSize: '40px 40px'
            }}
          />
        </div>
        
        {/* Cursor Glow */}
        <MouseGlow />
        
        {/* Navigation */}
        <Navbar />
        
        {/* Content */}
        <main className="flex-grow w-full z-20 relative pb-12 flex flex-col items-center" style={{ paddingTop: '8rem' }}>
          {children}
        </main>
        
        {/* Global Footer */}
        <footer className="z-20 border-t border-white/5 bg-black/35 backdrop-blur-md flex flex-col items-center justify-center w-full" style={{ padding: '2rem 1.5rem', gap: '1rem' }}>
          <div className="flex flex-col md:flex-row items-center justify-between w-full max-w-7xl mx-auto" style={{ gap: '1.5rem' }}>
            <div className="flex items-center" style={{ gap: '0.75rem' }}>
              <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse" />
              <span className="text-xs text-slate-400 font-mono font-semibold uppercase tracking-wider">
                System Online — Agentic Diagnostics v2.0
              </span>
            </div>
            
            <div className="flex items-center text-xs font-mono font-semibold text-slate-500 uppercase tracking-widest" style={{ gap: '1.5rem' }}>
              <a href="#" className="hover:text-primary-purple transition-colors">Documentation</a>
              <a href="#" className="hover:text-primary-purple transition-colors">API Status</a>
              <a href="#" className="hover:text-primary-purple transition-colors">Privacy Policy</a>
              <a href="#" className="hover:text-primary-purple transition-colors">Terms of Use</a>
            </div>
          </div>
          <div className="w-full max-w-7xl mx-auto border-t border-white/5" style={{ marginTop: '0.5rem', paddingTop: '1.5rem' }}>
            <p className="text-center text-[10px] text-slate-600 font-mono uppercase tracking-widest">
              © {new Date().getFullYear()} Agentic Disease Finder. For research and prototype demonstration only. Not for clinical use.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
