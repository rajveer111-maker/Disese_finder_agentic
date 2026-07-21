'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Menu, X, Activity, ShieldAlert, Cpu } from 'lucide-react';

const navItems = [
  { name: 'Home', path: '/', icon: Brain },
  { name: 'Virtual CMO', path: '/cmo', icon: ShieldAlert },
  { name: 'Performance', path: '/performance', icon: Activity },
  { name: 'System Info', path: '/system', icon: Cpu },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex justify-center border-b border-white/5 bg-[#09090e]/80 backdrop-blur-md" style={{ padding: '1rem 1.5rem' }}>
      <div className="w-full max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center group" style={{ gap: '0.75rem' }}>
          <img 
            src="/logo.png" 
            alt="Virtual CMO Logo" 
            className="w-11 h-11 object-contain rounded-xl shadow-[0_0_15px_rgba(139,92,246,0.4)] transition-transform duration-300 group-hover:scale-105"
          />
          <div className="flex flex-col">
            <span className="font-display font-extrabold text-base md:text-xl tracking-tight text-white">
              Virtual<span className="text-primary-purple font-light">CMO</span>
            </span>
            <span className="text-[10px] text-slate-400 uppercase tracking-widest hidden sm:inline" style={{ marginTop: '-0.125rem' }}>
              Agentic Disease Finder
            </span>
          </div>
        </Link>

        {/* Desktop Nav Items */}
        <div className="hidden md:flex items-center" style={{ gap: '0.5rem' }}>
          {navItems.map((item) => {
            const isActive = pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                href={item.path}
                className="relative rounded-lg text-sm font-semibold tracking-wide transition-colors duration-300 flex items-center text-slate-300 hover:text-white group"
                style={{ padding: '0.5rem 1rem', gap: '0.5rem' }}
              >
                {isActive && (
                  <motion.div
                    layoutId="activePill"
                    className="absolute inset-0 bg-gradient-to-r from-primary-purple/15 to-accent-cyan/15 rounded-lg border border-primary-purple/30 shadow-[0_0_15px_rgba(139,92,246,0.15)]"
                    transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                  />
                )}
                <Icon className={`w-4 h-4 ${isActive ? 'text-primary-purple' : 'text-slate-400 group-hover:text-slate-200'}`} />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </div>

        {/* Mobile Hamburger Toggle */}
        <div className="md:hidden">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition-colors"
            style={{ padding: '0.5rem' }}
          >
            {mobileMenuOpen ? <X className="w-6 h-6 text-white" /> : <Menu className="w-6 h-6 text-white" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="absolute top-[80px] left-4 right-4 md:hidden z-40"
          >
            <div className="glass-panel-heavy p-6 flex flex-col gap-4 bg-black/85">
              {navItems.map((item) => {
                const isActive = pathname === item.path;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.path}
                    href={item.path}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`flex items-center gap-4 p-3 rounded-xl border transition-all ${
                      isActive
                        ? 'border-primary-purple/40 bg-primary-purple/10 text-white font-bold shadow-[0_0_15px_rgba(139,92,246,0.15)]'
                        : 'border-white/5 bg-white/5 text-slate-300 hover:bg-white/10'
                    }`}
                  >
                    <div className={`p-2 rounded-lg ${isActive ? 'bg-primary-purple text-white' : 'bg-white/5 text-slate-450'}`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className="text-base font-semibold">{item.name}</span>
                  </Link>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
