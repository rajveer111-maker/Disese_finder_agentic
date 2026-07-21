'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { 
  Brain, 
  ArrowRight, 
  Activity, 
  Cpu, 
  ShieldAlert, 
  Zap, 
  GitBranch,
  Search,
  CheckCircle2,
  Upload,
  FileText,
  Clock,
  Server
} from 'lucide-react';
import Card3D from '@/components/Card3D';

const models = [
  {
    key: 'neuroformer',
    name: 'Neuroformer Classifier',
    icon: Brain,
    desc: "Long-range oscillatory transformer tracking temporal decline signatures.",
    target: "Alzheimer's & FTD",
    metrics: "94.2% Acc | 19 Channels",
    glow: "rgba(139, 92, 246, 0.2)"
  },
  {
    key: 'eeg_pd',
    name: "Parkinson's Detector",
    icon: Activity,
    desc: "Oscillatory spectrum router searching for resting-state basal ganglia anomalies.",
    target: "Parkinson's Disease",
    metrics: "92.8% Acc | 22 Channels",
    glow: "rgba(20, 184, 166, 0.2)"
  },
  {
    key: 'nhrn_pd',
    name: "NHRN Parkinson's Net",
    icon: Activity,
    desc: "10-level neuromorphic resonance network capturing high-density cortical oscillations.",
    target: "Parkinson's Disease",
    metrics: "94.8% Acc | 40 Channels",
    glow: "rgba(52, 211, 153, 0.2)"
  },
  {
    key: 'bci2a',
    name: 'BCI2A Motor Decoder',
    icon: Zap,
    desc: "Decodes motor cortical imagery commands (Left/Right Hand, Foot, Tongue).",
    target: "Motor Imagery BCI",
    metrics: "91.5% Acc | 22 Channels",
    glow: "rgba(244, 63, 94, 0.2)"
  },
  {
    key: 'tumor_mri',
    name: 'MRI Morphology Net',
    icon: Search,
    desc: "Converts signals into spatial grids, running transfer-learning for tumor screens.",
    target: "Brain Tumor (MRI)",
    metrics: "95.6% Acc | Spatial CNN",
    glow: "rgba(56, 189, 248, 0.2)"
  },
  {
    key: 'spectra_sz',
    name: 'SPECTRA Routing Net',
    icon: GitBranch,
    desc: "Multi-scale psychiatric routing network detecting complex cognitive states.",
    target: "Schizophrenia (SZ)",
    metrics: "90.4% Acc | 19 Channels",
    glow: "rgba(236, 72, 153, 0.2)"
  },
  {
    key: 'cmo_core',
    name: 'Virtual CMO Agent',
    icon: ShieldAlert,
    desc: "Consensus synthesis routing engine making final diagnostics decisions.",
    target: "Ensemble Consensus",
    metrics: "< 1s Routing | Multi-modal",
    glow: "rgba(251, 191, 36, 0.2)"
  }
];

const steps = [
  {
    title: "1. Data Ingestion",
    desc: "Accepts raw signals (EDF, BDF, NPY, CSV, TXT) and normalizes into 1D sequences and 2D spatial matrices.",
    icon: Upload,
    color: "hsl(265 89% 66%)"
  },
  {
    title: "2. Ensemble Routing",
    desc: "Routes data concurrently to all clinical neural nets. Pre-routing agent estimates data validity and channel quality.",
    icon: GitBranch,
    color: "hsl(160 84% 55%)"
  },
  {
    title: "3. Virtual CMO Analysis",
    desc: "Chief Medical Officer agent synthesizes predictions, maps activations, and drafts final medical feedback reports.",
    icon: FileText,
    color: "hsl(280 95% 70%)"
  }
];

export default function Home() {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 md:px-8 pt-8 pb-24 md:pb-36 relative overflow-hidden">
      
      {/* 1. Hero Section */}
      <section className="min-h-[75vh] flex flex-col md:flex-row items-center justify-between" style={{ gap: '3rem', paddingTop: '4rem', paddingBottom: '4rem' }}>
        
        {/* Left Side: Copy */}
        <div className="flex-1 text-left z-20 flex flex-col items-start">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center rounded-full border border-primary-purple/35 bg-primary-purple/10 text-primary-purple text-xs font-bold uppercase tracking-wider"
            style={{ gap: '0.5rem', padding: '0.375rem 0.875rem', marginBottom: '1.5rem' }}
          >
            <Cpu className="w-3.5 h-3.5 animate-spin [animation-duration:8s]" />
            <span>Next-Gen Agentic Diagnosis</span>
          </motion.div>
          
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="font-display font-black text-4xl sm:text-5xl md:text-6xl tracking-tighter leading-[1.08] text-white"
          >
            Virtual <span className="gradient-text-primary">Chief Medical Officer</span> Diagnostic Lab
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-slate-400 text-base sm:text-lg md:text-xl font-light leading-relaxed max-w-xl"
            style={{ marginTop: '1.5rem', marginBottom: '2.5rem' }}
          >
            Ingest raw neuro-signals and scan matrices. Let our virtual CMO agent orchestrate, model, and route clinical inputs across specialized neural networks for holistic reporting.
          </motion.p>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="flex flex-row items-center flex-wrap justify-start w-full"
            style={{ gap: '1rem' }}
          >
            <Link 
              href="/cmo" 
              className="flex-shrink-0 relative group inline-flex items-center justify-center bg-gradient-to-r from-primary-purple to-pink-500 rounded-xl font-bold text-white shadow-[0_0_20px_rgba(139,92,246,0.3)] transition-all hover:shadow-[0_0_35px_rgba(139,92,246,0.6)] hover:scale-[1.03] active:scale-95 whitespace-nowrap text-sm md:text-base"
              style={{ gap: '0.625rem', padding: '1rem 1.75rem' }}
            >
              <span>Intake Raw Data</span>
              <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
            </Link>
            
            <Link 
              href="/performance" 
              className="flex-shrink-0 inline-flex items-center justify-center bg-white/5 border border-white/10 rounded-xl font-bold text-slate-200 transition-all hover:bg-white/10 hover:border-white/20 hover:scale-[1.03] active:scale-95 whitespace-nowrap text-sm md:text-base"
              style={{ gap: '0.625rem', padding: '1rem 1.75rem' }}
            >
              <span>Explore Benchmarks</span>
            </Link>
          </motion.div>
        </div>

        {/* Right Side: Glowing Neural Network Brain SVG */}
        <div className="flex-1 flex items-center justify-center relative w-full max-w-[480px] h-[380px] md:h-[480px] z-10">
          {/* Outer glow background */}
          <div className="absolute inset-0 bg-primary-purple/10 rounded-full filter blur-[80px] pointer-events-none z-0" />
          
          <motion.div
            animate={{ y: [0, -12, 0] }}
            transition={{ duration: 6, ease: "easeInOut", repeat: Infinity }}
            className="relative z-10 w-full h-full flex items-center justify-center"
          >
            <svg 
              viewBox="0 0 500 500" 
              className="w-full h-full max-w-[430px] drop-shadow-[0_0_25px_rgba(139,92,246,0.45)]"
            >
              {/* Brain outline/connections */}
              <motion.path 
                d="M 250,80 C 150,80 100,160 100,250 C 100,340 160,420 250,420 C 340,420 400,340 400,250 C 400,160 350,80 250,80 Z"
                fill="none" 
                stroke="url(#brainGrad)" 
                strokeWidth="1.5" 
                strokeDasharray="5 5"
                className="opacity-45"
              />
              
              {/* Central Neural Tracks */}
              <path d="M 250,80 L 250,420" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <path d="M 100,250 L 400,250" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <path d="M 130,160 L 370,340" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <path d="M 130,340 L 370,160" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />

              {/* Glowing gradients */}
              <defs>
                <linearGradient id="brainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="hsl(265 89% 66%)" />
                  <stop offset="100%" stopColor="hsl(160 84% 55%)" />
                </linearGradient>
                
                <radialGradient id="purpleGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="hsl(265 89% 66%)" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="hsl(265 89% 66%)" stopOpacity="0" />
                </radialGradient>
                <radialGradient id="cyanGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="hsl(160 84% 55%)" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="hsl(160 84% 55%)" stopOpacity="0" />
                </radialGradient>
              </defs>

              {/* Interactive Node Coordinates */}
              {/* BCI2A */}
              <g className="cursor-pointer">
                <circle cx="150" cy="180" r="16" fill="url(#purpleGlow)" className="animate-pulse" />
                <circle cx="150" cy="180" r="5" fill="#a78bfa" />
                <text x="110" y="210" fill="#cbd5e1" fontSize="10" fontWeight="bold" fontFamily="monospace">BCI2A</text>
              </g>

              {/* EEG PD */}
              <g className="cursor-pointer">
                <circle cx="350" cy="180" r="16" fill="url(#cyanGlow)" className="animate-pulse" />
                <circle cx="350" cy="180" r="5" fill="#14b8a6" />
                <text x="325" y="210" fill="#cbd5e1" fontSize="10" fontWeight="bold" fontFamily="monospace">EEG-PD</text>
              </g>

              {/* Neuroformer */}
              <g className="cursor-pointer">
                <circle cx="250" cy="120" r="20" fill="url(#purpleGlow)" className="animate-pulse" />
                <circle cx="250" cy="120" r="7" fill="#c084fc" />
                <text x="210" y="95" fill="#fff" fontSize="11" fontWeight="extrabold" fontFamily="monospace">NEUROFORMER</text>
              </g>

              {/* MRI */}
              <g className="cursor-pointer">
                <circle cx="250" cy="380" r="18" fill="url(#cyanGlow)" className="animate-pulse" />
                <circle cx="250" cy="380" r="6" fill="#2dd4bf" />
                <text x="220" y="412" fill="#cbd5e1" fontSize="10" fontWeight="bold" fontFamily="monospace">BRAIN MRI</text>
              </g>

              {/* SPECTRA */}
              <g className="cursor-pointer">
                <circle cx="120" cy="300" r="15" fill="url(#purpleGlow)" className="animate-pulse" />
                <circle cx="120" cy="300" r="4" fill="#ec4899" />
                <text x="95" y="330" fill="#cbd5e1" fontSize="10" fontWeight="bold" fontFamily="monospace">SPECTRA</text>
              </g>

              {/* CMO consensus core */}
              <g className="cursor-pointer">
                <circle cx="250" cy="250" r="28" fill="url(#purpleGlow)" className="animate-pulse" />
                <circle cx="250" cy="250" r="10" fill="#f43f5e" />
                <text x="220" y="254" fill="#fff" fontSize="12" fontWeight="black" fontFamily="monospace">CMO CORE</text>
              </g>

              {/* Signals pulsating between nodes */}
              <motion.circle
                r="3"
                fill="#f43f5e"
                animate={{
                  cx: [250, 150, 250],
                  cy: [120, 180, 250],
                }}
                transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              />
              <motion.circle
                r="3"
                fill="#14b8a6"
                animate={{
                  cx: [250, 350, 250],
                  cy: [380, 180, 250],
                }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              />
            </svg>
          </motion.div>
        </div>
      </section>

      {/* 2. Platform Core Capabilities Grid */}
      <section className="py-24 border-t border-white/5 relative z-20">
        <div className="text-left w-full">
          <h2 className="font-display font-extrabold text-3xl md:text-4xl tracking-tight text-white flex items-center gap-3">
            <Brain className="w-8 h-8 text-primary-purple" />
            <span>Active Clinical Neural Networks</span>
          </h2>
          <p 
            style={{ marginBottom: '2.5rem' }}
            className="mt-4 text-slate-400 font-light text-sm md:text-base leading-relaxed max-w-2xl"
          >
            Each routing node operates a custom-weighted sequence, temporal, or spatial morphology model.
          </p>
        </div>

        {/* Stable Aligned Grid Layout to avoid parallax clipping and layout misalignment */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 md:gap-10 mt-6 md:mt-8">
          {models.map((model) => (
            <Card3D 
              key={model.key} 
              glowColor={model.glow} 
              className="p-6 md:p-8 flex flex-col justify-between min-h-[260px] bg-black/20"
            >
              <div className="flex flex-col gap-4">
                <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                  <model.icon className="w-6 h-6 text-primary-purple" />
                </div>
                <div className="px-2">
                  <h3 className="font-display font-bold text-lg text-white">{model.name}</h3>
                  <p className="text-xs text-primary-purple font-semibold mt-1 uppercase tracking-wider">{model.target}</p>
                </div>
                <p className="text-sm text-slate-400 font-light leading-relaxed px-2">{model.desc}</p>
              </div>
              
              <div className="mt-6 border-t border-white/5 pt-4 text-[10px] font-mono text-slate-500 flex justify-between uppercase tracking-wider">
                <span>SYSTEM TARGETS</span>
                <span className="font-bold text-slate-400">{model.metrics}</span>
              </div>
            </Card3D>
          ))}
        </div>
      </section>

      {/* Spacer between active networks grid and routing logic */}
      <div className="h-20 md:h-28" />

      {/* 3. CMO Agent Pipeline Section */}
      <section className="py-24 border-t border-white/5 relative z-20">
        <div className="text-left w-full">
          <h2 className="font-display font-extrabold text-3xl md:text-4xl tracking-tight text-white flex items-center gap-3">
            <ShieldAlert className="w-8 h-8 text-primary-purple animate-pulse" />
            <span>Virtual CMO Routing Logic</span>
          </h2>
          <p 
            style={{ marginBottom: '2.5rem' }}
            className="mt-4 text-slate-400 font-light text-sm md:text-base leading-relaxed max-w-2xl"
          >
            Rather than feeding raw data into isolated algorithms, the Agentic pipeline routes incoming metrics contextually across multi-modal neural systems.
          </p>
        </div>

        <div className="max-w-5xl mx-auto glass-panel-heavy relative overflow-hidden bg-black/45" style={{ padding: '3rem', marginTop: '2rem' }}>
          <div className="absolute top-0 right-0 w-[300px] h-[300px] bg-primary-purple/5 blur-[80px] -z-10" />

          <div className="grid grid-cols-1 md:grid-cols-3" style={{ gap: '2rem' }}>
            {steps.map((step, idx) => (
              <div key={idx} className="flex flex-col rounded-xl bg-white/[0.02] border border-white/5 hover:border-white/10 transition-colors" style={{ padding: '2rem', gap: '0.75rem' }}>
                <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-white/5 border border-white/10" style={{ marginBottom: '0.5rem', color: step.color }}>
                  <step.icon className="w-5 h-5" />
                </div>
                <span className="font-display font-extrabold text-primary-purple text-base">{step.title}</span>
                <p className="text-xs text-slate-400 font-light leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>

          <div className="flex justify-center" style={{ marginTop: '3rem' }}>
            <Link 
              href="/cmo" 
              className="inline-flex items-center border border-primary-purple/30 bg-primary-purple/10 rounded-xl text-xs font-bold uppercase tracking-widest text-primary-purple hover:text-white hover:bg-primary-purple/20 transition-all hover:scale-[1.03] active:scale-95 shadow-[0_0_15px_rgba(139,92,246,0.1)] hover:shadow-[0_0_25px_rgba(139,92,246,0.3)]"
              style={{ padding: '0.75rem 1.5rem', gap: '0.5rem' }}
            >
              <span>Access Clinical Ingest Terminal</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </section>

      {/* Spacer between routing logic and stats */}
      <div className="h-20 md:h-28" />

      {/* 4. Live Stats Overview Banner */}
      <section className="py-24 mb-24 md:mb-32 border-t border-white/5 relative z-20">
        <div className="text-left w-full">
          <h2 className="font-display font-extrabold text-3xl md:text-4xl tracking-tight text-white flex items-center gap-3">
            <Activity className="w-8 h-8 text-accent-cyan animate-pulse" />
            <span>Real-Time Engine Vitals</span>
          </h2>
          <p 
            style={{ marginBottom: '2.5rem' }}
            className="mt-4 text-slate-400 font-light text-sm md:text-base leading-relaxed max-w-2xl"
          >
            Operational consensus statistics, system latency metrics, and neural ensemble status feeds.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 mt-6 md:mt-8">
          <Card3D glowColor="rgba(139, 92, 246, 0.2)" className="p-6 md:p-8 flex flex-col justify-between min-h-[160px] bg-black/20 text-center">
            <div className="flex justify-center mb-4">
              <div className="w-10 h-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-primary-purple">
                <CheckCircle2 className="w-5 h-5" />
              </div>
            </div>
            <div className="px-2">
              <span className="block text-3xl font-display font-black gradient-text-primary">94.6%</span>
              <span className="block text-[10px] text-slate-500 uppercase tracking-widest mt-2 font-mono">Consensus Accuracy</span>
            </div>
          </Card3D>
          <Card3D glowColor="rgba(20, 184, 166, 0.2)" className="p-6 md:p-8 flex flex-col justify-between min-h-[160px] bg-black/20 text-center">
            <div className="flex justify-center mb-4">
              <div className="w-10 h-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-accent-cyan">
                <Clock className="w-5 h-5" />
              </div>
            </div>
            <div className="px-2">
              <span className="block text-3xl font-display font-black text-accent-cyan">&lt; 240ms</span>
              <span className="block text-[10px] text-slate-500 uppercase tracking-widest mt-2 font-mono">Ingestion Latency</span>
            </div>
          </Card3D>
          <Card3D glowColor="rgba(236, 72, 153, 0.2)" className="p-6 md:p-8 flex flex-col justify-between min-h-[160px] bg-black/20 text-center">
            <div className="flex justify-center mb-4">
              <div className="w-10 h-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-pink-500">
                <Cpu className="w-5 h-5" />
              </div>
            </div>
            <div className="px-2">
              <span className="block text-3xl font-display font-black text-pink-400">6</span>
              <span className="block text-[10px] text-slate-500 uppercase tracking-widest mt-2 font-mono">Neural Ensembles</span>
            </div>
          </Card3D>
          <Card3D glowColor="rgba(52, 211, 153, 0.2)" className="p-6 md:p-8 flex flex-col justify-between min-h-[160px] bg-black/20 text-center">
            <div className="flex justify-center mb-4">
              <div className="w-10 h-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-emerald-400">
                <Server className="w-5 h-5" />
              </div>
            </div>
            <div className="px-2">
              <span className="block text-3xl font-display font-black text-emerald-400 flex items-center justify-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                <span>Online</span>
              </span>
              <span className="block text-[10px] text-slate-500 uppercase tracking-widest mt-2 font-mono">System Node Status</span>
            </div>
          </Card3D>
        </div>
      </section>

      {/* Spacer before footer */}
      <div className="h-16 md:h-24" />
    </div>
  );
}
