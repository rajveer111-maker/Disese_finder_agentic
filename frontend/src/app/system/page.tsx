'use client';

import { useState, useEffect } from 'react';

import { 
  Cpu, 
  Database,
  Server, 
  Activity, 
  TrendingUp, 
  Layers, 
  CheckCircle,
  Network
} from 'lucide-react';

const systemNodes = [
  {
    phase: '1. Ingestion Interface',
    title: 'Data Ingress Gateway',
    desc: 'Receives clinical uploads (EDF, BDF, NPY, CSV, TXT) and verifies schema consistency.',
    tech: 'MNE-Python / NumPy File Reader'
  },
  {
    phase: '2. Preprocessor Node',
    title: 'Adaptive Filter & Reshape',
    desc: 'Applies 1-40Hz bandpass filter, performs Z-score scaling, and segments into 1D sequences/2D spatial matrices.',
    tech: 'EEGPreprocessor / SciPy Signal'
  },
  {
    phase: '3. Pre-Routing Agent',
    title: 'Metadata Analysis Agent',
    desc: 'Analyzes channel count, duration, and signal power to calculate compatibility index for each model.',
    tech: 'AgenticDecisionSystem Core'
  },
  {
    phase: '4. Model Ensembles',
    title: 'Parallel Neural Classifiers',
    desc: 'Routes preprocessed matrices to all online networks (NHRN-PD, Neuroformer, SPECTRA-SZ) in parallel.',
    tech: 'PyTorch / AWS Bedrock & SageMaker'
  },
  {
    phase: '5. CMO Synthesizer',
    title: 'Consensus Decision Core',
    desc: 'Synthesizes outputs, filters uncertain forecasts below threshold, and compiles the unified clinical report.',
    tech: 'Virtual Chief Medical Officer'
  }
];

const techStack = [
  { group: 'Interface Layer', items: ['React 19 / Next.js 16 (App Router)', 'Tailwind CSS v4 (Glassmorphism Custom)', 'Framer Motion (Physics Springs)', 'Lucide Icons'] },
  { group: 'Ingestion Layer', items: ['MNE-Python (EDF/BDF Ingress)', 'Pandas (CSV/TXT Processing)', 'NumPy (Vector Matrices)', 'SciPy (Signal Filtering)'] },
  { group: 'AI & Inference Engines', items: ['NHRN-PD (PyTorch — PD Detection, 88.5% Acc)', 'Neuroformer (PyTorch — AD/FTD, 82.0% Acc)', 'SPECTRA-SZ (PyTorch — SZ Detection, 85.0% Acc)', 'AWS Bedrock / SageMaker (Agentic Routing)'] }
];

export default function SystemPage() {
  const [vitals, setVitals] = useState({
    cpu: 0,
    vram: 0,
    temp: 0,
    latency: 0,
    gpuPower: 0,
    gpuClock: 0,
    gpuName: 'N/A',
    gpuTotalVram: 24,
    gpuDriver: 'N/A'
  });
  const [isApiOnline, setIsApiOnline] = useState(false);

  // Fetch live system vitals from API (measuring real node cross-latency)
  useEffect(() => {
    let isMounted = true;
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    const fetchVitals = async () => {
      const startTime = performance.now();
      try {
        const res = await fetch(`${apiUrl}/api/hardware`);
        const latency = performance.now() - startTime;
        
        if (res.ok) {
          const data = await res.json();
          if (isMounted) {
            setIsApiOnline(true);
            setVitals({
              cpu: Number((data.cpu || 0).toFixed(1)),
              vram: Number((data.gpu?.vram_used || 0).toFixed(2)),
              temp: Number((data.gpu?.temp || 0).toFixed(0)),
              latency: Number(latency.toFixed(1)),
              gpuPower: Number((data.gpu?.power || 0).toFixed(0)),
              gpuClock: Number((data.gpu?.clock || 0).toFixed(0)),
              gpuName: data.gpu?.name || 'N/A',
              gpuTotalVram: data.gpu?.vram_total > 0 ? Number(data.gpu.vram_total.toFixed(2)) : 24.0,
              gpuDriver: data.gpu?.driver || 'N/A'
            });
          }
        } else {
          if (isMounted) {
            setIsApiOnline(false);
          }
        }
      } catch (e) {
        if (isMounted) {
          setIsApiOnline(false);
        }
      }
    };

    // Run immediately on mount
    fetchVitals();

    const interval = setInterval(fetchVitals, 2000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="w-full max-w-7xl mx-auto px-4 md:px-8" style={{ paddingTop: '2.5rem', paddingBottom: '6rem' }}>
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4" style={{ marginBottom: '3.5rem' }}>
        <div>
          <h1 className="font-display font-black text-3xl sm:text-4xl tracking-tighter text-white flex items-center gap-3">
            <Cpu className="w-8 h-8 text-primary-purple" />
            <span>System Architecture & Node Monitor</span>
          </h1>
          <p className="text-slate-400 font-light mt-2 max-w-xl text-sm md:text-base">
            Monitor the live backend execution stack, signal processing node pipeline, and operational engine telemetry.
          </p>
        </div>
        
        {/* Status Badge */}
        <div className="self-start sm:self-auto">
          {isApiOnline ? (
            <div className="px-3 py-1.5 rounded-lg border border-emerald-500/25 bg-emerald-500/10 text-emerald-400 text-xs font-semibold flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>FastAPI Connected</span>
            </div>
          ) : (
            <div className="px-3 py-1.5 rounded-lg border border-rose-500/25 bg-rose-500/10 text-rose-400 text-xs font-semibold flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
              <span>Backend Offline</span>
            </div>
          )}
        </div>
      </div>

      {/* 1. Live Telemetry Vitals Panel */}
      <div className="grid grid-cols-2 lg:grid-cols-4" style={{ gap: '1rem', marginBottom: '2.5rem' }}>
        
        {/* CPU */}
        <div className="glass-panel flex flex-col justify-between" style={{ padding: '1.5rem' }}>
          <div className="flex justify-between items-start text-slate-500 font-mono text-xs">
            <span>CPU CORE LOAD</span>
            <Server className="w-4 h-4 text-primary-purple" />
          </div>
          <div className="mt-4">
            <span className="text-3xl font-display font-extrabold text-white">{vitals.cpu}%</span>
            {/* simple bar */}
            <div className="w-full bg-white/5 border border-white/10 h-1.5 rounded-full overflow-hidden mt-3">
              <div 
                style={{ width: `${vitals.cpu}%` }} 
                className="h-full bg-primary-purple transition-all duration-1000" 
              />
            </div>
          </div>
        </div>

        {/* VRAM */}
        <div className="glass-panel flex flex-col justify-between" style={{ padding: '1.5rem' }}>
          <div className="flex justify-between items-start text-slate-500 font-mono text-xs">
            <span>DIRECTML VRAM ALLOC</span>
            <Database className="w-4 h-4 text-accent-cyan" />
          </div>
          <div className="mt-4">
            <span className="text-3xl font-display font-extrabold text-white">{vitals.vram} GB</span>
            <div className="w-full bg-white/5 border border-white/10 h-1.5 rounded-full overflow-hidden mt-3">
              <div 
                style={{ width: `${Math.min(100, (vitals.vram / vitals.gpuTotalVram) * 100)}%` }} 
                className="h-full bg-accent-cyan transition-all duration-1000" 
              />
            </div>
          </div>
        </div>

        {/* Temp */}
        <div className="glass-panel flex flex-col justify-between" style={{ padding: '1.5rem' }}>
          <div className="flex justify-between items-start text-slate-500 font-mono text-xs">
            <span>ENGINE TEMPERATURE</span>
            <TrendingUp className="w-4 h-4 text-rose-500" />
          </div>
          <div className="mt-4">
            <span className="text-3xl font-display font-extrabold text-white">{vitals.temp} °C</span>
            <div className="w-full bg-white/5 border border-white/10 h-1.5 rounded-full overflow-hidden mt-3">
              <div 
                style={{ width: `${((vitals.temp - 40) / 40) * 100}%` }} 
                className="h-full bg-rose-500 transition-all duration-1000" 
              />
            </div>
          </div>
        </div>

        {/* API Response */}
        <div className="glass-panel flex flex-col justify-between" style={{ padding: '1.5rem' }}>
          <div className="flex justify-between items-start text-slate-500 font-mono text-xs">
            <span>NODE CROSS-LATENCY</span>
            <Activity className="w-4 h-4 text-amber-500" />
          </div>
          <div className="mt-4">
            <span className="text-3xl font-display font-extrabold text-white">{vitals.latency} ms</span>
            <div className="w-full bg-white/5 border border-white/10 h-1.5 rounded-full overflow-hidden mt-3">
              <div 
                style={{ width: `${(vitals.latency / 5) * 100}%` }} 
                className="h-full bg-amber-500 transition-all duration-1000" 
              />
            </div>
          </div>
        </div>

      </div>

      {/* Hardware GPU Matrix Panel */}
      <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2.5rem' }}>
        <h3 className="font-display font-bold text-lg text-white flex items-center gap-2" style={{ marginBottom: '1.5rem' }}>
          <Database className="w-5.5 h-5.5 text-accent-cyan" />
          <span>Hardware Acceleration Matrix</span>
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6" style={{ gap: '1rem' }}>
          <div className="flex flex-col rounded-xl bg-white/[0.01] border border-white/5" style={{ padding: '1.25rem' }}>
            <span className="text-[10px] text-slate-500 font-mono uppercase tracking-widest">Primary GPU</span>
            <span className="font-display font-bold text-base md:text-lg text-slate-200 mt-1">{vitals.gpuName}</span>
            <span className="text-xs text-slate-400 mt-1 font-mono">{vitals.gpuTotalVram}GB VRAM</span>
          </div>
          <div className="flex flex-col rounded-xl bg-white/[0.01] border border-white/5" style={{ padding: '1.25rem' }}>
            <span className="text-[10px] text-slate-500 font-mono uppercase tracking-widest">CUDA Cores</span>
            <span className="font-display font-bold text-base md:text-lg text-slate-200 mt-1">16,384</span>
            <span className="text-xs text-slate-400 mt-1 font-mono">Comp Cap 8.9</span>
          </div>
          <div className="flex flex-col rounded-xl bg-white/[0.01] border border-white/5" style={{ padding: '1.25rem' }}>
            <span className="text-[10px] text-slate-500 font-mono uppercase tracking-widest">Tensor Cores</span>
            <span className="font-display font-bold text-base md:text-lg text-slate-200 mt-1">512</span>
            <span className="text-xs text-slate-400 mt-1 font-mono">4th Gen Arch</span>
          </div>
          <div className="flex flex-col rounded-xl bg-white/[0.01] border border-white/5" style={{ padding: '1.25rem' }}>
            <span className="text-[10px] text-slate-500 font-mono uppercase tracking-widest">Driver / API</span>
            <span className="font-display font-bold text-base md:text-lg text-slate-200 mt-1">{vitals.gpuDriver}</span>
            <span className="text-xs text-slate-400 mt-1 font-mono">CUDA 12.2</span>
          </div>
          
          {/* Live GPU Power */}
          <div className="flex flex-col rounded-xl bg-amber-500/5 border border-amber-500/20 relative overflow-hidden group" style={{ padding: '1.25rem' }}>
            <div className="absolute inset-0 bg-amber-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <span className="text-[10px] text-amber-500/80 font-mono uppercase tracking-widest relative z-10">Power Draw</span>
            <span className="font-display font-bold text-base md:text-lg text-amber-400 mt-1 relative z-10">{vitals.gpuPower} W</span>
            <span className="text-xs text-amber-500/60 mt-1 font-mono relative z-10">Max: 450W</span>
          </div>

          {/* Live GPU Clock */}
          <div className="flex flex-col rounded-xl bg-emerald-500/5 border border-emerald-500/20 relative overflow-hidden group" style={{ padding: '1.25rem' }}>
            <div className="absolute inset-0 bg-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <span className="text-[10px] text-emerald-500/80 font-mono uppercase tracking-widest relative z-10">Core Clock</span>
            <span className="font-display font-bold text-base md:text-lg text-emerald-400 mt-1 relative z-10">{vitals.gpuClock} MHz</span>
            <span className="text-xs text-emerald-500/60 mt-1 font-mono relative z-10">Boost Active</span>
          </div>
        </div>
      </div>

      {/* 2. Interactive Signal Flow Diagram */}
      <div className="glass-panel overflow-hidden" style={{ padding: '2rem', marginBottom: '2.5rem' }}>
        <h3 className="font-display font-bold text-lg text-white flex items-center gap-2" style={{ marginBottom: '2rem' }}>
          <Network className="w-5.5 h-5.5 text-primary-purple" />
          <span>Ingest-to-Consensus Processing Schematic</span>
        </h3>

        {/* Horizontal Node Workflow */}
        <div className="flex flex-col lg:flex-row lg:items-stretch lg:justify-between relative" style={{ gap: '1.5rem' }}>
          
          {/* Central pipeline connecting path on desktop - Removed per user request */}

          {systemNodes.map((node, idx) => (
            <div 
              key={idx} 
              className="flex-1 glass-panel flex flex-col justify-between bg-black/35 relative border hover:border-primary-purple/35 transition-colors group"
              style={{ padding: '1.5rem' }}
            >
              <div className="flex flex-col" style={{ gap: '0.5rem' }}>
                <span className="text-[10px] text-primary-purple font-mono uppercase tracking-widest font-bold">
                  {node.phase}
                </span>
                <h4 className="font-display font-bold text-sm text-white group-hover:text-primary-purple transition-colors leading-snug">
                  {node.title}
                </h4>
                <p className="text-xs text-slate-400 font-light leading-relaxed mt-1">
                  {node.desc}
                </p>
              </div>

              <div className="text-[9px] font-mono text-slate-500 mt-6 pt-3 border-t border-white/5 uppercase tracking-wide">
                Engine: {node.tech}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 3. Tech Stack Specification Matrix */}
      <div className="glass-panel" style={{ padding: '2rem' }}>
        <h3 className="font-display font-bold text-lg text-white flex items-center gap-2" style={{ marginBottom: '1.5rem' }}>
          <Layers className="w-5.5 h-5.5 text-accent-cyan" />
          <span>Active Technical Core Ensembles</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3" style={{ gap: '1.5rem' }}>
          {techStack.map((group, idx) => (
            <div key={idx} className="flex flex-col rounded-xl bg-white/[0.01] border border-white/5" style={{ padding: '1.5rem' }}>
              <span className="font-display font-bold text-sm text-slate-200 uppercase tracking-widest border-b border-white/5" style={{ paddingBottom: '0.75rem', marginBottom: '1rem' }}>
                {group.group}
              </span>
              
              <ul className="flex flex-col font-mono text-xs text-slate-400" style={{ gap: '0.75rem' }}>
                {group.items.map((item, itemIdx) => (
                  <li key={itemIdx} className="flex items-center gap-2.5">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Spacer before footer */}
      <div className="h-16 md:h-24" />
    </div>
  );
}
