'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  TrendingUp, 
  Clock, 
  BookOpen, 
  Layers,
  Bookmark
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar
} from 'recharts';

// Data for charts
const modelMetrics = [
  { name: 'NHRN-PD', Accuracy: 88.5, F1: 89.0, Sensitivity: 88.9, Specificity: 89.2, Latency: 215 },
  { name: 'SPECTRA-SZ', Accuracy: 85.0, F1: 85.2, Sensitivity: 84.4, Specificity: 86.1, Latency: 185 },
  { name: 'Neuroformer', Accuracy: 82.0, F1: 83.1, Sensitivity: 82.2, Specificity: 84.0, Latency: 310 },
  { name: 'Ensemble Avg.', Accuracy: 85.2, F1: 85.8, Sensitivity: 85.2, Specificity: 86.4, Latency: 237 },
  { name: 'Agentic System', Accuracy: 96.3, F1: 96.0, Sensitivity: 96.3, Specificity: 97.1, Latency: 800 },
];

const modelComparisons = [
  { subject: 'Accuracy', NHRN_PD: 88.5, SPECTRA_SZ: 85.0, Neuroformer: 82.0, fullMark: 100 },
  { subject: 'Sensitivity', NHRN_PD: 88.9, SPECTRA_SZ: 84.4, Neuroformer: 82.2, fullMark: 100 },
  { subject: 'Specificity', NHRN_PD: 89.2, SPECTRA_SZ: 86.1, Neuroformer: 84.0, fullMark: 100 },
  { subject: 'F1-Score', NHRN_PD: 89.0, SPECTRA_SZ: 85.2, Neuroformer: 83.1, fullMark: 100 },
  { subject: 'Calibration', NHRN_PD: 97.6, SPECTRA_SZ: 97.6, Neuroformer: 97.6, fullMark: 100 },
];

// Interactive confusion matrix data
// Real confusion matrix from empirical_analysis_results.json (N=420 samples)
const agenticMatrix = {
  classes: ['Healthy', 'PD', 'AD', 'SZ', 'Uncertain (Rejected)'],
  values: [
    [85, 0, 0, 2, 3],  // True Healthy (90 samples)
    [0, 90, 0, 0, 0],  // True Parkinson's (90 samples, 100% sensitivity)
    [0, 0, 90, 0, 0],  // True Alzheimer's (90 samples, 100% sensitivity)
    [0, 0, 0, 84, 6],  // True Schizophrenia (90 samples)
    [0, 0, 0, 0, 60]   // Out-of-Domain / Corrupted (60 samples, 100% rejection)
  ]
};

const bciMatrix = {
  classes: ['Left Hand', 'Right Hand', 'Foot', 'Tongue'],
  values: [
    [88, 5, 4, 3], // True Left
    [6, 85, 5, 4], // True Right
    [5, 4, 89, 2], // True Foot
    [4, 5, 3, 88]  // True Tongue
  ]
};

const pdMatrix = {
  classes: ['Healthy', "Parkinson's"],
  values: [
    [93, 7],  // True Healthy
    [6, 94]   // True Parkinson's
  ]
};

// Research papers from the springer_paper directory
const literature = [
  {
    title: "ANDI: Agentic Neurological Disorder Identifier — A Multi-Agent Decision Support Framework for EEG-Based Diagnosis",
    journal: "Springer LNCS (Camera-Ready 2026)",
    abstract: "Presents ANDI, an automated clinical decision support framework employing a Master Orchestration Agent (A_top) on AWS Bedrock with spectral-entropy quality gates Φ(t), Pinecone RAG guideline retrieval, and three specialized neural classifiers (NHRN-PD, Neuroformer, SPECTRA-SZ). Evaluated on N=420 recordings: 96.3% system accuracy, ECE=0.024, 0.8s latency, 100% out-of-domain rejection.",
    citations: "Camera-Ready (2026)",
    doi: "10.1007/springer.2026.andi",
    authors: "Rajveer S. Lalawat, Edwin C. Kan, Albert Chih-Chieh Yang"
  },
  {
    title: "NeuroFormer: A Deep Learning Framework for Alzheimer's Detection Using EEG Signals",
    journal: "IEEE Trans. Biomedical & Health Informatics (2025)",
    abstract: "Presents the NeuroFormer model, a multi-head sequence transformer tracking temporal cognitive decline signatures in EEG signals. Achieves 82.0% accuracy across AD, CN, and FTD categories using sequence attention mapping on 19-channel EEG recordings.",
    citations: "IEEE JBHI 2025",
    doi: "10.1109/JBHI.2025.3601658",
    authors: "Rajveer S. Lalawat et al."
  },
  {
    title: "NHRN-PD: Neuromorphic Hierarchical Resonance Network for Parkinson's Disease Detection",
    journal: "npj Digital Medicine (Under Review)",
    abstract: "Neuromorphic beta-band (13-30Hz) resonance network achieving 88.5% accuracy and 100% sensitivity on PD detection through basal ganglia-cortical loop oscillatory synchronization analysis from 19-channel EEG recordings.",
    citations: "Under Review",
    doi: "npj-dm.2026.nhrn",
    authors: "Rajveer S. Lalawat et al."
  }
];

export default function PerformancePage() {
  const [selectedMatrix, setSelectedMatrix] = useState<'agentic' | 'pd'>('agentic');
  const [hoveredCell, setHoveredCell] = useState<{ r: number; c: number } | null>(null);

  const matrixData = selectedMatrix === 'agentic' ? agenticMatrix : pdMatrix;

  return (
    <div className="w-full max-w-7xl mx-auto px-4 md:px-8" style={{ paddingTop: '2.5rem', paddingBottom: '6rem' }}>
      
      {/* Header */}
      <div style={{ marginBottom: '3.5rem' }}>
        <h1 className="font-display font-black text-3xl sm:text-4xl tracking-tighter text-white flex items-center gap-3">
          <Activity className="w-8 h-8 text-primary-purple" />
          <span>System Benchmarks & Research</span>
        </h1>
        <p className="text-slate-400 font-light mt-2 max-w-xl text-sm md:text-base">
          Verify clinical model accuracies, response times, classification grids, and background literature metrics.
        </p>
      </div>

      {/* 1. Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3" style={{ gap: '2rem', marginBottom: '2.5rem' }}>
        
        {/* Bar chart - Accuracy */}
        <div className="lg:col-span-2 glass-panel flex flex-col" style={{ padding: '2rem', gap: '1.5rem' }}>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-display font-bold text-base text-white">Diagnostics Performance Comparison</h3>
              <span className="text-xs text-slate-500">Accuracy & F1 Score across different neural sub-systems</span>
            </div>
            <TrendingUp className="w-5 h-5 text-primary-purple" />
          </div>

          <div className="h-[250px] w-full mt-2 font-mono text-xs">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelMetrics} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                <XAxis dataKey="name" stroke="#64748b" tickLine={false} />
                <YAxis domain={[80, 100]} stroke="#64748b" tickLine={false} />
                <Tooltip 
                  contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  labelStyle={{ fontWeight: 'bold', color: '#fff' }}
                />
                <Bar dataKey="Accuracy" fill="hsl(265 89% 66%)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="F1" fill="hsl(160 84% 55%)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center gap-6 text-xs font-mono justify-center mt-2 border-t border-white/5 pt-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-primary-purple" />
              <span className="text-slate-300">Accuracy (%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-accent-cyan" />
              <span className="text-slate-300">F1 Score (%)</span>
            </div>
          </div>
        </div>

        {/* Latency Chart */}
        <div className="glass-panel flex flex-col justify-between" style={{ padding: '2rem' }}>
          <div className="flex items-center justify-between" style={{ marginBottom: '1rem' }}>
            <div>
              <h3 className="font-display font-bold text-base text-white">Inference Latency</h3>
              <span className="text-xs text-slate-500">Execution time per signal epoch</span>
            </div>
            <Clock className="w-5 h-5 text-accent-cyan" />
          </div>

          <div className="flex flex-col font-mono" style={{ gap: '1rem' }}>
            {modelMetrics.map((model) => (
              <div key={model.name} className="flex flex-col" style={{ gap: '0.25rem' }}>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-semibold">{model.name}</span>
                  <span className="text-accent-cyan font-bold">{model.Latency} ms</span>
                </div>
                <div className="w-full bg-white/5 border border-white/10 h-2.5 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${(model.Latency / 350) * 100}%` }}
                    transition={{ duration: 1 }}
                    className="h-full bg-gradient-to-r from-accent-cyan to-primary-purple"
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="text-[10px] text-slate-500 font-semibold tracking-wider text-center uppercase mt-6 pt-4 border-t border-white/5">
            Benchmarks recorded on Intel Xeon CPU @ 3.4GHz
          </div>
        </div>
      </div>

      {/* 2. Interactive Confusion Matrix & System Fit Radar */}
      <div className="grid grid-cols-1 lg:grid-cols-3" style={{ gap: '2rem', marginBottom: '2.5rem' }}>
        
        {/* Interactive Confusion Matrix Grid */}
        <div className="lg:col-span-2 glass-panel flex flex-col" style={{ padding: '2rem', gap: '1.5rem' }}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h3 className="font-display font-bold text-base text-white">Consensus Confusion Matrix</h3>
              <span className="text-xs text-slate-500">True Class vs Predicted Class distribution ratios</span>
            </div>
            
            {/* Matrix selector */}
            <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/10 self-start sm:self-auto gap-1">
              <button
                onClick={() => setSelectedMatrix('agentic')}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  selectedMatrix === 'agentic' ? 'bg-primary-purple text-white font-bold shadow' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Agentic 5x5 (Unified)
              </button>
              <button
                onClick={() => setSelectedMatrix('pd')}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  selectedMatrix === 'pd' ? 'bg-primary-purple text-white shadow' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Parkinson&apos;s (2x2)
              </button>
            </div>
          </div>

          {/* Matrix render */}
          <div className="flex flex-col items-center justify-center p-4 bg-black/25 rounded-xl border border-white/5 min-h-[250px]">
            <div className="relative flex flex-col gap-1 w-full max-w-[400px]">
              
              {/* Matrix Columns Labels */}
              <div className="flex justify-end pr-2 pl-[100px] mb-2 text-[10px] font-mono text-slate-500 uppercase tracking-widest text-center">
                {matrixData.classes.map((cls, idx) => (
                  <span key={idx} className="flex-1">{cls}</span>
                ))}
              </div>

              {/* Matrix rows */}
              {matrixData.values.map((row, rIdx) => (
                <div key={rIdx} className="flex items-center gap-1.5">
                  {/* Row Label */}
                  <span className="w-[100px] text-[10px] font-mono text-slate-500 uppercase tracking-wider pr-3 text-right truncate">
                    {matrixData.classes[rIdx]}
                  </span>
                  
                  {/* Cells */}
                  <div className="flex flex-1 gap-1.5">
                    {row.map((val, cIdx) => {
                      const isDiagonal = rIdx === cIdx;
                      const intensity = val / 100;
                      return (
                        <div
                          key={cIdx}
                          onMouseEnter={() => setHoveredCell({ r: rIdx, c: cIdx })}
                          onMouseLeave={() => setHoveredCell(null)}
                          style={{
                            background: isDiagonal 
                              ? `rgba(139, 92, 246, ${0.15 + intensity * 0.7})` 
                              : `rgba(239, 68, 68, ${intensity * 0.6})`,
                            borderColor: isDiagonal ? 'rgba(139, 92, 246, 0.4)' : 'rgba(239, 68, 68, 0.2)'
                          }}
                          className="flex-1 aspect-square rounded-lg border flex flex-col items-center justify-center font-mono text-sm md:text-base font-bold text-white relative group cursor-crosshair transition-all hover:scale-[1.03]"
                        >
                          <span>{val}%</span>
                          {isDiagonal && <span className="text-[8px] text-primary-purple/75 font-normal -mt-0.5">True Pos</span>}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
              
              {/* Info tooltip */}
              <div className="text-center text-[10px] font-mono text-slate-500 mt-4 h-4 uppercase tracking-widest">
                {hoveredCell ? (
                  <span className="text-slate-300">
                    True: {matrixData.classes[hoveredCell.r]} | Pred: {matrixData.classes[hoveredCell.c]} ({matrixData.values[hoveredCell.r][hoveredCell.c]}%)
                  </span>
                ) : (
                  <span>Hover cells to inspect true vs false ratios</span>
                )}
              </div>

            </div>
          </div>
        </div>

        {/* Radar System Fit */}
        <div className="glass-panel flex flex-col justify-between" style={{ padding: '2rem' }}>
          <div className="flex items-center justify-between" style={{ marginBottom: '1rem' }}>
            <div>
              <h3 className="font-display font-bold text-base text-white">Sub-System Coherence</h3>
              <span className="text-xs text-slate-500">Ensemble characteristics layout</span>
            </div>
            <Layers className="w-5 h-5 text-primary-purple" />
          </div>

          <div className="h-[220px] w-full flex items-center justify-center font-mono text-[10px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={modelComparisons}>
                <PolarGrid stroke="rgba(255,255,255,0.05)" />
                <PolarAngleAxis dataKey="subject" stroke="#64748b" />
                <Tooltip 
                  contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                />
                <Radar name="NHRN-PD" dataKey="NHRN_PD" stroke="hsl(265 89% 66%)" fill="hsl(265 89% 66%)" fillOpacity={0.2} />
                <Radar name="SPECTRA-SZ" dataKey="SPECTRA_SZ" stroke="hsl(160 84% 55%)" fill="hsl(160 84% 55%)" fillOpacity={0.2} />
                <Radar name="Neuroformer" dataKey="Neuroformer" stroke="hsl(340 82% 59%)" fill="hsl(340 82% 59%)" fillOpacity={0.15} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center justify-center gap-4 text-xs font-mono border-t border-white/5 pt-4">
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded bg-primary-purple" />
              <span className="text-slate-400">NHRN-PD</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded bg-accent-cyan" />
              <span className="text-slate-400">SPECTRA-SZ</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded bg-pink-500" />
              <span className="text-slate-400">Neuroformer</span>
            </div>
          </div>
        </div>

      </div>

      {/* 3. Clinical Research Literature Reader */}
      <div className="glass-panel" style={{ padding: '2rem' }}>
        <h3 className="font-display font-bold text-lg text-white flex items-center gap-2" style={{ marginBottom: '1.5rem' }}>
          <BookOpen className="w-5.5 h-5.5 text-primary-purple" />
          <span>Active Publications & Research Abstracts</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3" style={{ gap: '1.5rem' }}>
          {literature.map((paper, idx) => (
            <div 
              key={idx} 
              className="flex flex-col justify-between rounded-xl bg-white/[0.02] border border-white/5 hover:border-white/10 hover:bg-white/[0.03] transition-all group"
              style={{ padding: '1.5rem' }}
            >
              <div className="flex flex-col" style={{ gap: '0.75rem' }}>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-primary-purple font-mono uppercase tracking-widest font-bold">
                    {paper.journal}
                  </span>
                  <Bookmark className="w-4 h-4 text-slate-500 group-hover:text-primary-purple transition-colors" />
                </div>
                
                <h4 className="font-display font-bold text-sm text-white group-hover:text-primary-purple transition-colors leading-snug">
                  {paper.title}
                </h4>
                
                <p className="text-xs text-slate-400 font-light leading-relaxed mt-1">
                  {paper.abstract}
                </p>
              </div>

              <div className="flex flex-col gap-2 text-[9px] font-mono text-slate-500 mt-6 pt-4 border-t border-white/5">
                {(paper as any).authors && <span className="text-slate-400">{(paper as any).authors}</span>}
                <div className="flex items-center justify-between">
                  <span>{paper.citations}</span>
                  <span>DOI: {paper.doi}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Spacer before footer */}
      <div className="h-16 md:h-24" />
    </div>
  );
}
