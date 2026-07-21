'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ShieldAlert, 
  Upload, 
  Play, 
  Pause, 
  RefreshCw, 
  CheckCircle2, 
  AlertTriangle, 
  ChevronRight, 
  FileText,
  Activity,
  Layers,
  HeartPulse,
  Eye,
  Sliders,
  TrendingUp,
  BrainCircuit
} from 'lucide-react';
import Card3D from '@/components/Card3D';

// Clinical case details
const clinicalCases = [
  {
    id: 'case-1',
    name: 'Parkinsonian Oscillations Study',
    file: 'eeg_pd_session_09.edf',
    type: 'edf',
    specs: { channels: 22, duration: 12, rate: 250, quality: 'Medium' },
    waveConfig: { freq: [8, 12, 18], amp: [25, 45, 12], speed: 1.5, noise: 0.12 },
    results: [
      { key: 'eeg_pd', name: "EEG Parkinson's Detector", pred: "Parkinson's Disease", prob: 0.914, status: 'warn', report: "Neurological Parkinson's Detector: Detected abnormal rest-state basal ganglia beta rhythms (91.4% confidence), indicating early Parkinsonian activity. High-power beta coupling suggests potential dopaminergic pathway decay." },
      { key: 'neuroformer', name: 'Neuroformer Classifier', pred: 'Cognitively Normal', prob: 0.783, status: 'ok', report: "Neuroformer Classifier: Temporal sequence modeling identified Cognitively Normal temporal dynamics (78.3% confidence). Neuronal firing synchronization tracks expected cognitive pathways." },
      { key: 'bci2a', name: 'BCI2A Motor Decoder', pred: 'Uncertain (Left Hand)', prob: 0.352, status: 'info', report: "BCI2A Motor Decoder: Motor imagery classification is below threshold. Rest-state control baseline decoded movement intent as Uncertain (35.2% confidence)." },
      { key: 'spectra_sz', name: 'SPECTRA Routing Net', pred: 'Healthy', prob: 0.887, status: 'ok', report: "SPECTRA Routing: Complex psychiatric evaluation identified Healthy signatures (88.7% confidence). Rhythmic phase locking and multi-frequency band coupling indicate stable cognitive baseline." },
      { key: 'tumor_mri', name: 'MRI Morphology Net', pred: 'No Tumor', prob: 0.991, status: 'ok', report: "MRI Morphology: Spatial matrix CNN scans identified No Tumor (99.1% confidence). Multi-scale feature extraction maps trace structural density borders, indicating tissue layout consistency with healthy scans." }
    ],
    reasoning: "Metadata checks show 22 active electrodes. Signal preprocessor detects significant rhythmic slow oscillations in the 4-8Hz theta band and high-amplitude bursts in the 13-30Hz beta range, matching basal ganglia resting-state Parkinsonian abnormalities.",
    synthesis: "Consensus routing highlights a primary Parkinson's Disease classification (91.4% confidence) driven by theta-beta cortical power coupling. Cognitive networks show normal temporal dynamics, indicating absence of major Alzheimer's or dementia signatures. BCI command decoding is below threshold. Recommend full neurological evaluation, focusing on dopamine-responsive pathways."
  },
  {
    id: 'case-2',
    name: 'Alzheimer\'s Cohort Screening',
    file: 'neuro_ad_epoch_14.npy',
    type: 'npy',
    specs: { channels: 19, duration: 24, rate: 500, quality: 'High' },
    waveConfig: { freq: [4, 6, 10], amp: [50, 30, 8], speed: 0.8, noise: 0.05 },
    results: [
      { key: 'neuroformer', name: 'Neuroformer Classifier', pred: "Alzheimer's Disease (AD)", prob: 0.868, status: 'warn', report: "Neuroformer Classifier: Temporal sequence modeling identified Alzheimer's Disease (AD) signatures (86.8% confidence). The transformer self-attention map indicates focal synchrony decoupling in temporal-parietal node pathways." },
      { key: 'eeg_pd', name: "EEG Parkinson's Detector", pred: 'Healthy', prob: 0.925, status: 'ok', report: "Neurological Parkinson's Detector: Rhythmic activity is within healthy control ranges (92.5% confidence). Rest-state power spectrum density shows normal alpha/beta ratio with no significant parkinsonian tremor oscillations." },
      { key: 'spectra_sz', name: 'SPECTRA Routing Net', pred: 'Healthy', prob: 0.891, status: 'ok', report: "SPECTRA Routing: Complex psychiatric evaluation identified Healthy signatures (89.1% confidence). Rhythmic phase locking and multi-frequency band coupling indicate stable cognitive baseline." },
      { key: 'tumor_mri', name: 'MRI Morphology Net', pred: 'No Tumor', prob: 0.984, status: 'ok', report: "MRI Morphology: Spatial matrix CNN scans identified No Tumor (98.4% confidence). Multi-scale feature extraction maps trace structural density borders, indicating tissue layout consistency with healthy scans." },
      { key: 'bci2a', name: 'BCI2A Motor Decoder', pred: 'Uncertain', prob: 0.224, status: 'info', report: "BCI2A Motor Decoder: Motor imagery classification is below threshold. Rest-state control baseline decoded movement intent as Uncertain (22.4% confidence)." }
    ],
    reasoning: "Metadata checks show 19 active channels. Deep sequence modeling tracks delta/theta slowing along temporal-parietal nodes, indicating progressive synaptic decoupling and cognitive decline signatures.",
    synthesis: "Cognitive routing detects Alzheimer's Disease (AD) signatures with 86.8% confidence. Significant temporal coherence drop-offs match early-to-mid stage dementia sequences. Rest-state motor networks (Parkinsonian) and structural matrices are fully normal. Early cognitive testing and hippocampal volumetrics are recommended."
  },
  {
    id: 'case-3',
    name: 'Motor Imagery Rehabilitation',
    file: 'bci_imagery_rh.csv',
    type: 'csv',
    specs: { channels: 22, duration: 4, rate: 250, quality: 'High' },
    waveConfig: { freq: [10, 15, 24], amp: [30, 20, 25], speed: 2.2, noise: 0.03 },
    results: [
      { key: 'bci2a', name: 'BCI2A Motor Decoder', pred: 'Right Hand Imagery', prob: 0.947, status: 'ok', report: "Motor Function (BCI): Decoded movement intent as Right Hand Imagery with 94.7% confidence. The temporal convolutional layers tracked strong activity over motor cortex electrodes, matching the spatial patterns of right hand imagery planning." },
      { key: 'eeg_pd', name: "EEG Parkinson's Detector", pred: 'Healthy', prob: 0.952, status: 'ok', report: "Neurological Parkinson's Detector: Rhythmic activity is within healthy control ranges (95.2% confidence). Rest-state power spectrum density shows normal alpha/beta ratio with no parkinsonian tremor signatures." },
      { key: 'neuroformer', name: 'Neuroformer Classifier', pred: 'Cognitively Normal', prob: 0.913, status: 'ok', report: "Neuroformer Classifier: Temporal sequence modeling identified Cognitively Normal temporal dynamics (91.3% confidence). Neuronal firing synchronization tracks expected cognitive pathways." },
      { key: 'spectra_sz', name: 'SPECTRA Routing Net', pred: 'Healthy', prob: 0.928, status: 'ok', report: "SPECTRA Routing: Complex psychiatric evaluation identified Healthy signatures (92.8% confidence). Rhythmic phase locking and multi-frequency band coupling indicate stable cognitive baseline." },
      { key: 'tumor_mri', name: 'MRI Morphology Net', pred: 'No Tumor', prob: 0.998, status: 'ok', report: "MRI Morphology: Spatial matrix CNN scans identified No Tumor (99.8% confidence). Multi-scale feature extraction maps trace structural density borders, indicating tissue layout consistency with healthy scans." }
    ],
    reasoning: "Metadata checks show 22 active motor electrodes. Preprocessor isolated a localized 8-12Hz alpha/mu desynchronization (ERD) over the left hemisphere (C3 electrode), mapping motor cortex hand imagery activation.",
    synthesis: "BCI2A spatial-temporal decoder identifies a high-gain Right Hand Imagery command (94.7% confidence) suitable for motor rehabilitation control loops. All background clinical markers (Alzheimer's, Parkinson's, Tumor) are clean and within healthy boundaries."
  },
  {
    id: 'case-4',
    name: 'Spatial Brain MRI Screening',
    file: 'mri_scan_axial.png',
    type: 'image',
    specs: { channels: 1, duration: 1, rate: 0, quality: 'High (Spatial)' },
    waveConfig: { freq: [2, 5, 8], amp: [15, 20, 15], speed: 0.5, noise: 0.01 }, // slow laser scanner config
    results: [
      { key: 'tumor_mri', name: 'MRI Morphology Net', pred: 'Meningioma Detected', prob: 0.892, status: 'warn', report: "MRI Morphology: Spatial matrix CNN scans identified Meningioma Detected (89.2% confidence). Multi-scale feature extraction maps trace structural density borders, indicating tissue layout consistency with meningioma structures." },
      { key: 'eeg_pd', name: "EEG Parkinson's Detector", pred: 'Healthy', prob: 0.941, status: 'ok', report: "Neurological Parkinson's Detector: Rhythmic activity is within healthy control ranges (94.1% confidence). Rest-state power spectrum density shows normal alpha/beta ratio with no parkinsonian tremor signatures." },
      { key: 'neuroformer', name: 'Neuroformer Classifier', pred: 'Cognitively Normal', prob: 0.883, status: 'ok', report: "Neuroformer Classifier: Temporal sequence modeling identified Cognitively Normal temporal dynamics (88.3% confidence). Neuronal firing synchronization tracks expected cognitive pathways." },
      { key: 'spectra_sz', name: 'SPECTRA Routing Net', pred: 'Healthy', prob: 0.902, status: 'ok', report: "SPECTRA Routing: Complex psychiatric evaluation identified Healthy signatures (90.2% confidence). Rhythmic phase locking and multi-frequency band coupling indicate stable cognitive baseline." },
      { key: 'bci2a', name: 'BCI2A Motor Decoder', pred: 'Uncertain', prob: 0.284, status: 'info', report: "BCI2A Motor Decoder: Motor imagery classification is below threshold. Rest-state control baseline decoded movement intent as Uncertain (28.4% confidence)." }
    ],
    reasoning: "2D spatial conversion models pixel grid distributions. Multiscale feature fusion isolates a hyper-dense boundary mass in the dural matrix, matching meningioma cell clustering signatures.",
    synthesis: "Structural morphology CNN flags a local density indicative of Meningioma (89.2% confidence). The spatial lesion is localized to the axial plane with minimal perifocal edema. Resting-state neuro-oscillations are stable. Recommend urgent gadolinium-contrast MRI and neurosurgical consultation."
  }
];

export default function CMOPage() {
  const [selectedCase, setSelectedCase] = useState(clinicalCases[0]);
  const [customFile, setCustomFile] = useState<File | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [filterMode, setFilterMode] = useState<'raw' | 'filtered'>('filtered');
  const [isRouting, setIsRouting] = useState(false);
  const [routingStep, setRoutingStep] = useState(0);
  const [showReport, setShowReport] = useState(false);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  
  // REST API Integration States
  const [isApiOnline, setIsApiOnline] = useState(false);
  const [resultsMatrix, setResultsMatrix] = useState<Record<string, any>[]>(clinicalCases[0].results);
  const [reasoningText, setReasoningText] = useState(clinicalCases[0].reasoning);
  const [synthesisText, setSynthesisText] = useState(clinicalCases[0].synthesis);
  const [caseSpecs, setCaseSpecs] = useState(clinicalCases[0].specs);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const routingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check if FastAPI backend is online
  useEffect(() => {
    const checkApi = async () => {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      try {
        const res = await fetch(`${apiUrl}/api/status`);
        if (res.ok) {
          setIsApiOnline(true);
          console.log("FastAPI backend is ONLINE at", apiUrl);
        }
      } catch {
        console.log("FastAPI backend is OFFLINE, running in mock simulation mode");
      }
    };
    checkApi();
  }, []);

  // Sync results state with case selections when clicking a case
  const handleCaseSelect = (c: typeof clinicalCases[0]) => {
    setSelectedCase(c);
    setResultsMatrix(c.results);
    setReasoningText(c.reasoning);
    setSynthesisText(c.synthesis);
    setCaseSpecs(c.specs);
    setCustomFile(null);
    setShowReport(false);
    setIsRouting(false);
    setExpandedRow(null);
  };

  // Oscilloscope Animation Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let offset = 0;

    const resizeCanvas = () => {
      canvas.width = canvas.parentElement?.clientWidth || 600;
      canvas.height = 200;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const config = selectedCase.waveConfig;
      const chColors = [
        'rgba(139, 92, 246, 0.85)', // primary fuchsia
        'rgba(20, 184, 166, 0.85)', // accent cyan
        'rgba(236, 72, 153, 0.85)'  // pink
      ];

      // Draw Grid Lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;
      
      // vertical lines
      for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      // horizontal lines
      for (let y = 0; y < canvas.height; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      if (selectedCase.type === 'image') {
        // Draw Spatial Laser scan instead of waves
        const scanY = (Math.sin(offset * 0.05) * 0.5 + 0.5) * canvas.height;
        ctx.strokeStyle = 'rgba(20, 184, 166, 0.4)';
        ctx.lineWidth = 1.5;
        
        // draw matrix scan patterns
        ctx.beginPath();
        for (let i = 0; i < canvas.width; i += 8) {
          const heightOffset = Math.sin(i * 0.05 + offset) * 15;
          ctx.rect(i, scanY - 30 + heightOffset, 6, 6);
        }
        ctx.stroke();

        // Laser beam line
        ctx.strokeStyle = 'rgba(20, 184, 166, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, scanY);
        ctx.lineTo(canvas.width, scanY);
        ctx.stroke();

        // Glow
        ctx.shadowBlur = 15;
        ctx.shadowColor = '#14b8a6';
        ctx.strokeStyle = 'rgba(20, 184, 166, 0.9)';
        ctx.stroke();
        ctx.shadowBlur = 0; // reset
      } else {
        // Draw 3 Channels
        chColors.forEach((color, chIdx) => {
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.8;

          const baseHeight = (chIdx + 1) * (canvas.height / 4);
          
          for (let x = 0; x < canvas.width; x++) {
            let y = baseHeight;

            // Generate composite sin waves
            config.freq.forEach((f, fIdx) => {
              const amp = config.amp[fIdx] * (filterMode === 'filtered' ? 0.8 : 1.25);
              y += Math.sin((x * f * 0.003) - offset * config.speed + chIdx) * amp;
            });

            // Add raw noise if raw mode is toggled
            if (filterMode === 'raw') {
              y += (Math.random() - 0.5) * config.amp[0] * config.noise * 3.5;
            } else {
              y += (Math.random() - 0.5) * config.amp[0] * config.noise * 0.45;
            }

            if (x === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          }
          ctx.stroke();
        });
      }

      if (isPlaying) {
        offset += 0.05;
      }
      animationId = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationId);
    };
  }, [selectedCase, isPlaying, filterMode]);

  // File Selection Handlers
  const handleFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setCustomFile(file);
      setShowReport(false);
      setIsRouting(false);
      
      // Visual feedback specs guess
      setCaseSpecs({
        channels: file.name.endsWith('.npy') ? 1 : 22,
        duration: 10,
        rate: 250,
        quality: 'High (User File)'
      });
    }
  };

  // Handle running diagnostics (calling REST API or falling back to simulation)
  const handleStartAnalysis = async () => {
    setIsRouting(true);
    setRoutingStep(0);
    setShowReport(false);
    setExpandedRow(null);
    
    // Simulate steps log
    const animateLogs = () => {
      return new Promise<void>((resolve) => {
        setRoutingStep(0);
        routingTimerRef.current = setTimeout(() => {
          setRoutingStep(1);
          routingTimerRef.current = setTimeout(() => {
            setRoutingStep(2);
            routingTimerRef.current = setTimeout(() => {
              setRoutingStep(3);
              routingTimerRef.current = setTimeout(() => {
                resolve();
              }, 1200);
            }, 1200);
          }, 1200);
        }, 1200);
      });
    };

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    if (isApiOnline) {
      try {
        const formData = new FormData();
        if (customFile) {
          formData.append('file', customFile);
        } else {
          formData.append('case_id', selectedCase.id);
        }

        const [apiRes] = await Promise.all([
          fetch(`${apiUrl}/api/diagnose`, {
            method: 'POST',
            body: formData
          }).then(r => {
            if (!r.ok) throw new Error("Inference failed");
            return r.json();
          }),
          animateLogs()
        ]);

        const apiResults = apiRes.results;
        const apiRouting = apiRes.routing || {};
        
        // Map API results keys to match the frontend representation
        const getModelStatus = (pred: string, normalVal: string, isBci: boolean = false) => {
          if (pred.includes('Uncertain') || pred.includes('Error') || pred.includes('Unknown')) return 'info';
          if (isBci) return 'ok';
          return pred !== normalVal ? 'warn' : 'ok';
        };

        const mappedResults = [
          { key: 'eeg_pd', name: "EEG Parkinson's Detector", pred: apiResults.eeg_pd?.prediction || 'Uncertain', prob: apiResults.eeg_pd?.probability || 0, status: getModelStatus(apiResults.eeg_pd?.prediction || 'Uncertain', 'Healthy'), report: apiRes.individual_reports?.eeg_pd || "No report generated." },
          { key: 'nhrn_pd', name: "NHRN Parkinson's Net", pred: apiResults.nhrn_pd?.prediction || 'Uncertain', prob: apiResults.nhrn_pd?.probability || 0, status: getModelStatus(apiResults.nhrn_pd?.prediction || 'Uncertain', 'Healthy'), report: apiRes.individual_reports?.nhrn_pd || "No report generated." },
          { key: 'neuroformer', name: 'Neuroformer Classifier', pred: apiResults.neuroformer?.prediction || 'Uncertain', prob: apiResults.neuroformer?.probability || 0, status: getModelStatus(apiResults.neuroformer?.prediction || 'Uncertain', 'CN'), report: apiRes.individual_reports?.neuroformer || "No report generated." },
          { key: 'bci2a', name: 'BCI2A Motor Decoder', pred: apiResults.bci2a_crdae?.prediction || 'Uncertain', prob: apiResults.bci2a_crdae?.probability || 0, status: getModelStatus(apiResults.bci2a_crdae?.prediction || 'Uncertain', '', true), report: apiRes.individual_reports?.bci2a_crdae || "No report generated." },
          { key: 'spectra_sz', name: 'SPECTRA Routing Net', pred: apiResults.spectra_sz?.prediction || 'Uncertain', prob: apiResults.spectra_sz?.probability || 0, status: getModelStatus(apiResults.spectra_sz?.prediction || 'Uncertain', 'Healthy'), report: apiRes.individual_reports?.spectra_sz || "No report generated." },
          { key: 'tumor_mri', name: 'MRI Morphology Net', pred: apiResults.brain_tumor_mri?.prediction || 'Uncertain', prob: apiResults.brain_tumor_mri?.probability || 0, status: getModelStatus(apiResults.brain_tumor_mri?.prediction || 'Uncertain', 'No Tumor'), report: apiRes.individual_reports?.brain_tumor_mri || "No report generated." }
        ];

        setResultsMatrix(mappedResults);
        setReasoningText(apiRouting.reasoning || "Agent parsed characteristics.");
        setSynthesisText(apiRes.synthesis || "CMO Consensus finished.");
        
        if (apiRes.characteristics) {
          setCaseSpecs({
            channels: apiRes.characteristics.channels || 22,
            duration: Math.round(apiRes.characteristics.duration_estimate || 10),
            rate: apiRes.characteristics.samples ? Math.round(apiRes.characteristics.samples / (apiRes.characteristics.duration_estimate || 1)) : 250,
            quality: apiRes.characteristics.signal_quality || 'medium'
          });
        }

        setIsRouting(false);
        setShowReport(true);
      } catch (err) {
        console.error("API Error: falling back to simulation", err);
        await animateLogs();
        setIsRouting(false);
        setShowReport(true);
      }
    } else {
      await animateLogs();
      if (customFile) {
        // In simulation mode with a custom file, generate simulated individual reports for the active case results
        const simulatedResults = selectedCase.results.map(r => ({
          ...r,
          report: `Simulated Analysis for custom file "${customFile.name}": Node classified as ${r.pred} with ${Math.round(r.prob * 100)}% confidence target.`
        }));
        setResultsMatrix(simulatedResults);
      } else {
        // Reset matrix to select case results to guarantee we have the individual reports
        setResultsMatrix(selectedCase.results);
      }
      setIsRouting(false);
      setShowReport(true);
    }
  };

  useEffect(() => {
    return () => {
      if (routingTimerRef.current) {
        clearTimeout(routingTimerRef.current);
      }
    };
  }, []);

  return (
    <div className="w-full max-w-7xl mx-auto px-4 md:px-8 pt-10 pb-24 md:pb-36 flex flex-col" style={{ gap: '3.5rem', marginTop: '2rem' }}>
      
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6" style={{ marginBottom: '2rem' }}>
        <div className="max-w-3xl">
          <h1 className="font-display font-black text-4xl sm:text-5xl tracking-tighter text-white flex items-center gap-4">
            <ShieldAlert className="w-10 h-10 text-primary-purple animate-pulse" />
            <span>Virtual CMO Hub</span>
          </h1>
          <p className="text-slate-400 font-light mt-4 text-base md:text-lg leading-relaxed">
            Select a simulated patient profile or upload a custom signal matrix file. The Chief Medical Officer agent runs consensus diagnostics across all models simultaneously.
          </p>
        </div>
        
        {/* Quality status indicators */}
        <div className="flex items-center gap-3 shrink-0 mb-2">
          {isApiOnline ? (
            <div className="px-4 py-2.5 rounded-xl border border-emerald-500/25 bg-emerald-500/10 text-emerald-400 text-sm font-semibold flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>Backend Connected</span>
            </div>
          ) : (
            <div className="px-4 py-2.5 rounded-xl border border-amber-500/25 bg-amber-500/10 text-amber-400 text-sm font-semibold flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse" />
              <span>Simulation Mode</span>
            </div>
          )}
        </div>
      </div>

      {/* Section 1: Data Intake */}
      <div className="flex flex-col lg:flex-row gap-6 md:gap-8">
        {/* Left: Patient Records Grid */}
        <div className="lg:w-2/3 glass-panel card-content-wrapper flex flex-col" style={{ padding: '2.5rem' }}>
          <div className="flex items-center justify-between" style={{ marginBottom: '2.5rem' }}>
            <h2 className="font-display font-bold text-2xl text-white flex items-center gap-4">
              <BrainCircuit className="w-8 h-8 text-primary-purple" />
              <span>Select Patient Record</span>
            </h2>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 md:gap-8 flex-grow">
            {clinicalCases.map((c) => (
              <button
                key={c.id}
                onClick={() => handleCaseSelect(c)}
                className={`flex flex-col justify-center rounded-2xl border text-left transition-all ${
                  selectedCase.id === c.id && !customFile
                    ? 'border-primary-purple/50 bg-primary-purple/[0.08] shadow-[0_0_30px_rgba(155,91,245,0.15)]'
                    : 'border-white/10 bg-white/[0.02] hover:bg-white/[0.06] hover:border-white/20'
                }`}
                style={{ padding: '1.75rem' }}
              >
                <span className="block font-bold text-lg text-slate-100 mb-3">{c.name}</span>
                <span className="block text-[11px] text-slate-400 font-mono uppercase tracking-widest leading-relaxed">
                  File: {c.file}<br />({c.type})
                </span>
              </button>
            ))}
          </div>
        </div>
        
        {/* Right: Custom file drop simulator */}
        <div className="lg:w-1/3 glass-panel card-content-wrapper flex flex-col" style={{ padding: '2.5rem' }}>
          <h2 className="font-display font-bold text-2xl text-white flex items-center gap-4" style={{ marginBottom: '2.5rem' }}>
            <Upload className="w-8 h-8 text-primary-purple" />
            <span>Custom Ingestion</span>
          </h2>
          
          <div 
            onClick={handleFileClick}
            className={`flex-grow border-2 border-dashed rounded-2xl flex flex-col items-center justify-center text-center bg-white/[0.01] hover:bg-white/[0.04] transition-colors cursor-pointer group ${
              customFile ? 'border-primary-purple/50 bg-primary-purple/[0.05] animate-pulse' : 'border-white/10 hover:border-white/25'
            }`}
            style={{ padding: '2rem' }}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              className="hidden" 
              accept=".csv,.txt,.npy,.edf,.bdf" 
            />
            <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <Upload className={`w-10 h-10 ${
                customFile ? 'text-primary-purple' : 'text-slate-400 group-hover:text-primary-purple'
              }`} />
            </div>
            <span className="text-lg font-semibold text-slate-200 mb-3">
              {customFile ? `Loaded: ${customFile.name}` : 'Upload Signal Matrix'}
            </span>
            <span className="text-sm text-slate-500 leading-relaxed max-w-[220px]">
              {customFile ? `${Math.round(customFile.size / 1024)} KB` : 'Supports raw CSV, EDF, and NPY structural data'}
            </span>
          </div>
        </div>
      </div>

      {/* Section 2: Oscilloscope Panel */}
      <div className="glass-panel flex flex-col relative overflow-hidden" style={{ padding: '2.5rem', gap: '2rem' }}>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6" style={{ marginBottom: '1.5rem' }}>
          <div>
            <h2 className="font-display font-bold text-2xl text-white">Oscilloscope Feeds</h2>
            <span className="text-sm text-slate-400 mt-2 block">
              {selectedCase.type === 'image' && !customFile ? 'Axial Slice Scan Matrix Visualization' : 'Multichannel Continuous Signal Activity'}
            </span>
          </div>
          
          {/* Controls */}
          <div className="flex items-center" style={{ gap: '1rem' }}>
            {selectedCase.type !== 'image' && (
              <div className="flex bg-white/5 rounded-xl border border-white/10" style={{ padding: '0.25rem', gap: '0.25rem' }}>
                <button
                  onClick={() => setFilterMode('filtered')}
                  className={`rounded-lg text-sm font-semibold transition-all ${
                    filterMode === 'filtered' ? 'bg-primary-purple text-white shadow-lg shadow-primary-purple/20' : 'text-slate-400 hover:text-slate-200'
                  }`}
                  style={{ padding: '0.5rem 1.25rem' }}
                >
                  Filtered (1-40Hz)
                </button>
                <button
                  onClick={() => setFilterMode('raw')}
                  className={`rounded-lg text-sm font-semibold transition-all ${
                    filterMode === 'raw' ? 'bg-primary-purple text-white shadow-lg shadow-primary-purple/20' : 'text-slate-400 hover:text-slate-200'
                  }`}
                  style={{ padding: '0.5rem 1.25rem' }}
                >
                  Raw Signal
                </button>
              </div>
            )}

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="rounded-xl bg-white/5 border border-white/10 text-white hover:bg-white/10 transition-colors"
              style={{ padding: '0.75rem' }}
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Canvas viewport */}
        <div className="bg-black/60 border border-white/5 rounded-2xl overflow-hidden p-6 lg:p-8 flex items-center justify-center min-h-[300px] shadow-inner">
          <canvas ref={canvasRef} className="w-full h-[240px]" />
        </div>

        {/* Signal Specifications */}
        <div className="grid grid-cols-2 lg:grid-cols-4 border-t border-white/5 text-sm font-mono text-slate-500" style={{ gap: '2rem', marginTop: '1.5rem', paddingTop: '2.5rem' }}>
          <div className="bg-white/[0.02] rounded-xl border border-white/5" style={{ padding: '1.5rem' }}>
            <span className="block text-slate-400 uppercase tracking-widest text-[10px]" style={{ marginBottom: '0.75rem' }}>Ingested Nodes</span>
            <span className="font-bold text-slate-200 text-lg">{caseSpecs.channels} leads</span>
          </div>
          <div className="bg-white/[0.02] rounded-xl border border-white/5" style={{ padding: '1.5rem' }}>
            <span className="block text-slate-400 uppercase tracking-widest text-[10px]" style={{ marginBottom: '0.75rem' }}>Epoch Duration</span>
            <span className="font-bold text-slate-200 text-lg">{caseSpecs.duration} seconds</span>
          </div>
          <div className="bg-white/[0.02] rounded-xl border border-white/5" style={{ padding: '1.5rem' }}>
            <span className="block text-slate-400 uppercase tracking-widest text-[10px]" style={{ marginBottom: '0.75rem' }}>Sampling Frequency</span>
            <span className="font-bold text-slate-200 text-lg">{caseSpecs.rate} Hz</span>
          </div>
          <div className="bg-white/[0.02] rounded-xl border border-white/5" style={{ padding: '1.5rem' }}>
            <span className="block text-slate-400 uppercase tracking-widest text-[10px]" style={{ marginBottom: '0.75rem' }}>Signal Quality</span>
            <span className={`font-bold text-lg uppercase ${
              caseSpecs.quality === 'High' || caseSpecs.quality === 'high' || caseSpecs.quality === 'High (User File)' ? 'text-emerald-400' : 'text-amber-400'
            }`}>{caseSpecs.quality}</span>
          </div>
        </div>
      </div>

      {/* Section 3: Action & Synthesis */}
      <div className="grid grid-cols-1 lg:grid-cols-2" style={{ gap: '2rem' }}>
        
        {/* Action Trigger Box */}
        <div className="glass-panel bg-gradient-to-br from-primary-purple/[0.05] to-transparent border-primary-purple/20 flex flex-col items-center text-center justify-center min-h-[400px]" style={{ padding: '2.5rem' }}>
          <div className="w-20 h-20 rounded-3xl bg-primary-purple/10 border border-primary-purple/20 flex items-center justify-center text-primary-purple shadow-[0_0_20px_rgba(139,92,246,0.2)] mb-6 shrink-0">
            <ShieldAlert className="w-10 h-10 text-primary-purple animate-pulse" />
          </div>
          <h3 className="font-display font-bold text-2xl text-white tracking-tight mb-3 shrink-0">Consensus Diagnostics</h3>
          <p className="text-base text-slate-400 font-light max-w-sm leading-relaxed mb-8 shrink-0">
            Direct the Virtual CMO agent to feed the ingested signal across all loaded classifiers simultaneously.
          </p>
          <button
            onClick={handleStartAnalysis}
            disabled={isRouting}
            className="w-full max-w-sm shrink-0 bg-gradient-to-r from-primary-purple/90 to-accent-cyan/90 hover:from-primary-purple hover:to-accent-cyan rounded-2xl font-bold text-white uppercase tracking-widest text-sm shadow-[0_0_20px_rgba(139,92,246,0.25)] hover:shadow-[0_0_35px_rgba(139,92,246,0.4)] transition-all flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ padding: '1.25rem 2rem', marginTop: '1rem' }}
          >
            {isRouting ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" />
                <span>Agent Working...</span>
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                <span>Run Diagnostics</span>
              </>
            )}
          </button>
        </div>

        {/* Routing Steps Log */}
        <div className="glass-panel flex flex-col max-h-[500px]" style={{ padding: '2.5rem' }}>
          <h3 className="font-display font-bold text-base text-white uppercase tracking-widest flex items-center justify-between mb-8 pb-6 border-b border-white/5">
            <span>Routing Log Stream</span>
            {isRouting && <span className="text-[10px] px-3 py-1.5 rounded-full bg-primary-purple/10 text-primary-purple font-mono animate-pulse border border-primary-purple/20">LIVE</span>}
          </h3>
          
          {isRouting ? (
            <div className="flex flex-col gap-8 font-mono text-sm text-slate-400 mt-2">
              <div className="flex items-center gap-4">
                {routingStep >= 0 ? (
                  <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0" />
                ) : (
                  <RefreshCw className="w-5 h-5 animate-spin text-slate-500 shrink-0" />
                )}
                <span className={routingStep >= 0 ? 'text-slate-200' : ''}>Ingested raw clinical signal matrix</span>
              </div>

              <div className="flex items-center gap-4">
                {routingStep >= 1 ? (
                  <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0" />
                ) : routingStep === 0 ? (
                  <RefreshCw className="w-5 h-5 animate-spin text-primary-purple shrink-0" />
                ) : (
                  <span className="w-6 h-6 border border-white/10 rounded-full shrink-0" />
                )}
                <span className={routingStep >= 1 ? 'text-slate-200' : ''}>Agentic data evaluation & quality checks</span>
              </div>

              <div className="flex items-center gap-4">
                {routingStep >= 2 ? (
                  <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0" />
                ) : routingStep === 1 ? (
                  <RefreshCw className="w-5 h-5 animate-spin text-primary-purple shrink-0" />
                ) : (
                  <span className="w-6 h-6 border border-white/10 rounded-full shrink-0" />
                )}
                <span className={routingStep >= 2 ? 'text-slate-200' : ''}>Routing payload to concurrent ensembles</span>
              </div>

              <div className="flex items-center gap-4">
                {routingStep >= 3 ? (
                  <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0" />
                ) : routingStep === 2 ? (
                  <RefreshCw className="w-5 h-5 animate-spin text-primary-purple shrink-0" />
                ) : (
                  <span className="w-6 h-6 border border-white/10 rounded-full shrink-0" />
                )}
                <span className={routingStep >= 3 ? 'text-slate-200' : ''}>Compiling outputs & synthesizing report</span>
              </div>
            </div>
          ) : (
            <div className="flex-grow flex flex-col items-center justify-center opacity-40">
               <BrainCircuit className="w-16 h-16 text-slate-500 mb-6" />
               <p className="text-center text-slate-400 max-w-xs font-mono text-xs uppercase tracking-widest leading-relaxed">
                 Agent standby.<br/>Awaiting execution trigger to begin consensus sequence.
               </p>
            </div>
          )}
        </div>

      </div>

      {/* Diagnosis Report Section */}
      <AnimatePresence>
        {showReport && (
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col gap-10 md:gap-14 mt-6"
          >
            {/* Model Comparison Table */}
            <div className="glass-panel flex flex-col" style={{ padding: '3.5rem', gap: '2.5rem' }}>
              <div className="flex flex-col lg:flex-row lg:items-center justify-between border-b border-white/5" style={{ paddingBottom: '2rem', gap: '1.5rem' }}>
                <div>
                  <h3 className="font-display font-bold text-2xl text-white" style={{ marginBottom: '0.5rem' }}>Diagnostics Results Ensemble Matrix</h3>
                  <p className="text-slate-400 text-sm">Detailed predictions from the interconnected sub-model nodes.</p>
                </div>
                <div className="rounded-xl bg-white/[0.02] border border-white/5 text-xs font-mono text-slate-400 flex items-center" style={{ padding: '0.5rem 1.25rem', gap: '0.75rem' }}>
                  <Activity className="w-4 h-4 text-emerald-400" />
                  <span>Nodes Synchronized</span>
                </div>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b-2 border-white/10 text-slate-500 font-mono text-xs uppercase tracking-widest">
                      <th style={{ paddingBottom: '1.25rem', paddingLeft: '1rem', paddingRight: '1rem' }}>Clinical Model Node</th>
                      <th style={{ paddingBottom: '1.25rem', paddingLeft: '1rem', paddingRight: '1rem' }}>Diagnostic Prediction</th>
                      <th className="text-right" style={{ paddingBottom: '1.25rem', paddingLeft: '1rem', paddingRight: '1rem' }}>Confidence Target</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 text-slate-300">
                    {resultsMatrix.map((r) => {
                      const isUncertain = r.pred.includes('Uncertain');
                      const isWarning = r.status === 'warn';
                      const isInfo = r.status === 'info';
                      const isExpanded = expandedRow === r.key;
                      return (
                        <React.Fragment key={r.key}>
                          <tr 
                            onClick={() => setExpandedRow(isExpanded ? null : r.key)}
                            className="hover:bg-white/[0.04] active:bg-white/[0.06] transition-colors group cursor-pointer"
                          >
                            <td className="font-semibold text-white flex items-center text-base" style={{ paddingTop: '1.5rem', paddingBottom: '1.5rem', paddingLeft: '1rem', paddingRight: '1rem', gap: '1rem' }}>
                              <div className={`p-2 rounded-lg ${
                                isWarning ? 'bg-amber-500/10' : isInfo ? 'bg-slate-500/10' : 'bg-emerald-500/10'
                              }`}>
                                {isWarning ? (
                                  <AlertTriangle className="w-5 h-5 text-amber-500" />
                                ) : isInfo ? (
                                  <Activity className="w-5 h-5 text-slate-500" />
                                ) : (
                                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                                )}
                              </div>
                              <div className="flex flex-col text-left">
                                <span className="flex items-center gap-2">
                                  {r.name}
                                  <ChevronRight className={`w-4 h-4 text-slate-500 transition-transform duration-300 ${isExpanded ? 'rotate-90 text-primary-purple' : 'group-hover:text-slate-300'}`} />
                                </span>
                                <span className="text-[10px] text-slate-500 uppercase tracking-widest font-mono mt-1">Click to view model report</span>
                              </div>
                            </td>
                            <td className={`font-mono text-sm ${
                              isWarning ? 'text-rose-400 font-bold' : isUncertain ? 'text-slate-500' : 'text-slate-200'
                            }`} style={{ paddingTop: '1.5rem', paddingBottom: '1.5rem', paddingLeft: '1rem', paddingRight: '1rem' }}>
                              {r.pred}
                            </td>
                            <td className="text-right" style={{ paddingTop: '1.5rem', paddingBottom: '1.5rem', paddingLeft: '1rem', paddingRight: '1rem' }}>
                              <div className="flex items-center justify-end" style={{ gap: '1.25rem' }}>
                                <span className="font-bold font-mono text-sm text-slate-200">{Math.round(r.prob * 100)}%</span>
                                <div className="w-32 bg-black/40 border border-white/10 h-3 rounded-full overflow-hidden">
                                  <motion.div 
                                    initial={{ width: 0 }}
                                    animate={{ width: `${r.prob * 100}%` }}
                                    transition={{ duration: 1.2, ease: 'easeOut' }}
                                    className={`h-full rounded-full ${
                                      isWarning ? 'bg-gradient-to-r from-rose-500/80 to-amber-500/80' : 'bg-gradient-to-r from-primary-purple to-accent-cyan'
                                    }`} 
                                  />
                                </div>
                              </div>
                            </td>
                          </tr>
                          
                          {/* Expanded Detail Panel */}
                          {isExpanded && (
                            <tr className="bg-black/25">
                              <td colSpan={3} className="border-l-2 border-l-primary-purple" style={{ paddingTop: '2.25rem', paddingBottom: '2.25rem', paddingLeft: '3.5rem', paddingRight: '2.5rem' }}>
                                <motion.div
                                  initial={{ opacity: 0, y: -5 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  transition={{ duration: 0.2 }}
                                  className="flex flex-col gap-4 text-left"
                                >
                                  <div className="flex items-center gap-2 text-slate-500 font-mono text-[10px] uppercase tracking-wider">
                                    <FileText className="w-4 h-4 text-primary-purple animate-pulse" />
                                    <span>Model-Specific Clinical Analysis</span>
                                  </div>
                                  <p className="text-slate-300 text-sm leading-relaxed max-w-4xl italic mt-1">
                                    &quot;{r.report || "No specific report generated for this node prediction."}&quot;
                                  </p>
                                </motion.div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div className="rounded-2xl bg-black/40 border border-white/5 text-sm font-mono text-slate-400 flex items-start" style={{ padding: '2rem', gap: '1.25rem' }}>
                <Sliders className="w-6 h-6 text-accent-cyan shrink-0 mt-0.5" />
                <p className="leading-relaxed">
                  <strong className="text-slate-200 text-base block" style={{ marginBottom: '0.5rem' }}>Routing Meta reasoning:</strong> 
                  <span className="text-slate-400 text-sm">{reasoningText}</span>
                </p>
              </div>
            </div>

            {/* Synthesized CMO Report Letter */}
            <div className="glass-panel bg-gradient-to-b from-white/[0.02] to-black/60 relative border-t-2 border-white/10 flex flex-col overflow-hidden" style={{ padding: '3.5rem', gap: '2.5rem' }}>
              {/* Decorative background element */}
              <div className="absolute top-0 right-0 w-96 h-96 bg-primary-purple/10 rounded-full blur-[100px] pointer-events-none transform translate-x-1/3 -translate-y-1/3" />
              
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6 relative z-10">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-xl bg-primary-purple/20 border border-primary-purple/30">
                    <FileText className="w-6 h-6 text-primary-purple" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-xl uppercase tracking-wide text-white">Synthesized Medical Report</h3>
                    <p className="text-slate-400 text-sm mt-1">Official Output from Virtual CMO</p>
                  </div>
                </div>
                <span className="px-4 py-2 rounded-lg bg-black/50 border border-white/10 text-xs text-slate-400 font-mono uppercase tracking-widest">
                  SOURCE: {customFile ? 'USER_FILE' : selectedCase.id.toUpperCase()}
                </span>
              </div>
              
              {/* Report content */}
              <div className="border-t border-b border-white/10 text-sm leading-relaxed text-slate-400 font-sans flex flex-col relative z-10" style={{ padding: '2.5rem 0', margin: '0.5rem 0', gap: '2rem' }}>
                <div className="grid grid-cols-1 sm:grid-cols-3 bg-black/30 rounded-2xl border border-white/5" style={{ gap: '1.5rem', padding: '1.5rem' }}>
                  <div>
                    <p className="text-slate-500 uppercase tracking-widest text-[10px]" style={{ marginBottom: '0.25rem' }}>Date</p>
                    <p className="text-slate-200 font-medium">May 25, 2026</p>
                  </div>
                  <div>
                    <p className="text-slate-500 uppercase tracking-widest text-[10px]" style={{ marginBottom: '0.25rem' }}>Subject</p>
                    <p className="text-slate-200 font-medium">Integrated Neural Analysis</p>
                  </div>
                  <div>
                    <p className="text-slate-500 uppercase tracking-widest text-[10px]" style={{ marginBottom: '0.25rem' }}>Ingested Source</p>
                    <p className="text-slate-200 font-medium truncate" title={customFile ? customFile.name : selectedCase.file}>
                      {customFile ? customFile.name : selectedCase.file}
                    </p>
                  </div>
                </div>
                
                <div className="bg-white/[0.03] border-l-4 border-l-primary-purple rounded-r-2xl" style={{ padding: '2rem' }}>
                  <p className="italic text-slate-200 leading-relaxed text-base md:text-lg">
                    &quot;{synthesisText}&quot;
                  </p>
                </div>
                
                <div className="text-[10px] font-mono border-t border-white/5 leading-normal text-slate-500 uppercase tracking-widest flex justify-between items-center" style={{ paddingTop: '2rem' }}>
                  <span>Digitally Synthesized by Virtual Chief Medical Officer Agent</span>
                  <span className="text-primary-purple/60 font-bold">{"/// SIGNED_VIRTUAL_CMO"}</span>
                </div>
              </div>

              <div className="flex items-start text-sm text-slate-300 border border-amber-500/20 bg-amber-500/5 rounded-2xl relative z-10" style={{ gap: '1.25rem', padding: '1.5rem' }}>
                <AlertTriangle className="w-6 h-6 text-amber-500 shrink-0 mt-0.5" />
                <p className="leading-relaxed">
                  <strong className="text-amber-400 font-bold">Clinical Disclaimer:</strong> This report is an AI-synthesized research outcome from simulated agentic networks and does not substitute a formal clinical diagnosis by a licensed medical professional.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Spacer before footer */}
      <div className="h-16 md:h-24" />
    </div>
  );
}
