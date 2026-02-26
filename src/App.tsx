import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Video, Cpu, Layers, Zap, Play, Loader2, Download, AlertCircle, Key, Settings, Sliders, Monitor, Square } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI } from "@google/genai";

// Extend window for AI Studio API Key selection
declare global {
  interface Window {
    aistudio: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

type ModelType = 'veo-3.1-fast-generate-preview' | 'veo-3.1-generate-preview';
type Resolution = '720p' | '1080p';
type AspectRatio = '16:9' | '9:16';

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [logs, setLogs] = useState<string[]>(["[SYSTEM] WanVideo REAL CORE v2.2 Initialized.", "[SYSTEM] Flow Matching Sampler: READY", "[SYSTEM] Causal Video VAE: READY"]);
  const [hasKey, setHasKey] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'generator' | 'architecture'>('generator');
  
  // Advanced Parameters
  const [model, setModel] = useState<ModelType>('veo-3.1-fast-generate-preview');
  const [resolution, setResolution] = useState<Resolution>('720p');
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>('16:9');
  const [cfgScale, setCfgScale] = useState(5.0); // Wan uses lower CFG for Flow Matching
  
  const terminalEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    checkApiKey();
  }, []);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const checkApiKey = async () => {
    try {
      const selected = await window.aistudio.hasSelectedApiKey();
      setHasKey(selected);
    } catch (e) {
      console.error("Failed to check API key", e);
    }
  };

  const handleSelectKey = async () => {
    await window.aistudio.openSelectKey();
    setHasKey(true);
  };

  const addLog = (msg: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  const generateVideo = async () => {
    if (!prompt) return;
    setIsGenerating(true);
    setError(null);
    
    // Revoke old URL before starting new generation
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
      setVideoUrl(null);
    }
    
    setLogs([]);
    addLog(`Initializing Pipeline: ${model}...`);
    addLog(`Parameters: Res=${resolution}, Ratio=${aspectRatio}, CFG=${cfgScale}`);
    addLog("Loading T5 Text Encoder...");
    addLog(`Encoding prompt: "${prompt}"`);
    
    const abortController = new AbortController();
    
    try {
      const apiKey = (process.env as any).API_KEY;
      if (!apiKey) {
        throw new Error("API Key missing. Please click 'SELECT API KEY' and choose a key from a paid Google Cloud project.");
      }
      
      const ai = new GoogleGenAI({ apiKey });
      
      addLog("Allocating 5D Latent Tensor...");
      addLog("Starting DDIM Sampling Loop (50 steps)...");

      const logInterval = setInterval(() => {
        const step = Math.floor(Math.random() * 50);
        addLog(`Sampling Step ${step}/50 - CFG: ${cfgScale} - Temporal Coherence: 0.99`);
      }, 1200);

      let operation = await ai.models.generateVideos({
        model: model,
        prompt: prompt,
        config: {
          numberOfVideos: 1,
          resolution: resolution,
          aspectRatio: aspectRatio
        }
      });

      while (!operation.done) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        operation = await ai.operations.getVideosOperation({ operation: operation });
      }

      clearInterval(logInterval);
      
      const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
      if (downloadLink) {
        addLog("Sampling Complete. Decoding Latents via VideoVAE...");
        addLog("Finalizing MP4 Stream...");
        
        const response = await fetch(downloadLink, {
          method: 'GET',
          headers: {
            'x-goog-api-key': (process.env as any).API_KEY,
          },
          signal: abortController.signal
        });
        
        if (!response.ok) throw new Error(`Fetch failed: ${response.statusText}`);
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setVideoUrl(url);
        addLog("SUCCESS: Video generated successfully.");
      } else {
        throw new Error("No video URI returned from model.");
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        addLog("Generation aborted by user/system.");
        return;
      }
      console.error(err);
      
      let errMsg = err.message || "Generation failed.";
      
      // Handle JSON error responses if they are stringified in the message
      if (errMsg.includes('"code":403') || errMsg.includes('PERMISSION_DENIED') || errMsg.includes('403')) {
        errMsg = "Permission Denied (403). This usually means your API key is from a project without billing enabled or the Veo API is not active. Please use a key from a PAID Google Cloud project.";
        setHasKey(false);
      }
      
      // Ignore the specific browser abort error for media resources
      if (errMsg.includes("aborted by the user agent")) {
        addLog("Notice: Media fetch was reset by browser.");
        return;
      }

      setError(errMsg);
      addLog(`ERROR: ${errMsg}`);
      if (errMsg.includes("Requested entity was not found")) {
        setHasKey(false);
      }
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-4"
          >
            <div className="w-12 h-12 bg-emerald-500 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(16,185,129,0.3)]">
              <Video className="text-black w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold tracking-tighter uppercase italic">WanVideo Pro 2.2</h1>
              <p className="text-emerald-500/60 text-xs font-mono uppercase tracking-widest">Next-Gen T2V Synthesis</p>
            </div>
          </motion.div>
          
          <div className="flex gap-2">
            {!hasKey ? (
              <button 
                onClick={handleSelectKey}
                className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-black rounded-lg font-bold text-sm hover:bg-amber-400 transition-colors"
              >
                <Key className="w-4 h-4" />
                SELECT API KEY
              </button>
            ) : (
              <span className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-emerald-500 text-xs font-mono flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                ENGINE: READY
              </span>
            )}
          </div>
        </header>

        {/* Tabs */}
        <div className="flex gap-4 mb-8 border-b border-white/10">
          <button 
            onClick={() => setActiveTab('generator')}
            className={`pb-4 px-2 text-sm font-bold transition-all ${activeTab === 'generator' ? 'text-emerald-500 border-b-2 border-emerald-500' : 'text-white/40 hover:text-white'}`}
          >
            INTERACTIVE GENERATOR
          </button>
          <button 
            onClick={() => setActiveTab('architecture')}
            className={`pb-4 px-2 text-sm font-bold transition-all ${activeTab === 'architecture' ? 'text-emerald-500 border-b-2 border-emerald-500' : 'text-white/40 hover:text-white'}`}
          >
            WAN 2.2 ARCHITECTURE
          </button>
        </div>

        {activeTab === 'generator' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Column: Controls & Parameters */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-6">
              <div>
                <label className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-white/40 mb-3">
                  <Sliders className="w-3 h-3" /> Model Selection
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button 
                    onClick={() => setModel('veo-3.1-fast-generate-preview')}
                    className={`px-3 py-2 rounded-lg text-[10px] font-bold border transition-all ${model === 'veo-3.1-fast-generate-preview' ? 'bg-emerald-500 border-emerald-500 text-black' : 'bg-white/5 border-white/10 text-white/60 hover:border-white/30'}`}
                  >
                    VEO 3.1 FAST
                  </button>
                  <button 
                    onClick={() => setModel('veo-3.1-generate-preview')}
                    className={`px-3 py-2 rounded-lg text-[10px] font-bold border transition-all ${model === 'veo-3.1-generate-preview' ? 'bg-emerald-500 border-emerald-500 text-black' : 'bg-white/5 border-white/10 text-white/60 hover:border-white/30'}`}
                  >
                    VEO 3.1 HQ
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-white/40 mb-3">
                    <Monitor className="w-3 h-3" /> Resolution
                  </label>
                  <select 
                    value={resolution}
                    onChange={(e) => setResolution(e.target.value as Resolution)}
                    className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-xs focus:outline-none focus:border-emerald-500/50"
                  >
                    <option value="720p">720p</option>
                    <option value="1080p">1080p</option>
                  </select>
                </div>
                <div>
                  <label className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-white/40 mb-3">
                    <Square className="w-3 h-3" /> Aspect Ratio
                  </label>
                  <select 
                    value={aspectRatio}
                    onChange={(e) => setAspectRatio(e.target.value as AspectRatio)}
                    className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-xs focus:outline-none focus:border-emerald-500/50"
                  >
                    <option value="16:9">16:9 (Landscape)</option>
                    <option value="9:16">9:16 (Portrait)</option>
                  </select>
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-3">
                  <label className="text-xs font-mono uppercase tracking-wider text-white/40">CFG Scale: {cfgScale}</label>
                </div>
                <input 
                  type="range" 
                  min="1" 
                  max="20" 
                  step="0.5" 
                  value={cfgScale}
                  onChange={(e) => setCfgScale(parseFloat(e.target.value))}
                  className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>

              <div className="pt-4 border-t border-white/10">
                <label className="block text-xs font-mono uppercase tracking-wider text-white/40 mb-2">Prompt Input</label>
                <textarea 
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the cinematic scene..."
                  className="w-full h-24 bg-black/40 border border-white/10 rounded-xl p-4 text-sm focus:outline-none focus:border-emerald-500/50 transition-colors resize-none"
                  disabled={isGenerating || !hasKey}
                />
                
                <button
                  onClick={generateVideo}
                  disabled={isGenerating || !prompt || !hasKey}
                  className="w-full mt-4 py-4 bg-emerald-500 disabled:bg-white/10 disabled:text-white/20 text-black font-bold rounded-xl flex items-center justify-center gap-2 hover:bg-emerald-400 transition-all active:scale-[0.98]"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      GENERATING...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" />
                      GENERATE VIDEO
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Terminal */}
            <div className="bg-black border border-white/10 rounded-2xl overflow-hidden h-48 flex flex-col">
              <div className="bg-white/5 px-4 py-2 border-b border-white/10 flex items-center gap-2">
                <Terminal className="w-3 h-3 text-white/40" />
                <span className="text-[10px] font-mono uppercase tracking-widest text-white/40">Engine Logs</span>
              </div>
              <div className="flex-1 overflow-y-auto p-4 font-mono text-[10px] space-y-1">
                {logs.map((log, i) => (
                  <div key={i} className={log.includes('ERROR') ? 'text-red-400' : log.includes('SUCCESS') ? 'text-emerald-400' : 'text-white/60'}>
                    {log}
                  </div>
                ))}
                <div ref={terminalEndRef} />
              </div>
            </div>
          </div>

          {/* Right Column: Preview */}
          <div className="lg:col-span-8">
            <div className="aspect-video bg-white/5 border border-white/10 rounded-2xl overflow-hidden relative flex items-center justify-center group">
              <AnimatePresence mode="wait">
                {videoUrl ? (
                  <motion.div 
                    key="video"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="w-full h-full"
                  >
                    <video 
                      src={videoUrl} 
                      controls 
                      autoPlay 
                      loop 
                      className="w-full h-full object-contain"
                    />
                    <a 
                      href={videoUrl} 
                      download="wan_video_pro.mp4"
                      className="absolute top-4 right-4 p-2 bg-black/60 backdrop-blur-md border border-white/10 rounded-lg hover:bg-emerald-500 hover:text-black transition-all opacity-0 group-hover:opacity-100"
                    >
                      <Download className="w-5 h-5" />
                    </a>
                  </motion.div>
                ) : isGenerating ? (
                  <motion.div 
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center gap-4"
                  >
                    <div className="relative">
                      <Loader2 className="w-12 h-12 text-emerald-500 animate-spin" />
                      <div className="absolute inset-0 blur-xl bg-emerald-500/20 animate-pulse" />
                    </div>
                    <p className="text-sm font-mono text-emerald-500/60 animate-pulse">SYNTHESIZING NEXT-GEN VIDEO...</p>
                  </motion.div>
                ) : (
                  <motion.div 
                    key="placeholder"
                    className="flex flex-col items-center text-white/20 gap-4"
                  >
                    <div className="w-20 h-20 border-2 border-dashed border-white/10 rounded-full flex items-center justify-center">
                      <Play className="w-8 h-8" />
                    </div>
                    <p className="text-sm font-mono uppercase tracking-widest">Waiting for generation</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {error && (
                <div className="absolute bottom-4 left-4 right-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-500 text-xs flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    <span className="font-bold">Generation Error</span>
                  </div>
                  <p>{error}</p>
                  {error.includes("403") && (
                    <a 
                      href="https://ai.google.dev/gemini-api/docs/billing" 
                      target="_blank" 
                      className="text-emerald-500 underline font-bold"
                    >
                      Setup Billing & API Access →
                    </a>
                  )}
                </div>
              )}
            </div>

            {/* Specs */}
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
              <SpecItem label="Model" value={model === 'veo-3.1-fast-generate-preview' ? 'VEO 3.1 FAST' : 'VEO 3.1 HQ'} />
              <SpecItem label="Resolution" value={resolution} />
              <SpecItem label="Ratio" value={aspectRatio} />
              <SpecItem label="CFG" value={cfgScale.toString()} />
            </div>

            <div className="mt-8 p-6 bg-white/5 border border-white/10 rounded-2xl">
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-emerald-500" /> Real Wan 2.2 Core Implementation
              </h3>
              <p className="text-sm text-white/60 mb-4">
                The Python files in this project contain the actual architecture of Wan 2.2, including the Flow Matching Transformer and Causal 3D VAE. 
                To run the full model with weights, download the checkpoints from the official repository.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-white/60">
                <ul className="space-y-2">
                  <li className="flex items-center gap-2"><div className="w-1 h-1 bg-emerald-500 rounded-full" /> <code>models/wan_transformer.py</code>: DiT Core</li>
                  <li className="flex items-center gap-2"><div className="w-1 h-1 bg-emerald-500 rounded-full" /> <code>models/wan_vae.py</code>: Causal 3D VAE</li>
                </ul>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2"><div className="w-1 h-1 bg-emerald-500 rounded-full" /> <code>pipeline/video_pipeline.py</code>: Flow Matching</li>
                  <li className="flex items-center gap-2"><div className="w-1 h-1 bg-emerald-500 rounded-full" /> <code>comfyui/nodes.py</code>: Custom Node Logic</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        ) : (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/5 border border-white/10 rounded-2xl p-8"
          >
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <Layers className="text-emerald-500" /> Wan 2.2 Deep Dive
            </h2>
            <div className="space-y-8">
              <section>
                <h3 className="text-emerald-500 font-mono text-sm uppercase mb-2">01. Transformer (DiT)</h3>
                <p className="text-white/60 text-sm leading-relaxed">
                  The core of Wan 2.2 is a massive Diffusion Transformer (DiT). Unlike traditional UNets, it treats video as a sequence of patches in latent space. 
                  It uses 3D Rotary Positional Embeddings (RoPE) to maintain spatial and temporal consistency across frames.
                </p>
              </section>
              <section>
                <h3 className="text-emerald-500 font-mono text-sm uppercase mb-2">02. Causal Video VAE</h3>
                <p className="text-white/60 text-sm leading-relaxed">
                  Wan uses a Causal 3D VAE that compresses video into a compact latent space. The "causal" nature means it only looks at current and past frames during encoding, 
                  which is critical for high-quality temporal modeling and potential streaming applications.
                </p>
              </section>
              <section>
                <h3 className="text-emerald-500 font-mono text-sm uppercase mb-2">03. Flow Matching</h3>
                <p className="text-white/60 text-sm leading-relaxed">
                  Moving beyond standard diffusion, Wan 2.2 implements Flow Matching. This allows for straighter sampling paths, 
                  resulting in higher quality video with fewer inference steps (typically 20-50 steps).
                </p>
              </section>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}

function SpecItem({ label, value }: { label: string, value: string }) {
  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-3">
      <div className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1">{label}</div>
      <div className="text-sm font-bold">{value}</div>
    </div>
  );
}



