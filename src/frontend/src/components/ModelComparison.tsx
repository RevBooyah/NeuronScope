import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { apiService } from '../services/api';

const ModelComparison: React.FC = () => {
  const [models, setModels] = useState<string[]>([]);
  const [modelA, setModelA] = useState<string>('gpt2');
  const [modelB, setModelB] = useState<string>('gpt2-medium');
  const [prompts, setPrompts] = useState<string[]>([]);
  const [layerIndex, setLayerIndex] = useState<number>(0);
  const [result, setResult] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const modelInfo = await apiService.getAvailableModels();
      const m = Object.keys(modelInfo.models || {});
      setModels(m);
      const s = await apiService.getSamplePrompts();
      setPrompts(s);
    })();
  }, []);

  const runComparison = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    const data = await apiService.compareModels({ modelA, modelB, prompts });
    if (data) {
      setResult(data);
      setLayerIndex(0);
    } else {
      setError('Comparison failed');
    }
    setLoading(false);
  };

  const similarityForLayer = () => {
    if (!result || !result.layers || result.layers.length === 0) return null;
    const layer = result.layers[Math.min(layerIndex, result.layers.length - 1)];
    return layer?.similarity || null;
  };

  const heatmapData = () => {
    const sim = similarityForLayer();
    if (!sim) return [];
    return [{
      z: sim,
      type: 'heatmap' as const,
      colorscale: 'Viridis',
      name: 'Cosine similarity'
    }];
  };

  return (
    <div className="model-comparison">
      <div className="controls" style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <div>
          <label>Model A</label>
          <select value={modelA} onChange={(e) => setModelA(e.target.value)}>
            {models.map(m => (<option key={m} value={m}>{m}</option>))}
          </select>
        </div>
        <div>
          <label>Model B</label>
          <select value={modelB} onChange={(e) => setModelB(e.target.value)}>
            {models.map(m => (<option key={m} value={m}>{m}</option>))}
          </select>
        </div>
        <div>
          <label>Layer</label>
          <input type="number" min={0} value={layerIndex} onChange={(e) => setLayerIndex(parseInt(e.target.value || '0'))} />
        </div>
        <button onClick={runComparison} disabled={loading}>Compare</button>
      </div>

      {loading && <div style={{ marginTop: 16 }}>Running comparison...</div>}
      {error && <div style={{ color: 'red', marginTop: 16 }}>{error}</div>}

      {result && (
        <div style={{ marginTop: 16 }}>
          <h3>Similarity Heatmap (Layer {Math.min(layerIndex, result.layers.length - 1)})</h3>
          <Plot data={heatmapData()} layout={{ autosize: true, height: 500, title: `${result.models.a} vs ${result.models.b}` }} style={{ width: '100%' }} />
        </div>
      )}
    </div>
  );
};

export default ModelComparison;


