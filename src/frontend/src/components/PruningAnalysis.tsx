import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import './PruningAnalysis.css';

interface WeightAnalysis {
  overall_sparsity: number;
  total_parameters: number;
  non_zero_parameters: number;
  layer_sparsity: Record<string, any>;
  layer_stats: Array<{
    layer_name: string;
    layer_index: number;
    total_parameters: number;
    non_zero_parameters: number;
    sparsity: number;
    mean_magnitude: number;
    std_magnitude: number;
    min_magnitude: number;
    max_magnitude: number;
    l1_norm: number;
    l2_norm: number;
  }>;
}

interface PruningCandidate {
  layer_index: number;
  neuron_index: number;
  weight_magnitude: number;
  weight_rank: number;
  is_pruning_candidate: boolean;
  pruning_score: number;
}

interface PruningImpact {
  impact_score: number;
  mean_change: number;
  max_change: number;
  affected_neurons: number[];
  safe_to_prune: boolean;
}

const PruningAnalysis: React.FC = () => {
  const [weightAnalysis, setWeightAnalysis] = useState<WeightAnalysis | null>(null);
  const [pruningCandidates, setPruningCandidates] = useState<PruningCandidate[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [threshold, setThreshold] = useState<number>(10.0);
  const [loading, setLoading] = useState<boolean>(false);
  const [impactResults, setImpactResults] = useState<PruningImpact | null>(null);
  const [testInput, setTestInput] = useState<string>('Hello world, this is a test for pruning analysis.');
  const [cacheStats, setCacheStats] = useState<any>(null);

  // Fetch weight analysis
  const fetchWeightAnalysis = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5001/api/pruning/weight-analysis');
      if (response.ok) {
        const data = await response.json();
        setWeightAnalysis(data);
      }
    } catch (error) {
      console.error('Error fetching weight analysis:', error);
    }
    setLoading(false);
  };

  // Fetch pruning candidates
  const fetchPruningCandidates = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:5001/api/pruning/candidates?threshold=${threshold}`);
      if (response.ok) {
        const data = await response.json();
        setPruningCandidates(data.candidates || []);
      }
    } catch (error) {
      console.error('Error fetching pruning candidates:', error);
    }
    setLoading(false);
  };

  // Test pruning impact
  const testPruningImpact = async () => {
    if (pruningCandidates.length === 0) return;
    
    setLoading(true);
    try {
      const testCandidates = pruningCandidates
        .filter(c => c.layer_index === selectedLayer)
        .slice(0, 5); // Test first 5 candidates
      
      const response = await fetch('http://localhost:5001/api/pruning/impact-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          layer_index: selectedLayer,
          neuron_indices: testCandidates.map(c => c.neuron_index),
          input_text: testInput
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setImpactResults(data);
      }
    } catch (error) {
      console.error('Error testing pruning impact:', error);
    }
    setLoading(false);
  };

  // Fetch cache stats
  const fetchCacheStats = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/cache/stats');
      if (response.ok) {
        const data = await response.json();
        setCacheStats(data);
      }
    } catch (error) {
      console.error('Error fetching cache stats:', error);
    }
  };

  // Clear cache
  const clearCache = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/cache/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        const data = await response.json();
        setCacheStats(data.stats);
        alert('Cache cleared successfully!');
      }
    } catch (error) {
      console.error('Error clearing cache:', error);
    }
  };

  useEffect(() => {
    fetchWeightAnalysis();
    fetchCacheStats();
  }, []);

  useEffect(() => {
    fetchPruningCandidates();
  }, [threshold]);

  // Prepare data for visualizations
  const prepareLayerSparsityData = () => {
    if (!weightAnalysis) return null;
    
    const layers = Object.keys(weightAnalysis.layer_sparsity);
    const sparsityValues = layers.map(layer => weightAnalysis.layer_sparsity[layer].sparsity);
    const layerIndices = layers.map(layer => {
      const match = layer.match(/transformer\.h\.(\d+)/);
      return match ? parseInt(match[1]) : 0;
    });
    
    return {
      x: layerIndices,
      y: sparsityValues,
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: 'Layer Sparsity',
      line: { color: '#6366f1', width: 3 },
      marker: { size: 8, color: '#6366f1' }
    };
  };

  const prepareWeightMagnitudeHeatmap = () => {
    if (!weightAnalysis?.layer_stats) return null;
    
    const layers = weightAnalysis.layer_stats.map(stat => stat.layer_index);
    const magnitudes = weightAnalysis.layer_stats.map(stat => stat.mean_magnitude);
    const stds = weightAnalysis.layer_stats.map(stat => stat.std_magnitude);
    
    return {
      z: [magnitudes, stds],
      x: layers,
      y: ['Mean Magnitude', 'Std Magnitude'],
      type: 'heatmap' as const,
      colorscale: 'Viridis',
      name: 'Weight Magnitudes'
    };
  };

  const preparePruningCandidatesScatter = () => {
    if (pruningCandidates.length === 0) return null;
    
    const layerGroups = pruningCandidates.reduce((acc, candidate) => {
      if (!acc[candidate.layer_index]) {
        acc[candidate.layer_index] = [];
      }
      acc[candidate.layer_index].push(candidate);
      return acc;
    }, {} as Record<number, PruningCandidate[]>);
    
    const traces = Object.entries(layerGroups).map(([layerIndex, candidates]) => ({
      x: candidates.map(c => c.neuron_index),
      y: candidates.map(c => c.weight_magnitude),
      mode: 'markers' as const,
      type: 'scatter' as const,
      name: `Layer ${layerIndex}`,
      marker: {
        size: candidates.map(c => c.is_pruning_candidate ? 10 : 6),
        color: candidates.map(c => c.is_pruning_candidate ? '#ef4444' : '#3b82f6'),
        opacity: 0.7
      },
      text: candidates.map(c => 
        `Neuron ${c.neuron_index}<br>Magnitude: ${c.weight_magnitude.toFixed(4)}<br>Score: ${c.pruning_score.toFixed(3)}`
      ),
      hoverinfo: 'text'
    }));
    
    return traces;
  };

  const prepareImpactAnalysisChart = () => {
    if (!impactResults) return null;
    
    return {
      values: [impactResults.impact_score, 1 - impactResults.impact_score],
      labels: ['Impact Score', 'Remaining Performance'],
      type: 'pie' as const,
      name: 'Pruning Impact',
      marker: {
        colors: impactResults.safe_to_prune ? ['#10b981', '#d1fae5'] : ['#ef4444', '#fee2e2']
      },
      textinfo: 'label+percent',
      hole: 0.4
    };
  };

  return (
    <div className="pruning-analysis">
      <div className="pruning-header">
        <h2>🧠 Pruning Analysis Dashboard</h2>
        <div className="pruning-controls">
          <div className="control-group">
            <label>Threshold (%):</label>
            <input
              type="range"
              min="1"
              max="50"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
            />
            <span>{threshold}%</span>
          </div>
          <div className="control-group">
            <label>Test Layer:</label>
            <select value={selectedLayer} onChange={(e) => setSelectedLayer(parseInt(e.target.value))}>
              {weightAnalysis?.layer_stats.map(stat => (
                <option key={stat.layer_index} value={stat.layer_index}>
                  Layer {stat.layer_index}
                </option>
              ))}
            </select>
          </div>
          <button 
            className="refresh-btn"
            onClick={fetchWeightAnalysis}
            disabled={loading}
          >
            🔄 Refresh Analysis
          </button>
          <button 
            className="clear-cache-btn"
            onClick={clearCache}
            title="Clear cache to force fresh analysis"
          >
            🗑️ Clear Cache
          </button>
        </div>
      </div>

      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Analyzing model weights...</p>
        </div>
      )}

      <div className="pruning-grid">
        {/* Overall Statistics */}
        <div className="stats-card">
          <h3>📊 Model Statistics</h3>
          {cacheStats && (
            <div className="cache-info">
              <span className="cache-stat">Cache: {cacheStats.total_entries} entries ({cacheStats.total_size_mb.toFixed(2)} MB)</span>
            </div>
          )}
          {weightAnalysis && (
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Total Parameters</span>
                <span className="stat-value">{weightAnalysis.total_parameters.toLocaleString()}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Non-Zero Parameters</span>
                <span className="stat-value">{weightAnalysis.non_zero_parameters.toLocaleString()}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Overall Sparsity</span>
                <span className="stat-value">{(weightAnalysis.overall_sparsity * 100).toFixed(2)}%</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Pruning Candidates</span>
                <span className="stat-value">{pruningCandidates.filter(c => c.is_pruning_candidate).length}</span>
              </div>
            </div>
          )}
        </div>

        {/* Layer Sparsity Chart */}
        <div className="chart-card">
          <h3>📈 Layer Sparsity Analysis</h3>
          {prepareLayerSparsityData() && (
            <Plot
              data={[prepareLayerSparsityData()!]}
              layout={{
                title: { text: 'Sparsity by Layer' },
                xaxis: { title: { text: 'Layer Index' } },
                yaxis: { title: { text: 'Sparsity' }, tickformat: ',.0%' },
                height: 300,
                margin: { t: 40, b: 40, l: 60, r: 20 }
              }}
              config={{ displayModeBar: false }}
            />
          )}
        </div>

        {/* Weight Magnitude Heatmap */}
        <div className="chart-card">
          <h3>🔥 Weight Magnitude Heatmap</h3>
          {prepareWeightMagnitudeHeatmap() && (
            <Plot
              data={[prepareWeightMagnitudeHeatmap()!]}
              layout={{
                title: { text: 'Weight Magnitudes Across Layers' },
                xaxis: { title: { text: 'Layer Index' } },
                yaxis: { title: { text: 'Magnitude Type' } },
                height: 300,
                margin: { t: 40, b: 40, l: 60, r: 20 }
              }}
              config={{ displayModeBar: false }}
            />
          )}
        </div>

        {/* Pruning Candidates Scatter */}
        <div className="chart-card wide">
          <h3>🎯 Pruning Candidates Analysis</h3>
          <div className="candidates-controls">
            <input
              type="text"
              value={testInput}
              onChange={(e) => setTestInput(e.target.value)}
              placeholder="Enter test input for impact analysis..."
              className="test-input"
            />
            <button 
              onClick={testPruningImpact}
              disabled={loading || pruningCandidates.length === 0}
              className="test-btn"
            >
              🧪 Test Impact
            </button>
          </div>
          {preparePruningCandidatesScatter() && (
            <Plot
              data={preparePruningCandidatesScatter()!}
              layout={{
                title: { text: 'Neuron Weight Magnitudes by Layer' },
                xaxis: { title: { text: 'Neuron Index' } },
                yaxis: { title: { text: 'Weight Magnitude' } },
                height: 400,
                margin: { t: 40, b: 40, l: 60, r: 20 },
                showlegend: true
              }}
              config={{ displayModeBar: false }}
            />
          )}
          <div className="legend">
            <span className="legend-item">
              <span className="legend-color safe"></span>
              Safe to Prune
            </span>
            <span className="legend-item">
              <span className="legend-color risky"></span>
              Risky to Prune
            </span>
          </div>
        </div>

        {/* Impact Analysis */}
        {impactResults && (
          <div className="chart-card">
            <h3>⚡ Pruning Impact Analysis</h3>
            <Plot
              data={[prepareImpactAnalysisChart()!]}
              layout={{
                title: { text: 'Impact Score Distribution' },
                height: 300,
                margin: { t: 40, b: 40, l: 20, r: 20 }
              }}
              config={{ displayModeBar: false }}
            />
            <div className="impact-details">
              <div className="impact-stat">
                <span>Impact Score:</span>
                <span className={impactResults.safe_to_prune ? 'safe' : 'risky'}>
                  {(impactResults.impact_score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="impact-stat">
                <span>Mean Change:</span>
                <span>{impactResults.mean_change.toFixed(4)}</span>
              </div>
              <div className="impact-stat">
                <span>Max Change:</span>
                <span>{impactResults.max_change.toFixed(4)}</span>
              </div>
              <div className="impact-stat">
                <span>Status:</span>
                <span className={impactResults.safe_to_prune ? 'safe' : 'risky'}>
                  {impactResults.safe_to_prune ? '✅ Safe to Prune' : '⚠️ Risky to Prune'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PruningAnalysis; 