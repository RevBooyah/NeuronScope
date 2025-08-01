import React, { useState } from 'react';
import { useActivationData } from '../hooks/useActivationData';
import { ActivationVisualization } from './ActivationVisualization';
import ModelInfoModal from './ModelInfoModal';
import './Dashboard.css';

interface DashboardProps {
  // Add props as needed
}

const Dashboard: React.FC<DashboardProps> = () => {
  const [selectedPrompt, setSelectedPrompt] = useState<string>('');
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [visualizationType, setVisualizationType] = useState<'heatmap' | 'scatter' | 'summary'>('heatmap');
  const [showModelInfo, setShowModelInfo] = useState<boolean>(false);

  const {
    activationData,
    loading,
    error,
    availableFiles,
    samplePrompts,
    loadActivationData,
    generateNewActivation
  } = useActivationData();

  const handlePromptChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const prompt = event.target.value;
    setSelectedPrompt(prompt);
    if (prompt) {
      generateNewActivation(prompt);
    }
  };

  const handleLayerChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedLayer(parseInt(event.target.value));
  };

  const handleVisualizationChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setVisualizationType(event.target.value as 'heatmap' | 'scatter' | 'summary');
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const filename = event.target.value;
    if (filename) {
      loadActivationData(filename);
    }
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="header-content">
          <div className="header-text">
            <h1>🧠 NeuronScope</h1>
            <p>Interactive visualization platform for exploring neuron activations in transformer models</p>
          </div>
          <button 
            className="model-info-button"
            onClick={() => setShowModelInfo(true)}
            title="View model information"
          >
            📋 Model Info
          </button>
        </div>
      </header>

      <div className="dashboard-content">
        <div className="control-panel">
          <div className="control-group">
            <label htmlFor="file-select">Select Data File:</label>
            <select 
              id="file-select" 
              onChange={handleFileChange}
              className="control-select"
            >
              <option value="">Choose a file...</option>
              {availableFiles.map((filename, index) => (
                <option key={index} value={filename}>{filename}</option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="prompt-select">Or Select Prompt:</label>
            <select 
              id="prompt-select" 
              value={selectedPrompt} 
              onChange={handlePromptChange}
              className="control-select"
            >
              <option value="">Choose a prompt...</option>
              {samplePrompts.map((prompt, index) => (
                <option key={index} value={prompt}>{prompt}</option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="layer-select">Select Layer:</label>
            <select 
              id="layer-select" 
              value={selectedLayer} 
              onChange={handleLayerChange}
              className="control-select"
              disabled={!activationData}
            >
              {activationData ? Array.from({ length: activationData.layers.length }, (_, i) => (
                <option key={i} value={i}>Layer {i}</option>
              )) : (
                <option value="">No data loaded</option>
              )}
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="viz-select">Visualization Type:</label>
            <select 
              id="viz-select" 
              value={visualizationType} 
              onChange={handleVisualizationChange}
              className="control-select"
              disabled={!activationData}
            >
              <option value="heatmap">Heatmap</option>
              <option value="scatter">Scatter Plot (PCA/t-SNE)</option>
              <option value="summary">Activation Summary</option>
            </select>
          </div>

          {loading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <span>Loading...</span>
            </div>
          )}

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        <div className="visualization-area">
          {activationData ? (
            <ActivationVisualization
              activationData={activationData}
              visualizationType={visualizationType}
              layerIndex={selectedLayer}
            />
          ) : (
            <div className="visualization-placeholder">
              <h3>Visualization Area</h3>
              <p>Select a data file or prompt to load activation data</p>
              <div className="placeholder-content">
                <div className="placeholder-icon">📊</div>
                <p>Interactive visualizations will appear here</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="dashboard-footer">
        <div className="status-info">
          <span className="status-item">✅ Backend: Connected</span>
          <span className="status-item">📊 Visualizations: Ready</span>
          <span className="status-item">🧠 Model: GPT-2</span>
        </div>
      </div>

      <ModelInfoModal
        isOpen={showModelInfo}
        onClose={() => setShowModelInfo(false)}
        modelName="gpt2"
      />
    </div>
  );
};

export default Dashboard; 