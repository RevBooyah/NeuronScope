import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import './ModelSelector.css';

interface ModelInfo {
  name: string;
  family: string;
  layers: number;
  hidden_size: number;
  num_attention_heads: number;
  description: string;
  recommended: boolean;
  size_category: string;
  requires_auth?: boolean;
  loaded?: boolean;
}

interface ModelSelectorProps {
  onModelChange?: (modelName: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ onModelChange }) => {
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [recommendedModels, setRecommendedModels] = useState<Record<string, ModelInfo>>({});
  const [currentModel, setCurrentModel] = useState<string>('gpt2');
  const [selectedModel, setSelectedModel] = useState<string>('gpt2');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [memoryUsage, setMemoryUsage] = useState<any>({});
  const [showAllModels, setShowAllModels] = useState<boolean>(false);

  useEffect(() => {
    loadModels();
    loadMemoryUsage();
  }, []);

  const loadModels = async () => {
    try {
      const data = await apiService.getAvailableModels();
      setModels(data.models || {});
      setRecommendedModels(data.recommended || {});
      setCurrentModel(data.current_model || 'gpt2');
      setSelectedModel(data.current_model || 'gpt2');
    } catch (err) {
      setError('Failed to load models');
      console.error('Failed to load models:', err);
    }
  };

  const loadMemoryUsage = async () => {
    try {
      const usage = await apiService.getMemoryUsage();
      setMemoryUsage(usage);
    } catch (err) {
      console.error('Failed to load memory usage:', err);
    }
  };

  const handleModelSwitch = async () => {
    if (selectedModel === currentModel) return;

    setLoading(true);
    setError(null);

    try {
      const result = await apiService.switchModel(selectedModel);
      if (result.success) {
        setCurrentModel(selectedModel);
        onModelChange?.(selectedModel);
        await loadMemoryUsage(); // Refresh memory usage after switch
      } else {
        // Show detailed error message
        let errorMessage = result.error || 'Failed to switch model';
        
        if (result.requiresAuth) {
          errorMessage += '\n\nThis model requires Hugging Face authentication. Please set the HF_TOKEN environment variable or login with huggingface-cli.';
        }
        
        if (result.details) {
          errorMessage += `\n\nDetails: ${result.details}`;
        }
        
        setError(errorMessage);
      }
    } catch (err) {
      setError('Failed to switch model');
      console.error('Failed to switch model:', err);
    } finally {
      setLoading(false);
    }
  };

  const getModelDisplayName = (modelName: string, modelInfo: ModelInfo) => {
    const shortName = modelName.split('/').pop() || modelName;
    return `${shortName} (${modelInfo.description})`;
  };

  const getSizeCategoryColor = (category: string) => {
    switch (category) {
      case 'tiny': return '#28a745';
      case 'small': return '#17a2b8';
      case 'medium': return '#ffc107';
      case 'large': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const renderModelOption = (modelName: string, modelInfo: ModelInfo) => (
    <option key={modelName} value={modelName}>
      {getModelDisplayName(modelName, modelInfo)}
    </option>
  );

  const modelsToShow = showAllModels ? models : recommendedModels;

  return (
    <div className="model-selector">
      <div className="model-selector-header">
        <h3>🧠 Model Selection</h3>
        <div className="model-selector-controls">
          <button
            className="toggle-models-btn"
            onClick={() => setShowAllModels(!showAllModels)}
          >
            {showAllModels ? 'Show Recommended' : 'Show All Models'}
          </button>
        </div>
      </div>

      <div className="model-selector-content">
        <div className="model-select-group">
          <label htmlFor="model-select">Select Model:</label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="model-select"
            disabled={loading}
          >
            {Object.entries(modelsToShow).map(([modelName, modelInfo]) =>
              renderModelOption(modelName, modelInfo)
            )}
          </select>
        </div>

        {selectedModel && models[selectedModel] && (
          <div className="model-info">
            <div className="model-info-header">
              <span className="model-name">
                {getModelDisplayName(selectedModel, models[selectedModel])}
              </span>
              <span 
                className="size-badge"
                style={{ backgroundColor: getSizeCategoryColor(models[selectedModel].size_category) }}
              >
                {models[selectedModel].size_category.toUpperCase()}
              </span>
            </div>
            
            <div className="model-details">
              <div className="detail-item">
                <span className="detail-label">Family:</span>
                <span className="detail-value">{models[selectedModel].family}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Layers:</span>
                <span className="detail-value">{models[selectedModel].layers}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Hidden Size:</span>
                <span className="detail-value">{models[selectedModel].hidden_size}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Attention Heads:</span>
                <span className="detail-value">{models[selectedModel].num_attention_heads}</span>
              </div>
              {models[selectedModel].requires_auth && (
                <div className="detail-item">
                  <span className="detail-label">Auth Required:</span>
                  <span className="detail-value auth-warning">Yes (HF Token)</span>
                </div>
              )}
            </div>
          </div>
        )}

        {memoryUsage.gpu_memory_total && (
          <div className="memory-info">
            <h4>💾 Memory Usage</h4>
            <div className="memory-details">
              <div className="memory-item">
                <span>Total:</span>
                <span>{memoryUsage.gpu_memory_total}</span>
              </div>
              <div className="memory-item">
                <span>Used:</span>
                <span>{memoryUsage.gpu_memory_allocated}</span>
              </div>
              <div className="memory-item">
                <span>Free:</span>
                <span>{memoryUsage.gpu_memory_free}</span>
              </div>
            </div>
          </div>
        )}

        <div className="model-actions">
          <button
            className="switch-model-btn"
            onClick={handleModelSwitch}
            disabled={loading || selectedModel === currentModel}
          >
            {loading ? 'Switching...' : selectedModel === currentModel ? 'Current Model' : 'Switch Model'}
          </button>
          
          {selectedModel !== currentModel && (
            <div className="switch-warning">
              ⚠️ Switching models will reset current activations and may take time to load
            </div>
          )}
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelSelector; 