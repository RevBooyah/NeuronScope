import React, { useState, useEffect } from 'react';
import './ModelInfoModal.css';

interface ModelInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelName?: string;
}

const ModelInfoModal: React.FC<ModelInfoModalProps> = ({ 
  isOpen, 
  onClose, 
  modelName = 'gpt2' 
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      // Simulate loading
      setTimeout(() => setLoading(false), 500);
    }
  }, [isOpen]);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'architecture', label: 'Architecture', icon: '🏗️' },
    { id: 'equations', label: 'Equations', icon: '📐' },
    { id: 'training', label: 'Training', icon: '🎯' },
    { id: 'usage', label: 'Usage', icon: '📋' }
  ];

  const renderOverview = () => (
    <div className="tab-content">
      <div className="info-grid">
        <div className="info-card">
          <h3>Basic Information</h3>
          <div className="info-item">
            <strong>Model:</strong> GPT-2 Small (124M parameters)
          </div>
          <div className="info-item">
            <strong>Parameters:</strong> 124,439,808
          </div>
          <div className="info-item">
            <strong>Layers:</strong> 12
          </div>
          <div className="info-item">
            <strong>Hidden Size:</strong> 768
          </div>
          <div className="info-item">
            <strong>Attention Heads:</strong> 12
          </div>
        </div>

        <div className="info-card">
          <h3>Architecture Overview</h3>
          <div className="info-item">
            <strong>Type:</strong> GPT-2 (Generative Pre-trained Transformer 2)
          </div>
          <div className="info-item">
            <strong>Architecture:</strong> Transformer Decoder-Only
          </div>
          <div className="info-item">
            <strong>Activation:</strong> GELU (Gaussian Error Linear Unit)
          </div>
          <div className="info-item">
            <strong>Paper:</strong> Language Models are Unsupervised Multitask Learners
          </div>
        </div>

        <div className="info-card">
          <h3>Performance</h3>
          <div className="info-item">
            <strong>Download Size:</strong> ~500MB
          </div>
          <div className="info-item">
            <strong>Memory Usage:</strong> ~500MB RAM
          </div>
          <div className="info-item">
            <strong>Inference Speed:</strong> ~50ms per token (CPU)
          </div>
        </div>

        <div className="info-card">
          <h3>Parameter Distribution</h3>
          <div className="info-item">
            <strong>Embeddings:</strong> 38,731,776 (31.1%)
          </div>
          <div className="info-item">
            <strong>Transformer Layers:</strong> 85,056,000 (68.4%)
          </div>
          <div className="info-item">
            <strong>Language Model Head:</strong> 651,264 (0.5%)
          </div>
        </div>
      </div>
    </div>
  );

  const renderArchitecture = () => (
    <div className="tab-content">
      <div className="info-section">
        <h3>Model Dimensions</h3>
        <div className="info-grid">
          <div className="info-item">
            <strong>Embedding Dimension:</strong> 768
          </div>
          <div className="info-item">
            <strong>Attention Heads:</strong> 12
          </div>
          <div className="info-item">
            <strong>Head Dimension:</strong> 64
          </div>
          <div className="info-item">
            <strong>MLP Intermediate Size:</strong> 3072
          </div>
          <div className="info-item">
            <strong>Total Layers:</strong> 12
          </div>
          <div className="info-item">
            <strong>Vocabulary Size:</strong> 50,257
          </div>
          <div className="info-item">
            <strong>Max Sequence Length:</strong> 1024
          </div>
        </div>
      </div>

      <div className="info-section">
        <h3>Layer Structure</h3>
        <div className="info-card">
          <h4>Each Transformer Block Contains:</h4>
          <ul>
            <li>Layer Normalization</li>
            <li>Multi-Head Self-Attention</li>
            <li>Residual Connection</li>
            <li>Layer Normalization</li>
            <li>Feed-Forward Network (MLP)</li>
            <li>Residual Connection</li>
          </ul>
        </div>
      </div>
    </div>
  );

  const renderEquations = () => (
    <div className="tab-content">
      <div className="equations-section">
        <h3>Attention Mechanism</h3>
        <div className="equation-block">
          <div className="equation-item">
            <strong>Query, Key, Value:</strong>
            <code>Q = XW_q, K = XW_k, V = XW_v</code>
          </div>
          <div className="equation-item">
            <strong>Attention Scores:</strong>
            <code>Attention(Q,K,V) = softmax(QK^T/√d_k)V</code>
          </div>
          <div className="equation-item">
            <strong>Multi-Head:</strong>
            <code>MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O</code>
          </div>
        </div>

        <h3>Feed-Forward Network</h3>
        <div className="equation-block">
          <div className="equation-item">
            <strong>MLP:</strong>
            <code>FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2</code>
          </div>
          <div className="equation-item">
            <strong>GELU Activation:</strong>
            <code>GELU(x) = x * Φ(x) where Φ is the CDF of N(0,1)</code>
          </div>
        </div>

        <h3>Layer Normalization</h3>
        <div className="equation-block">
          <div className="equation-item">
            <strong>Normalization:</strong>
            <code>LayerNorm(x) = γ * (x - μ)/√(σ² + ε) + β</code>
          </div>
        </div>

        <h3>Residual Connections</h3>
        <div className="equation-block">
          <div className="equation-item">
            <strong>Attention Residual:</strong>
            <code>x' = x + MultiHeadAttention(LayerNorm(x))</code>
          </div>
          <div className="equation-item">
            <strong>MLP Residual:</strong>
            <code>x'' = x' + FFN(LayerNorm(x'))</code>
          </div>
        </div>
      </div>
    </div>
  );

  const renderTraining = () => (
    <div className="tab-content">
      <div className="info-grid">
        <div className="info-card">
          <h3>Dataset</h3>
          <div className="info-item">
            <strong>Name:</strong> WebText
          </div>
          <div className="info-item">
            <strong>Size:</strong> ~40GB of text data
          </div>
          <div className="info-item">
            <strong>Source:</strong> Reddit posts with &gt;3 karma
          </div>
          <div className="info-item">
            <strong>Vocabulary:</strong> 50,257 tokens
          </div>
        </div>

        <div className="info-card">
          <h3>Training Objective</h3>
          <div className="info-item">
            <strong>Loss Function:</strong> Cross-entropy loss
          </div>
          <div className="info-item">
            <strong>Objective:</strong> Next token prediction (causal language modeling)
          </div>
          <div className="info-item">
            <strong>Formula:</strong> L = -∑(t=1 to T) log P(x<sub>t</sub> | x<sub>&lt;t</sub>)
          </div>
        </div>

        <div className="info-card">
          <h3>Optimization</h3>
          <div className="info-item">
            <strong>Optimizer:</strong> Adam
          </div>
          <div className="info-item">
            <strong>Learning Rate:</strong> 2.5e-4
          </div>
          <div className="info-item">
            <strong>Scheduling:</strong> Cosine annealing
          </div>
          <div className="info-item">
            <strong>Batch Size:</strong> 512 sequences
          </div>
          <div className="info-item">
            <strong>Context Length:</strong> 1024 tokens
          </div>
        </div>

        <div className="info-card">
          <h3>Hardware</h3>
          <div className="info-item">
            <strong>GPUs:</strong> 8 P100 GPUs
          </div>
          <div className="info-item">
            <strong>Memory per GPU:</strong> 16GB
          </div>
          <div className="info-item">
            <strong>Training Method:</strong> Data parallel across GPUs
          </div>
        </div>
      </div>
    </div>
  );

  const renderUsage = () => (
    <div className="tab-content">
      <div className="info-grid">
        <div className="info-card">
          <h3>Recommended Use Cases</h3>
          <ul>
            <li>Text generation</li>
            <li>Language modeling</li>
            <li>Text completion</li>
            <li>Creative writing</li>
            <li>Code generation</li>
            <li>Question answering (with fine-tuning)</li>
          </ul>
        </div>

        <div className="info-card">
          <h3>Limitations</h3>
          <ul>
            <li>No factual knowledge cutoff date</li>
            <li>May generate biased or harmful content</li>
            <li>Limited context window (1024 tokens)</li>
            <li>No built-in safety mechanisms</li>
            <li>Can hallucinate information</li>
          </ul>
        </div>

        <div className="info-card">
          <h3>Best Practices</h3>
          <ul>
            <li>Use appropriate temperature and top-k/top-p sampling</li>
            <li>Implement content filtering</li>
            <li>Validate generated content</li>
            <li>Use few-shot prompting for better results</li>
            <li>Consider fine-tuning for specific tasks</li>
          </ul>
        </div>

        <div className="info-card">
          <h3>Safety Considerations</h3>
          <ul>
            <li>Implement content moderation</li>
            <li>Use safety classifiers</li>
            <li>Monitor for harmful outputs</li>
            <li>Provide clear usage guidelines</li>
            <li>Consider ethical implications</li>
          </ul>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'architecture':
        return renderArchitecture();
      case 'equations':
        return renderEquations();
      case 'training':
        return renderTraining();
      case 'usage':
        return renderUsage();
      default:
        return renderOverview();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>🧠 Model Information: {modelName}</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="modal-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-icon">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        <div className="modal-body">
          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
              <p>Loading model information...</p>
            </div>
          ) : (
            renderTabContent()
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelInfoModal; 