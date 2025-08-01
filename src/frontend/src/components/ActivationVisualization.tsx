import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ActivationData } from '../services/api';
import './ActivationVisualization.css';

// Type definitions for Plotly
type PlotData = any[];
type PlotLayout = any;

interface ActivationVisualizationProps {
  activationData: ActivationData;
  visualizationType: 'heatmap' | 'scatter' | 'summary';
  layerIndex: number;
}

export const ActivationVisualization: React.FC<ActivationVisualizationProps> = ({
  activationData,
  visualizationType,
  layerIndex
}) => {
  const plotData = useMemo((): PlotData | null => {
    if (!activationData || layerIndex >= activationData.layers.length) {
      return null;
    }

    const layer = activationData.layers[layerIndex];
    const tokens = activationData.tokens;

    switch (visualizationType) {
      case 'heatmap':
        return createHeatmapData(layer, tokens);
      case 'scatter':
        return createScatterData(layer, tokens);
      case 'summary':
        return createSummaryData(activationData);
      default:
        return null;
    }
  }, [activationData, visualizationType, layerIndex]);

  const layout = useMemo((): PlotLayout => {
    const baseLayout = {
      title: `${visualizationType.charAt(0).toUpperCase() + visualizationType.slice(1)} - Layer ${layerIndex}`,
      autosize: true,
      margin: { l: 50, r: 50, t: 80, b: 50 },
      height: 500
    };

    switch (visualizationType) {
      case 'heatmap':
        return {
          ...baseLayout,
          xaxis: { title: 'Tokens' },
          yaxis: { title: 'Neurons' }
        };
      case 'scatter':
        return {
          ...baseLayout,
          xaxis: { title: 'PC1' },
          yaxis: { title: 'PC2' }
        };
      case 'summary':
        return {
          ...baseLayout,
          grid: { rows: 2, columns: 2, pattern: 'independent' }
        };
      default:
        return baseLayout;
    }
  }, [visualizationType, layerIndex]);

  if (!plotData) {
    return (
      <div className="visualization-error">
        <p>No data available for visualization</p>
      </div>
    );
  }

  return (
    <div className="activation-visualization">
      <Plot
        data={plotData}
        layout={layout}
        config={{ responsive: true }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

function createHeatmapData(layer: any, tokens: string[]) {
  const neurons = layer.neurons.slice(0, 100); // Limit to first 100 neurons for performance
  const z = neurons.map((neuron: any) => neuron.activations);
  
  return [{
    type: 'heatmap',
    z: z,
    x: tokens,
    y: neurons.map((_: any, i: number) => `N${i}`),
    colorscale: 'RdBu_r',
    name: 'Activation Values'
  }];
}

function createScatterData(layer: any, tokens: string[]) {
  const neurons = layer.neurons.slice(0, 200); // Limit for performance
  const activations = neurons.map((neuron: any) => neuron.activations);
  
  // Simple 2D projection (in a real app, you'd use PCA/t-SNE)
  const x = activations.map((acts: number[]) => acts[0] || 0);
  const y = activations.map((acts: number[]) => acts[1] || 0);
  const colors = activations.map((acts: number[]) => 
    acts.reduce((sum, val) => sum + Math.abs(val), 0) / acts.length
  );

  return [{
    type: 'scatter',
    mode: 'markers',
    x: x,
    y: y,
    marker: {
      color: colors,
      colorscale: 'Viridis',
      size: 8,
      opacity: 0.7
    },
    text: neurons.map((_: any, i: number) => `Neuron ${i}`),
    hovertemplate: 'Neuron %{text}<br>PC1: %{x}<br>PC2: %{y}<extra></extra>'
  }];
}

function createSummaryData(activationData: ActivationData) {
  const layers = activationData.layers;
  const layerIndices = layers.map((_, i) => i);
  
  // Calculate statistics for each layer
  const means = layers.map(layer => {
    const allActivations = layer.neurons.flatMap(neuron => neuron.activations);
    return allActivations.reduce((sum, val) => sum + val, 0) / allActivations.length;
  });

  const stds = layers.map(layer => {
    const allActivations = layer.neurons.flatMap(neuron => neuron.activations);
    const mean = allActivations.reduce((sum, val) => sum + val, 0) / allActivations.length;
    const variance = allActivations.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / allActivations.length;
    return Math.sqrt(variance);
  });

  return [
    {
      type: 'scatter',
      x: layerIndices,
      y: means,
      mode: 'lines+markers',
      name: 'Mean Activation',
      line: { color: 'blue' },
      marker: { size: 8 }
    },
    {
      type: 'scatter',
      x: layerIndices,
      y: stds,
      mode: 'lines+markers',
      name: 'Std Deviation',
      line: { color: 'red' },
      marker: { size: 8 }
    }
  ];
} 