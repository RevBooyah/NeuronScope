export interface ActivationData {
  prompt: string;
  tokens: string[];
  layers: LayerData[];
}

export interface LayerData {
  layer_index: number;
  neurons: NeuronData[];
}

export interface NeuronData {
  neuron_index: number;
  activations: number[];
}

export interface VisualizationData {
  type: 'heatmap' | 'scatter' | 'summary';
  layerIndex: number;
  data: any;
}

class ApiService {
  private baseUrl: string;

  constructor() {
    // Connect to Flask backend API server
    this.baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
  }

  async getAvailableActivations(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/activations/files`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data.files || [];
    } catch (error) {
      console.error('Failed to fetch available activations:', error);
      return [];
    }
  }

  async loadActivationData(filename: string): Promise<ActivationData | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/activations/${filename}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data: ActivationData = await response.json();
      return data;
    } catch (error) {
      console.error('Failed to load activation data:', error);
      return null;
    }
  }

  async getSamplePrompts(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/samples`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const prompts: string[] = await response.json();
      return prompts;
    } catch (error) {
      console.error('Failed to load sample prompts:', error);
      // Return fallback prompts
      return [
        "Hello world",
        "What is the capital of France?",
        "Translate this sentence to Spanish.",
        "The cat sat on the mat.",
        "If it rains tomorrow, we'll cancel the picnic."
      ];
    }
  }

  async addSamplePrompt(prompt: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/samples`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Added sample prompt:', result);
      return true;
    } catch (error) {
      console.error('Failed to add sample prompt:', error);
      return false;
    }
  }

  async generateActivationData(prompt: string): Promise<ActivationData | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/activations/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      return result.data || null;
    } catch (error) {
      console.error('Failed to generate activation data:', error);
      return null;
    }
  }

  async getAvailableClusters(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/clusters/files`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data.files || [];
    } catch (error) {
      console.error('Failed to fetch available clusters:', error);
      return [];
    }
  }

  async loadClusterData(filename: string): Promise<any | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/clusters/${filename}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Failed to load cluster data:', error);
      return null;
    }
  }

  async generateClusterData(layerIndex: number, nClusters: number = 5): Promise<any | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/clusters/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          layer_index: layerIndex, 
          n_clusters: nClusters 
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      return result.data || null;
    } catch (error) {
      console.error('Failed to generate cluster data:', error);
      return null;
    }
  }

  async getAvailableQueries(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/queries/files`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data.files || [];
    } catch (error) {
      console.error('Failed to fetch available queries:', error);
      return [];
    }
  }

  async loadQueryData(filename: string): Promise<any | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/queries/${filename}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Failed to load query data:', error);
      return null;
    }
  }

  async queryNeuron(neuronIndex: number, layerIndex: number = 0, topK: number = 10): Promise<any | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/queries/neuron`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          neuron_index: neuronIndex, 
          layer_index: layerIndex,
          top_k: topK
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      return result.data || null;
    } catch (error) {
      console.error('Failed to query neuron:', error);
      return null;
    }
  }

  async queryCluster(clusterIndices: number[], layerIndex: number = 0, topK: number = 10): Promise<any | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/queries/cluster`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          cluster_indices: clusterIndices, 
          layer_index: layerIndex,
          top_k: topK
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      return result.data || null;
    } catch (error) {
      console.error('Failed to query cluster:', error);
      return null;
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`);
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
}

export const apiService = new ApiService(); 