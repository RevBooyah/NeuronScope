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
    // For now, we'll serve the data files from the public directory
    // In a real app, this would be a proper backend API
    this.baseUrl = process.env.PUBLIC_URL || '';
  }

  async getAvailableActivations(): Promise<string[]> {
    try {
      // For now, return a hardcoded list of available files
      // In a real implementation, this would fetch from the backend
      return [
        'Hello_world_20250731_210100.json'
      ];
    } catch (error) {
      console.error('Failed to fetch available activations:', error);
      return [];
    }
  }

  async loadActivationData(filename: string): Promise<ActivationData | null> {
    try {
      // For development, we'll load from the public/data directory
      const response = await fetch(`${this.baseUrl}/data/activations/${filename}`);
      
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
      const response = await fetch(`${this.baseUrl}/data/samples.json`);
      
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

  // Mock method to simulate generating new activation data
  async generateActivationData(prompt: string): Promise<ActivationData | null> {
    try {
      // In a real implementation, this would call the Python backend
      console.log(`Generating activation data for prompt: "${prompt}"`);
      
      // For now, return null to indicate we need to use existing data
      // This would trigger the backend to process the prompt
      return null;
    } catch (error) {
      console.error('Failed to generate activation data:', error);
      return null;
    }
  }
}

export const apiService = new ApiService(); 