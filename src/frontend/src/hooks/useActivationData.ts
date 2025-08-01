import { useState, useEffect, useCallback } from 'react';
import { apiService, ActivationData } from '../services/api';

export interface UseActivationDataReturn {
  activationData: ActivationData | null;
  loading: boolean;
  error: string | null;
  availableFiles: string[];
  samplePrompts: string[];
  loadActivationData: (filename: string) => Promise<void>;
  generateNewActivation: (prompt: string) => Promise<void>;
}

export const useActivationData = (): UseActivationDataReturn => {
  const [activationData, setActivationData] = useState<ActivationData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [samplePrompts, setSamplePrompts] = useState<string[]>([]);

  // Load available files and sample prompts on mount
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const [files, prompts] = await Promise.all([
          apiService.getAvailableActivations(),
          apiService.getSamplePrompts()
        ]);
        
        setAvailableFiles(files);
        setSamplePrompts(prompts);
        
        // Load the first available file by default
        if (files.length > 0) {
          await loadActivationData(files[0]);
        }
      } catch (err) {
        setError('Failed to load initial data');
        console.error('Failed to load initial data:', err);
      }
    };

    loadInitialData();
  }, []);

  const loadActivationData = useCallback(async (filename: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await apiService.loadActivationData(filename);
      
      if (data) {
        setActivationData(data);
      } else {
        setError('Failed to load activation data');
      }
    } catch (err) {
      setError('Failed to load activation data');
      console.error('Failed to load activation data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const generateNewActivation = useCallback(async (prompt: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await apiService.generateActivationData(prompt);
      
      if (data) {
        setActivationData(data);
      } else {
        setError('Failed to generate activation data. Please use an existing prompt.');
      }
    } catch (err) {
      setError('Failed to generate activation data');
      console.error('Failed to generate activation data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    activationData,
    loading,
    error,
    availableFiles,
    samplePrompts,
    loadActivationData,
    generateNewActivation
  };
}; 