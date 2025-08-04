// Simple test to verify frontend API connection
const testAPI = async () => {
  try {
    console.log('Testing API connection...');
    
    // Test health endpoint
    const healthResponse = await fetch('http://localhost:5001/api/health');
    const healthData = await healthResponse.json();
    console.log('Health check:', healthData);
    
    // Test models endpoint
    const modelsResponse = await fetch('http://localhost:5001/api/models');
    const modelsData = await modelsResponse.json();
    console.log('Models count:', Object.keys(modelsData.models || {}).length);
    console.log('Current model:', modelsData.current_model);
    
    // Test memory endpoint
    const memoryResponse = await fetch('http://localhost:5001/api/models/memory');
    const memoryData = await memoryResponse.json();
    console.log('Memory usage:', memoryData);
    
    console.log('✅ All API tests passed!');
  } catch (error) {
    console.error('❌ API test failed:', error);
  }
};

// Run the test
testAPI(); 