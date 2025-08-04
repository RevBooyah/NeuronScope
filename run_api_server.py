#!/usr/bin/env python3
"""
Simple script to run the NeuronScope API server from the root directory.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.api_server import app

if __name__ == "__main__":
    print("🧠 Starting NeuronScope API server on port 5001")
    print("📁 Data directory: data")
    print("🌐 API endpoints available at: http://localhost:5001/api/")
    app.run(host='0.0.0.0', port=5001, debug=True) 