#!/usr/bin/env python3
"""
Run the HOMETOWN recommendation API server.
"""

import uvicorn
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from stream_rec.api.main import app

if __name__ == "__main__":
    print("🏠 Starting HOMETOWN Recommendation API Server...")
    print("📍 Available endpoints:")
    print("   • GET  /                       - Health check")
    print("   • POST /v1/scenarios/hometown  - HOMETOWN recommendations")
    print("   • GET  /v1/scenarios/hometown/stats - System statistics")
    print("🌐 Server will be available at http://localhost:8000")
    print("📖 API docs at http://localhost:8000/docs")
    
    uvicorn.run(
        "stream_rec.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
