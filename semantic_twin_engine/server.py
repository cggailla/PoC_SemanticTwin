#!/usr/bin/env python
"""Development server launcher for Semantic Twin Engine API."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app

if __name__ == "__main__":
    app = create_app()
    print("\n" + "=" * 60)
    print("Semantic Twin Engine - Development Server")
    print("=" * 60)
    print("Running on: http://0.0.0.0:8000")
    print("API Health: http://127.0.0.1:8000/api/health")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8000, use_reloader=False, threaded=True)
