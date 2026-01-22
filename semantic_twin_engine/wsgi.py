"""
WSGI entry point for Gunicorn deployment.
"""

import sys
from pathlib import Path

# Add semantic_twin_engine to path
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
