import sys
from pathlib import Path

# Make the project modules (model.py, inference.py) importable from the tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
