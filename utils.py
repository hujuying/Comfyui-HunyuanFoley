import os
import sys


def ensure_in_path():
    """Add this package's directory to Python path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)