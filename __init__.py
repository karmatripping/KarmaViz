"""
KarmaViz - Audio Visualizer Package
"""
__version__ = "1.0.0"
try:
    from .modules.karmaviz import KarmaVisualizer
    __all__ = ['KarmaVisualizer']
except ImportError:
    # Handle import errors gracefully for testing
    __all__ = []