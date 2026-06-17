"""
rdacca.hp - Hierarchical and Variation Partitioning for Canonical Analysis
Python implementation of the R package for hierarchical partitioning in canonical analysis.
"""

from .core import rdacca_hp, RdaccaHpResult, calculate_rda, calculate_cca, calculate_dbrda
from .utils import create_test_data, create_cca_test_data, create_distance_test_data

# Import permutation functions
try:
    from .permutation import permu_hp
except ImportError as e:
    print(f"Warning: Could not import permutation functions: {e}")

    def permu_hp(*args, **kwargs):
        """Permutation test - not available"""
        raise NotImplementedError("Permutation test not available")

# Import plotting functions
try:
    from .plotting import plot_rdaccahp, plot_comparison
except ImportError as e:
    print(f"Warning: Could not import plotting functions: {e}")

    def plot_rdaccahp(*args, **kwargs):
        """Plotting function - not available"""
        raise NotImplementedError("Plotting functionality not available")

    def plot_comparison(*args, **kwargs):
        """Comparison plotting function - not available"""
        raise NotImplementedError("Comparison plotting functionality not available")

__version__ = "0.1.2"
__author__ = "Jiangshan Lai"
__email__ = "lai@njfu.edu.cn"

__all__ = [
    'rdacca_hp',
    'RdaccaHpResult',
    'calculate_rda',
    'calculate_cca',
    'calculate_dbrda',
    'create_test_data',
    'create_cca_test_data',
    'create_distance_test_data',
    'permu_hp',
    'plot_rdaccahp',
    'plot_comparison',
]