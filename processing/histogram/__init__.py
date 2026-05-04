"""
Histogram Processing Module
============================
Provides histogram computation and equalization functions
for medical image enhancement.

Menna Hesham Ragab Allam - 1220321

"""

from .local_equalization import (
    compute_histogram,
    compute_cdf,
    global_histogram_equalization,
    local_histogram_equalization,
    local_histogram_equalization_optimized
)

__all__ = [
    'compute_histogram',
    'compute_cdf',
    'global_histogram_equalization',
    'local_histogram_equalization',
    'local_histogram_equalization_optimized'
]