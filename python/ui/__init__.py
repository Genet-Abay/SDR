"""
User Interface module

Provides real-time visualization for SDR data.
"""

from .waterfall import (
    WaterfallDisplay,
    create_waterfall
)

from .spectrum import (
    SpectrumDisplay,
    create_spectrum
)

__all__ = [
    # Waterfall
    'WaterfallDisplay',
    'create_waterfall',
    # Spectrum
    'SpectrumDisplay',
    'create_spectrum',
]
