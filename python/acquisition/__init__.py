"""
Acquisition module for SDR data capture.

Provides hardware abstraction layer for RTL-SDR devices.
"""

from .hw_interface import HardwareInterface, HardwareInterfaceError
from .device_manager import (
    DeviceManager,
    DeviceInfo,
    choose_device
)

__all__ = [
    'HardwareInterface',
    'HardwareInterfaceError',
    'DeviceManager',
    'DeviceInfo',
    'choose_device',
]
