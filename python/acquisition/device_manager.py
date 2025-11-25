"""
Device Manager - Handles RTL-SDR device discovery and selection
"""

from typing import List, Optional, Dict
import sdrhw


class DeviceInfo:
    """Information about an RTL-SDR device."""
    
    def __init__(self, index: int, name: str):
        self.index = index
        self.name = name
    
    def __repr__(self) -> str:
        return f"DeviceInfo(index={self.index}, name='{self.name}')"


class DeviceManager:
    """
    Manages RTL-SDR device discovery and selection.
    """
    
    @staticmethod
    def get_device_count() -> int:
        """
        Get the number of available RTL-SDR devices.
        
        Returns:
            Number of devices
        """
        try:
            return sdrhw.RtlSdrDriver.get_device_count()
        except Exception:
            return 0
    
    @staticmethod
    def list_devices() -> List[DeviceInfo]:
        """
        List all available RTL-SDR devices.
        
        Returns:
            List of DeviceInfo objects
        """
        devices = []
        try:
            count = sdrhw.RtlSdrDriver.get_device_count()
            for i in range(count):
                name = sdrhw.RtlSdrDriver.get_device_name(i)
                devices.append(DeviceInfo(index=i, name=name))
        except Exception as e:
            print(f"Error listing devices: {e}")
        
        return devices
    
    @staticmethod
    def find_device(device_index: Optional[int] = None, 
                   device_name: Optional[str] = None) -> Optional[DeviceInfo]:
        """
        Find a device by index or name.
        
        Args:
            device_index: Device index to find (takes precedence)
            device_name: Device name to find (partial match supported)
            
        Returns:
            DeviceInfo if found, None otherwise
        """
        devices = DeviceManager.list_devices()
        
        if not devices:
            return None
        
        # If index is specified, return that device
        if device_index is not None:
            for device in devices:
                if device.index == device_index:
                    return device
            return None
        
        # If name is specified, search for it
        if device_name is not None:
            device_name_lower = device_name.lower()
            for device in devices:
                if device_name_lower in device.name.lower():
                    return device
        
        # Default: return first device
        return devices[0] if devices else None
    
    @staticmethod
    def get_default_device() -> Optional[DeviceInfo]:
        """
        Get the default (first available) device.
        
        Returns:
            DeviceInfo for first device, or None if no devices
        """
        devices = DeviceManager.list_devices()
        return devices[0] if devices else None
    
    @staticmethod
    def print_devices():
        """Print all available devices to stdout."""
        devices = DeviceManager.list_devices()
        
        if not devices:
            print("No RTL-SDR devices found")
            return
        
        print(f"Found {len(devices)} RTL-SDR device(s):")
        for device in devices:
            print(f"  [{device.index}] {device.name}")


def choose_device(device_index: Optional[int] = None,
                 device_name: Optional[str] = None,
                 interactive: bool = False) -> Optional[DeviceInfo]:
    """
    Choose a device, optionally interactively.
    
    Args:
        device_index: Device index to use
        device_name: Device name to search for
        interactive: If True, prompt user to choose device
        
    Returns:
        Selected DeviceInfo, or None if no device available
    """
    devices = DeviceManager.list_devices()
    
    if not devices:
        print("No RTL-SDR devices found")
        return None
    
    # If index or name specified, use that
    if device_index is not None or device_name is not None:
        return DeviceManager.find_device(device_index, device_name)
    
    # If interactive, let user choose
    if interactive:
        DeviceManager.print_devices()
        try:
            choice = input(f"Select device [0-{len(devices)-1}] (default: 0): ").strip()
            if choice:
                device_index = int(choice)
            else:
                device_index = 0
            return DeviceManager.find_device(device_index)
        except (ValueError, KeyboardInterrupt):
            print("Invalid selection or cancelled")
            return None
    
    # Default: return first device
    return devices[0]

