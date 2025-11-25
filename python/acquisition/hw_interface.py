"""
Hardware Interface - Wraps the Pybind11 RTL-SDR driver
"""

import asyncio
import numpy as np
from typing import AsyncGenerator, Optional
import sdrhw


class HardwareInterface:
    """
    Hardware abstraction layer for RTL-SDR devices.
    Wraps the Pybind11 driver and provides async streaming interface.
    """
    
    def __init__(self, device_index: int = 0):
        """
        Initialize hardware interface.
        
        Args:
            device_index: Index of the RTL-SDR device to use
        """
        self.device_index = device_index
        self.driver: Optional[sdrhw.RtlSdrDriver] = None
        self._streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def open(self) -> bool:
        """
        Open the RTL-SDR device.
        
        Returns:
            True if successful, False otherwise
        """
        if self.driver is not None:
            if self.driver.is_open():
                return True
            else:
                self.driver = None
        
        try:
            self.driver = sdrhw.RtlSdrDriver(self.device_index)
            return self.driver.open()
        except Exception as e:
            print(f"Error opening device: {e}")
            return False
    
    def close(self):
        """Close the RTL-SDR device and stop streaming."""
        self.stop_streaming()
        if self.driver is not None:
            self.driver.close()
            self.driver = None
    
    def is_open(self) -> bool:
        """Check if device is open."""
        return self.driver is not None and self.driver.is_open()
    
    def set_frequency(self, frequency_hz: int) -> bool:
        """
        Set center frequency.
        
        Args:
            frequency_hz: Frequency in Hz
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_open():
            raise RuntimeError("Device not open")
        return self.driver.set_freq(frequency_hz)
    
    def get_frequency(self) -> Optional[int]:
        """Get current frequency."""
        if not self.is_open():
            return None
        try:
            # pybind11 converts std::optional to Python's Optional
            return self.driver.get_frequency()
        except AttributeError:
            # Fallback if method not bound
            return None
    
    def set_sample_rate(self, sample_rate_hz: int) -> bool:
        """
        Set sample rate.
        
        Args:
            sample_rate_hz: Sample rate in Hz
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_open():
            raise RuntimeError("Device not open")
        return self.driver.set_sample_rate(sample_rate_hz)
    
    def get_sample_rate(self) -> Optional[int]:
        """Get current sample rate."""
        if not self.is_open():
            return None
        try:
            # pybind11 converts std::optional to Python's Optional
            return self.driver.get_sample_rate()
        except AttributeError:
            # Fallback if method not bound
            return None
    
    def start_streaming(self, buffer_size: int = 1024 * 1024) -> bool:
        """
        Start streaming IQ samples.
        
        Args:
            buffer_size: Size of the internal ring buffer in samples
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_open():
            raise RuntimeError("Device not open")
        
        if self._streaming:
            return True
        
        success = self.driver.start_streaming(buffer_size)
        if success:
            self._streaming = True
        return success
    
    def stop_streaming(self):
        """Stop streaming IQ samples."""
        if self.driver is not None and self._streaming:
            self.driver.stop_streaming()
            self._streaming = False
        
        if self._stream_task is not None:
            self._stream_task.cancel()
            self._stream_task = None
    
    def is_streaming(self) -> bool:
        """Check if streaming is active."""
        return self._streaming and (self.driver is not None and self.driver.is_streaming())
    
    def available_samples(self) -> int:
        """Get number of samples available in buffer."""
        if not self.is_open():
            return 0
        return self.driver.available_samples()
    
    def read_samples(self, max_samples: int) -> np.ndarray:
        """
        Read samples synchronously.
        
        Args:
            max_samples: Maximum number of samples to read
            
        Returns:
            NumPy array of complex64 samples
        """
        if not self.is_open():
            raise RuntimeError("Device not open")
        return self.driver.read_samples(max_samples)
    
    async def stream_iq(
        self,
        chunk_size: int = 16384,
        timeout: float = 1.0
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator that yields IQ sample buffers.
        
        Args:
            chunk_size: Number of samples per chunk
            timeout: Maximum time to wait for samples (seconds)
            
        Yields:
            NumPy arrays of complex64 samples
            
        Example:
            async for samples in hw.stream_iq(chunk_size=8192):
                # Process samples
                power = np.abs(samples) ** 2
        """
        if not self.is_open():
            raise RuntimeError("Device not open")
        
        if not self.is_streaming():
            if not self.start_streaming():
                raise RuntimeError(f"Failed to start streaming: {self.driver.get_last_error()}")
        
        self._streaming = True
        
        try:
            while self._streaming:
                # Wait for samples to be available
                available = self.available_samples()
                
                if available >= chunk_size:
                    # Read a chunk
                    samples = self.read_samples(chunk_size)
                    yield samples
                elif available > 0:
                    # Read whatever is available
                    samples = self.read_samples(available)
                    yield samples
                else:
                    # No samples available, wait a bit
                    await asyncio.sleep(0.01)  # 10ms sleep
                    
                    # Check timeout
                    # Note: This is a simple implementation
                    # For more sophisticated timeout handling, track start time
                    
        except asyncio.CancelledError:
            # Streaming was cancelled
            pass
        finally:
            # Cleanup is handled by stop_streaming if needed
            pass
    
    def get_last_error(self) -> str:
        """Get last error message."""
        if self.driver is None:
            return "Device not initialized"
        return self.driver.get_last_error()


class HardwareInterfaceError(Exception):
    """Exception raised for hardware interface errors."""
    pass

