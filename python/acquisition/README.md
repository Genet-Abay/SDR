# Acquisition Module

Hardware abstraction layer for RTL-SDR device acquisition with async streaming support.

## Components

### HardwareInterface (`hw_interface.py`)

Wraps the Pybind11 driver and provides a Pythonic interface with async streaming.

**Key Features:**
- Context manager support (`with` statement)
- Async generator for streaming IQ samples
- Synchronous sample reading
- Device configuration (frequency, sample rate)

**Example:**
```python
from acquisition import HardwareInterface

# Using context manager
with HardwareInterface(device_index=0) as hw:
    hw.set_frequency(100000000)  # 100 MHz
    hw.set_sample_rate(2048000)  # 2.048 MS/s
    
    # Synchronous reading
    samples = hw.read_samples(1024)
    
    # Async streaming
    async for samples in hw.stream_iq(chunk_size=8192):
        # Process samples
        power = np.abs(samples) ** 2
```

### DeviceManager (`device_manager.py`)

Manages device discovery and selection.

**Key Features:**
- List available devices
- Find devices by index or name
- Interactive device selection
- Device information

**Example:**
```python
from acquisition import DeviceManager, choose_device

# List all devices
DeviceManager.print_devices()

# Get device count
count = DeviceManager.get_device_count()

# Choose a device (interactive)
device = choose_device(interactive=True)

# Find device by name
device = DeviceManager.find_device(device_name="RTL")
```

## Async Streaming

The `stream_iq()` method is an async generator that yields NumPy arrays of `complex64` samples:

```python
import asyncio
import numpy as np
from acquisition import HardwareInterface

async def main():
    hw = HardwareInterface(device_index=0)
    hw.open()
    hw.set_frequency(100000000)
    hw.set_sample_rate(2048000)
    
    async for samples in hw.stream_iq(chunk_size=16384):
        # samples is a numpy.ndarray with dtype=complex64
        print(f"Received {len(samples)} samples")
        
        # Process samples
        magnitude = np.abs(samples)
        phase = np.angle(samples)
        power = np.abs(samples) ** 2
    
    hw.close()

asyncio.run(main())
```

### Parameters

- `chunk_size`: Number of samples per chunk (default: 16384)
- `timeout`: Maximum time to wait for samples (not fully implemented yet)

### Notes

- The async generator automatically starts streaming if not already started
- Streaming continues until the generator is cancelled or the device is closed
- Samples are read from the internal ring buffer
- The generator yields whatever samples are available, up to `chunk_size`

## Complete Example

See `example_async.py` for a complete working example.

## API Reference

### HardwareInterface

- `open()` - Open device
- `close()` - Close device
- `is_open()` - Check if device is open
- `set_frequency(frequency_hz)` - Set center frequency
- `get_frequency()` - Get current frequency
- `set_sample_rate(sample_rate_hz)` - Set sample rate
- `get_sample_rate()` - Get current sample rate
- `start_streaming(buffer_size)` - Start streaming
- `stop_streaming()` - Stop streaming
- `is_streaming()` - Check if streaming
- `read_samples(max_samples)` - Read samples synchronously
- `stream_iq(chunk_size, timeout)` - Async generator for streaming
- `available_samples()` - Get available sample count
- `get_last_error()` - Get last error message

### DeviceManager

- `get_device_count()` - Get number of devices
- `list_devices()` - List all devices
- `find_device(device_index, device_name)` - Find device
- `get_default_device()` - Get first device
- `print_devices()` - Print device list

### choose_device()

- `choose_device(device_index, device_name, interactive)` - Choose device

