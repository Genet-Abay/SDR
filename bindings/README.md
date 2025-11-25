# SDR Hardware Bindings

Pybind11 bindings for the RTL-SDR driver.

## Building

### Prerequisites

- CMake 3.15 or higher
- C++17 compatible compiler
- Python 3.x with development headers
- pybind11 (install via pip: `pip install pybind11`)
- librtlsdr development libraries

### Linux

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install librtlsdr-dev cmake build-essential python3-dev

# Install pybind11
pip install pybind11

# Build
mkdir build
cd build
cmake ..
make

# The module will be in build/sdrhw.cpython-*.so
```

### Windows

```bash
# Install dependencies
# 1. Install librtlsdr (or build from source)
#    Place headers in C:/librtlsdr/include
#    Place libraries in C:/librtlsdr/lib
#    Or set RTLSDR_INCLUDE_DIR and RTLSDR_LIB_DIR environment variables

# 2. Install pybind11
pip install pybind11

# 3. Build with CMake
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# The module will be in build/Release/sdrhw.pyd
```

### Using Environment Variables (Windows)

If librtlsdr is not in standard locations:

```cmd
set RTLSDR_INCLUDE_DIR=C:\path\to\librtlsdr\include
set RTLSDR_LIB_DIR=C:\path\to\librtlsdr\lib
cmake ..
```

## Usage

```python
import sdrhw
import numpy as np

# Create driver
driver = sdrhw.RtlSdrDriver(device_index=0)

# Open device
if not driver.open():
    print(f"Error: {driver.get_last_error()}")
    exit(1)

# Configure
driver.set_freq(100000000)  # 100 MHz
driver.set_sample_rate(2048000)  # 2.048 MS/s

# Start streaming
driver.start_streaming(buffer_size=1024*1024)

# Read samples
samples = driver.read_samples(1024)  # Returns numpy array of complex64
print(f"Read {len(samples)} samples")
print(f"Sample type: {samples.dtype}")

# Stop streaming
driver.stop_streaming()

# Close device
driver.close()
```

## Module API

- `RtlSdrDriver(device_index=0)` - Constructor
- `open()` - Open device
- `close()` - Close device
- `set_freq(frequency_hz)` - Set frequency in Hz
- `set_sample_rate(sample_rate_hz)` - Set sample rate in Hz
- `read_samples(max_samples)` - Read samples (returns numpy array)
- `start_streaming(buffer_size)` - Start streaming
- `stop_streaming()` - Stop streaming
- `is_open()` - Check if device is open
- `is_streaming()` - Check if streaming
- `available_samples()` - Get available sample count
- `get_last_error()` - Get last error message
- `get_device_count()` - Static: Get device count
- `get_device_name(device_index)` - Static: Get device name

