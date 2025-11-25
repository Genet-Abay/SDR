# UI Module

Real-time visualization components for SDR data using matplotlib.

## Components

### Waterfall Display (`waterfall.py`)

Real-time frequency vs time waterfall plot.

**Features:**
- Updates at ~10 FPS (configurable)
- Accepts IQ buffers (complex64 numpy arrays)
- FFT-based frequency analysis
- Configurable colormap
- Automatic color scaling
- FPS counter

**Example:**
```python
from ui import WaterfallDisplay
from acquisition import HardwareInterface

hw = HardwareInterface(device_index=0)
hw.open()
hw.set_frequency(100000000)
hw.set_sample_rate(2048000)

# Create waterfall
waterfall = WaterfallDisplay(
    sample_rate=2048000,
    center_freq=100000000,
    fft_size=2048,
    num_lines=200,
    update_rate=10.0,
    colormap='viridis'
)

# Start display
waterfall.start()

# Feed samples
async for iq_samples in hw.stream_iq(chunk_size=16384):
    waterfall.add_samples(iq_samples)
    await asyncio.sleep(0.01)

waterfall.close()
```

**Using Context Manager:**
```python
with WaterfallDisplay(sample_rate=2048000, center_freq=100000000) as waterfall:
    async for iq_samples in hw.stream_iq():
        waterfall.add_samples(iq_samples)
        await asyncio.sleep(0.01)
```

### Spectrum Display (`spectrum.py`)

Real-time FFT magnitude spectrum plot.

**Features:**
- Updates at ~10 FPS (configurable)
- Accepts IQ buffers (complex64 numpy arrays)
- Autoscaling Y-axis
- Peak frequency marker with label
- FPS counter

**Example:**
```python
from ui import SpectrumDisplay
from acquisition import HardwareInterface

hw = HardwareInterface(device_index=0)
hw.open()
hw.set_frequency(100000000)
hw.set_sample_rate=2048000

# Create spectrum
spectrum = SpectrumDisplay(
    sample_rate=2048000,
    center_freq=100000000,
    fft_size=2048,
    update_rate=10.0,
    show_peak=True
)

# Start display
spectrum.start()

# Feed samples
async for iq_samples in hw.stream_iq(chunk_size=16384):
    spectrum.add_samples(iq_samples)
    await asyncio.sleep(0.01)

spectrum.close()
```

**Using Context Manager:**
```python
with SpectrumDisplay(sample_rate=2048000, center_freq=100000000) as spectrum:
    async for iq_samples in hw.stream_iq():
        spectrum.add_samples(iq_samples)
        await asyncio.sleep(0.01)
```

## Complete Example

See `example_display.py` for a complete working example that demonstrates both displays.

## API Reference

### WaterfallDisplay

**Constructor Parameters:**
- `sample_rate`: Sample rate in Hz
- `center_freq`: Center frequency in Hz (for display)
- `fft_size`: FFT size for frequency resolution (default: 2048)
- `num_lines`: Number of time lines to display (default: 200)
- `update_rate`: Target update rate in FPS (default: 10.0)
- `colormap`: Matplotlib colormap name (default: 'viridis')
- `title`: Window title

**Methods:**
- `add_samples(iq_samples)`: Add IQ samples to buffer
- `start()`: Start the display animation
- `stop()`: Stop the display
- `close()`: Close the window
- `set_center_frequency(freq_hz)`: Update center frequency

### SpectrumDisplay

**Constructor Parameters:**
- `sample_rate`: Sample rate in Hz
- `center_freq`: Center frequency in Hz (for display)
- `fft_size`: FFT size for frequency resolution (default: 2048)
- `update_rate`: Target update rate in FPS (default: 10.0)
- `title`: Window title
- `show_peak`: Show peak frequency marker (default: True)

**Methods:**
- `add_samples(iq_samples)`: Add IQ samples to buffer
- `start()`: Start the display animation
- `stop()`: Stop the display
- `close()`: Close the window
- `set_center_frequency(freq_hz)`: Update center frequency
- `set_autoscale(enabled)`: Enable/disable autoscaling
- `set_ylim(ymin, ymax)`: Set Y-axis limits manually

## Performance Notes

- Both displays update at approximately 10 FPS by default
- Samples are buffered and processed in batches
- FFT computation is done on the most recent samples
- For best performance, use appropriate chunk sizes (e.g., 16384 samples)
- The displays use matplotlib's animation framework for smooth updates

## Dependencies

- `matplotlib` - Plotting and animation
- `numpy` - Array operations and FFT

