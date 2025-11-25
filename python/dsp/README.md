# DSP Module

Digital Signal Processing functions for SDR applications.

## Components

### Filters (`filters.py`)

#### FIRLowpass

Finite Impulse Response lowpass filter using `scipy.signal.firwin`.

**Features:**
- Configurable cutoff frequency
- Automatic tap calculation based on transition bandwidth
- Window function selection (Hamming, Hann, Blackman, etc.)
- Streaming support with state preservation

**Example:**
```python
from dsp import FIRLowpass

# Create filter
filter = FIRLowpass(
    cutoff=100000,  # 100 kHz cutoff
    sample_rate=2048000,  # 2.048 MS/s
    numtaps=201,  # Filter order
    window='hamming'
)

# Filter signal
filtered = filter.filter(iq_samples)
```

#### frequency_shift()

Frequency shift (mix) a signal by a given offset.

**Example:**
```python
from dsp import frequency_shift

# Shift signal up by 100 kHz
shifted = frequency_shift(
    signal_in=iq_samples,
    frequency_hz=100000,
    sample_rate=2048000
)
```

#### Resampler

Resample signals with rational or arbitrary rate changes.

**Features:**
- Polyphase resampling for rational ratios (efficient)
- FFT-based resampling for arbitrary ratios
- Streaming support

**Example:**
```python
from dsp import Resampler

# Resample from 2.048 MS/s to 48 kHz
resampler = Resampler(
    input_rate=2048000,
    output_rate=48000,
    method='poly'  # or 'fft'
)

resampled = resampler.resample(iq_samples)
```

### FM Demodulation (`demod_fm.py`)

#### FMDemodulator

FM demodulator using phase differentiation.

**Features:**
- Phase differentiation for FM demodulation
- De-emphasis filtering (75 μs for North America, 50 μs for Europe)
- Audio normalization to float32
- Configurable audio bandwidth and gain

**Example:**
```python
from dsp import FMDemodulator

# Create demodulator
fm_demod = FMDemodulator(
    sample_rate=48000,  # Audio sample rate
    deemphasis_tau=75e-6,  # 75 μs (North America)
    audio_cutoff=15000,  # 15 kHz audio bandwidth
    audio_gain=0.5
)

# Demodulate
audio = fm_demod.demodulate(iq_samples)
# audio is numpy.ndarray with dtype=float32, range [-1, 1]
```

## Complete FM Receiver Pipeline

```python
import asyncio
import numpy as np
from acquisition import HardwareInterface, choose_device
from dsp import FIRLowpass, Resampler, FMDemodulator

async def fm_receiver():
    # Setup hardware
    device = choose_device()
    hw = HardwareInterface(device_index=device.index)
    hw.open()
    hw.set_frequency(101500000)  # 101.5 MHz
    hw.set_sample_rate(2048000)  # 2.048 MS/s
    
    # Create DSP pipeline
    channel_filter = FIRLowpass(cutoff=100000, sample_rate=2048000)
    resampler = Resampler(input_rate=2048000, output_rate=48000)
    fm_demod = FMDemodulator(sample_rate=48000, deemphasis_tau=75e-6)
    
    # Process samples
    async for iq_samples in hw.stream_iq(chunk_size=16384):
        # 1. Channel filter
        filtered = channel_filter.filter(iq_samples)
        
        # 2. Resample to audio rate
        audio_rate = resampler.resample(filtered)
        
        # 3. FM demodulate
        audio = fm_demod.demodulate(audio_rate)
        
        # Use audio (write to file, play, etc.)
        print(f"Audio: {len(audio)} samples, RMS={np.sqrt(np.mean(audio**2)):.4f}")
    
    hw.close()

asyncio.run(fm_receiver())
```

## Technical Details

### FM Demodulation

FM demodulation uses phase differentiation:
1. Normalize IQ samples to unit magnitude
2. Compute phase difference: `arg(s[n] * conj(s[n-1]))`
3. Phase difference is proportional to frequency deviation
4. Apply de-emphasis filter
5. Normalize to [-1, 1] range
6. Convert to float32

### De-emphasis

FM broadcasting uses pre-emphasis (high frequencies boosted) during transmission.
De-emphasis is applied during reception to restore flat frequency response.

- **North America**: 75 μs time constant
- **Europe**: 50 μs time constant

The de-emphasis filter is implemented as a first-order IIR lowpass filter using bilinear transform.

### Audio Output

Audio output is:
- Normalized to [-1, 1] range
- Float32 dtype
- Ready for WAV file writing or audio playback

## Dependencies

- `numpy` - Array operations
- `scipy` - Signal processing functions (firwin, resample_poly, lfilter)

