#!/usr/bin/env python3
"""
Example: FM Demodulation Pipeline

Demonstrates:
- Frequency shifting to center FM signal
- Lowpass filtering
- FM demodulation with de-emphasis
- Audio output
"""

import numpy as np
import asyncio
from acquisition import HardwareInterface, choose_device
from dsp import (
    FIRLowpass,
    frequency_shift,
    Resampler,
    FMDemodulator
)


async def fm_receiver_example():
    """Complete FM receiver pipeline example."""
    # Choose device
    device = choose_device()
    if device is None:
        print("No device available")
        return
    
    # FM station parameters
    center_freq = 101500000  # 101.5 MHz (example FM station)
    rf_sample_rate = 2048000  # 2.048 MS/s
    audio_sample_rate = 48000  # 48 kHz audio output
    
    # Create hardware interface
    hw = HardwareInterface(device_index=device.index)
    
    try:
        # Open and configure device
        hw.open()
        hw.set_frequency(center_freq)
        hw.set_sample_rate(rf_sample_rate)
        
        print(f"Tuned to {center_freq / 1e6:.2f} MHz")
        print(f"RF sample rate: {rf_sample_rate / 1e6:.2f} MS/s")
        
        # Create DSP components
        # 1. Lowpass filter for channel selection (200 kHz bandwidth for FM)
        channel_filter = FIRLowpass(
            cutoff=100000,  # 100 kHz (200 kHz total bandwidth)
            sample_rate=rf_sample_rate,
            numtaps=201,
            window='hamming'
        )
        
        # 2. Resampler to audio rate
        resampler = Resampler(
            input_rate=rf_sample_rate,
            output_rate=audio_sample_rate,
            method='poly'
        )
        
        # 3. FM demodulator
        fm_demod = FMDemodulator(
            sample_rate=audio_sample_rate,
            deemphasis_tau=75e-6,  # North America standard
            audio_cutoff=15000,  # 15 kHz audio bandwidth
            audio_gain=0.5  # Adjust gain as needed
        )
        
        print("\nStarting FM demodulation...")
        print("Press Ctrl+C to stop\n")
        
        chunk_count = 0
        
        # Stream and process samples
        async for iq_samples in hw.stream_iq(chunk_size=16384):
            # Process RF samples
            # 1. Channel filter (select FM channel)
            filtered = channel_filter.filter(iq_samples)
            
            # 2. Resample to audio rate
            audio_rate_samples = resampler.resample(filtered)
            
            # 3. FM demodulate
            audio = fm_demod.demodulate(audio_rate_samples)
            
            # Process audio (example: calculate statistics)
            chunk_count += 1
            if chunk_count % 10 == 0:
                rms = np.sqrt(np.mean(audio ** 2))
                peak = np.max(np.abs(audio))
                print(f"Chunk {chunk_count}: RMS={rms:.4f}, Peak={peak:.4f}, "
                      f"Audio samples: {len(audio)}")
            
            # Here you would typically:
            # - Write audio to file (WAV)
            # - Play audio through soundcard
            # - Further audio processing
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        hw.close()
        print("Device closed")


async def simple_fm_demo():
    """Simple example with synthetic FM signal."""
    print("Generating synthetic FM signal...")
    
    # Parameters
    sample_rate = 240000  # 240 kHz
    duration = 1.0  # 1 second
    carrier_freq = 100000  # 100 kHz carrier
    modulation_freq = 1000  # 1 kHz audio tone
    modulation_index = 5.0  # FM modulation index
    
    # Generate time vector
    t = np.arange(int(sample_rate * duration)) / sample_rate
    
    # Generate FM signal: s(t) = A * cos(2*pi*fc*t + beta*sin(2*pi*fm*t))
    # In complex form: s(t) = A * exp(j * (2*pi*fc*t + beta*sin(2*pi*fm*t)))
    phase = 2.0 * np.pi * carrier_freq * t + modulation_index * np.sin(2.0 * np.pi * modulation_freq * t)
    fm_signal = np.exp(1j * phase).astype(np.complex64)
    
    print(f"Generated {len(fm_signal)} samples")
    
    # Demodulate
    fm_demod = FMDemodulator(
        sample_rate=sample_rate,
        deemphasis_tau=75e-6,
        audio_gain=1.0
    )
    
    audio = fm_demod.demodulate(fm_signal)
    
    print(f"Demodulated {len(audio)} audio samples")
    print(f"Audio dtype: {audio.dtype}")
    print(f"Audio range: [{np.min(audio):.4f}, {np.max(audio):.4f}]")
    print(f"Audio RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    
    # Verify we recovered the modulation tone
    # (In a real scenario, you'd do FFT to see the 1 kHz tone)
    print("\nFM demodulation complete!")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'synthetic':
        # Run synthetic signal demo
        asyncio.run(simple_fm_demo())
    else:
        # Run real hardware demo
        asyncio.run(fm_receiver_example())


if __name__ == "__main__":
    main()

