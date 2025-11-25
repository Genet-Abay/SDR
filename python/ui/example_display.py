#!/usr/bin/env python3
"""
Example: Real-time Waterfall and Spectrum Display

Demonstrates using the UI components with live SDR data.
"""

import asyncio
import numpy as np
from acquisition import HardwareInterface, choose_device
from ui import WaterfallDisplay, SpectrumDisplay


async def display_example():
    """Example with both waterfall and spectrum displays."""
    # Choose device
    device = choose_device()
    if device is None:
        print("No device available")
        return
    
    # Configuration
    center_freq = 100000000  # 100 MHz
    sample_rate = 2048000  # 2.048 MS/s
    
    # Create hardware interface
    hw = HardwareInterface(device_index=device.index)
    
    try:
        # Open and configure device
        hw.open()
        hw.set_frequency(center_freq)
        hw.set_sample_rate(sample_rate)
        
        print(f"Tuned to {center_freq / 1e6:.2f} MHz")
        print(f"Sample rate: {sample_rate / 1e6:.2f} MS/s")
        
        # Create displays
        waterfall = WaterfallDisplay(
            sample_rate=sample_rate,
            center_freq=center_freq,
            fft_size=2048,
            num_lines=200,
            update_rate=10.0,
            colormap='viridis',
            title="RTL-SDR Waterfall"
        )
        
        spectrum = SpectrumDisplay(
            sample_rate=sample_rate,
            center_freq=center_freq,
            fft_size=2048,
            update_rate=10.0,
            title="RTL-SDR Spectrum",
            show_peak=True
        )
        
        # Start displays
        waterfall.start()
        spectrum.start()
        
        print("\nDisplays started. Close windows or press Ctrl+C to stop.\n")
        
        # Stream samples and feed to displays
        async for iq_samples in hw.stream_iq(chunk_size=16384):
            # Feed samples to both displays
            waterfall.add_samples(iq_samples)
            spectrum.add_samples(iq_samples)
            
            # Small delay to allow GUI updates
            await asyncio.sleep(0.01)
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        waterfall.close()
        spectrum.close()
        hw.close()
        print("Displays closed, device closed")


async def waterfall_only_example():
    """Example with only waterfall display."""
    device = choose_device()
    if device is None:
        return
    
    hw = HardwareInterface(device_index=device.index)
    
    try:
        hw.open()
        hw.set_frequency(100000000)
        hw.set_sample_rate(2048000)
        
        # Create and start waterfall
        with WaterfallDisplay(
            sample_rate=2048000,
            center_freq=100000000,
            fft_size=2048,
            num_lines=200,
            update_rate=10.0
        ) as waterfall:
            
            print("Waterfall display running. Close window to stop.")
            
            async for iq_samples in hw.stream_iq(chunk_size=16384):
                waterfall.add_samples(iq_samples)
                await asyncio.sleep(0.01)
        
    finally:
        hw.close()


async def spectrum_only_example():
    """Example with only spectrum display."""
    device = choose_device()
    if device is None:
        return
    
    hw = HardwareInterface(device_index=device.index)
    
    try:
        hw.open()
        hw.set_frequency(100000000)
        hw.set_sample_rate(2048000)
        
        # Create and start spectrum
        with SpectrumDisplay(
            sample_rate=2048000,
            center_freq=100000000,
            fft_size=2048,
            update_rate=10.0,
            show_peak=True
        ) as spectrum:
            
            print("Spectrum display running. Close window to stop.")
            
            async for iq_samples in hw.stream_iq(chunk_size=16384):
                spectrum.add_samples(iq_samples)
                await asyncio.sleep(0.01)
        
    finally:
        hw.close()


def main():
    """Main entry point."""
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    
    if mode == 'waterfall':
        asyncio.run(waterfall_only_example())
    elif mode == 'spectrum':
        asyncio.run(spectrum_only_example())
    else:
        asyncio.run(display_example())


if __name__ == "__main__":
    main()

