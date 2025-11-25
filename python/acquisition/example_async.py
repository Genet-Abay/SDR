#!/usr/bin/env python3
"""
Example usage of the acquisition module with async streaming
"""

import asyncio
import numpy as np
from acquisition import HardwareInterface, DeviceManager, choose_device


async def process_samples(samples: np.ndarray):
    """
    Process a chunk of IQ samples.
    
    Args:
        samples: NumPy array of complex64 samples
    """
    # Calculate power spectrum
    power = np.abs(samples) ** 2
    avg_power = np.mean(power)
    
    # Calculate some statistics
    magnitude = np.abs(samples)
    phase = np.angle(samples)
    
    print(f"Received {len(samples)} samples - "
          f"Avg power: {avg_power:.6f}, "
          f"Magnitude range: [{np.min(magnitude):.3f}, {np.max(magnitude):.3f}]")


async def stream_example():
    """Example of async IQ streaming."""
    # List available devices
    print("Available devices:")
    DeviceManager.print_devices()
    
    # Choose a device (or use default)
    device_info = choose_device()
    if device_info is None:
        print("No device available")
        return
    
    print(f"\nUsing device: {device_info}")
    
    # Create hardware interface
    hw = HardwareInterface(device_index=device_info.index)
    
    try:
        # Open device
        if not hw.open():
            print(f"Failed to open device: {hw.get_last_error()}")
            return
        
        print("Device opened successfully")
        
        # Configure device
        frequency = 100000000  # 100 MHz
        sample_rate = 2048000  # 2.048 MS/s
        
        if not hw.set_frequency(frequency):
            print(f"Failed to set frequency: {hw.get_last_error()}")
            return
        
        if not hw.set_sample_rate(sample_rate):
            print(f"Failed to set sample rate: {hw.get_last_error()}")
            return
        
        print(f"Configured: {frequency} Hz @ {sample_rate} Hz")
        
        # Stream samples asynchronously
        print("\nStarting async stream...")
        print("Press Ctrl+C to stop\n")
        
        chunk_count = 0
        total_samples = 0
        
        async for samples in hw.stream_iq(chunk_size=8192):
            # Process the samples
            await process_samples(samples)
            
            chunk_count += 1
            total_samples += len(samples)
            
            # Stop after 100 chunks (or use Ctrl+C)
            if chunk_count >= 100:
                print(f"\nProcessed {chunk_count} chunks, {total_samples} total samples")
                break
        
    except KeyboardInterrupt:
        print("\n\nStreaming interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Cleanup
        hw.close()
        print("Device closed")


async def stream_with_timeout():
    """Example with timeout handling."""
    device_info = choose_device()
    if device_info is None:
        return
    
    hw = HardwareInterface(device_index=device_info.index)
    
    try:
        hw.open()
        hw.set_frequency(100000000)
        hw.set_sample_rate(2048000)
        
        # Stream for 5 seconds
        import time
        start_time = time.time()
        duration = 5.0
        
        async for samples in hw.stream_iq(chunk_size=16384):
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            
            # Process samples
            power = np.mean(np.abs(samples) ** 2)
            print(f"[{elapsed:.2f}s] {len(samples)} samples, avg power: {power:.6f}")
        
        print(f"\nStreamed for {time.time() - start_time:.2f} seconds")
        
    finally:
        hw.close()


def main():
    """Main entry point."""
    print("RTL-SDR Async Streaming Example")
    print("=" * 40)
    
    # Run the async example
    asyncio.run(stream_example())


if __name__ == "__main__":
    main()

