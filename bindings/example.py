#!/usr/bin/env python3
"""
Example usage of the sdrhw module
"""

import sdrhw
import numpy as np

def main():
    # Check available devices
    device_count = sdrhw.RtlSdrDriver.get_device_count()
    print(f"Found {device_count} RTL-SDR device(s)")
    
    if device_count == 0:
        print("No RTL-SDR devices found!")
        return
    
    # Print device names
    for i in range(device_count):
        name = sdrhw.RtlSdrDriver.get_device_name(i)
        print(f"  Device {i}: {name}")
    
    # Create driver instance
    driver = sdrhw.RtlSdrDriver(device_index=0)
    
    # Open device
    print("\nOpening device...")
    if not driver.open():
        print(f"Error opening device: {driver.get_last_error()}")
        return
    
    print("Device opened successfully")
    
    # Configure device
    print("\nConfiguring device...")
    frequency = 100000000  # 100 MHz
    sample_rate = 2048000  # 2.048 MS/s
    
    if not driver.set_freq(frequency):
        print(f"Error setting frequency: {driver.get_last_error()}")
        driver.close()
        return
    
    if not driver.set_sample_rate(sample_rate):
        print(f"Error setting sample rate: {driver.get_last_error()}")
        driver.close()
        return
    
    print(f"Frequency: {frequency} Hz")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Start streaming
    print("\nStarting streaming...")
    buffer_size = 1024 * 1024  # 1M samples
    if not driver.start_streaming(buffer_size):
        print(f"Error starting stream: {driver.get_last_error()}")
        driver.close()
        return
    
    print("Streaming started")
    
    # Read some samples
    print("\nReading samples...")
    import time
    time.sleep(0.1)  # Wait for some samples to accumulate
    
    max_samples = 1024
    samples = driver.read_samples(max_samples)
    
    print(f"Read {len(samples)} samples")
    print(f"Sample type: {samples.dtype}")
    print(f"Sample shape: {samples.shape}")
    print(f"First few samples: {samples[:5]}")
    
    # Check available samples
    available = driver.available_samples()
    print(f"Available samples in buffer: {available}")
    
    # Stop streaming
    print("\nStopping streaming...")
    driver.stop_streaming()
    print("Streaming stopped")
    
    # Close device
    print("\nClosing device...")
    driver.close()
    print("Device closed")

if __name__ == "__main__":
    main()

