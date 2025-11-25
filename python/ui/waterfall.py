"""
Real-time Waterfall Display

Displays frequency vs time waterfall plot using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import get_cmap
from collections import deque
from typing import Optional, Callable
import time
import threading


class WaterfallDisplay:
    """
    Real-time waterfall display for IQ samples.
    
    Updates at approximately 10 FPS by accumulating samples
    and displaying FFT results as a time-frequency plot.
    """
    
    def __init__(
        self,
        sample_rate: float,
        center_freq: float = 0.0,
        fft_size: int = 2048,
        num_lines: int = 200,
        update_rate: float = 10.0,
        colormap: str = 'viridis',
        title: str = "Waterfall Display"
    ):
        """
        Initialize waterfall display.
        
        Args:
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz (for display)
            fft_size: FFT size for frequency resolution
            num_lines: Number of time lines to display
            update_rate: Target update rate in FPS
            colormap: Matplotlib colormap name
            title: Window title
        """
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.fft_size = fft_size
        self.num_lines = num_lines
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        
        # Frequency axis
        self.freq_axis = np.fft.fftshift(
            np.fft.fftfreq(fft_size, 1.0 / sample_rate)
        ) + center_freq
        
        # Initialize display data
        self.waterfall_data = np.zeros((num_lines, fft_size), dtype=np.float32)
        self.sample_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(title)
        
        # Create colormap
        self.cmap = get_cmap(colormap)
        
        # Create image for waterfall
        self.im = self.ax.imshow(
            self.waterfall_data,
            aspect='auto',
            origin='lower',
            cmap=self.cmap,
            interpolation='nearest',
            extent=[
                self.freq_axis[0] / 1e6,  # Convert to MHz
                self.freq_axis[-1] / 1e6,
                0,
                num_lines * self.update_interval
            ]
        )
        
        # Labels and formatting
        self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel('Time (seconds)')
        self.ax.set_title(title)
        
        # Colorbar
        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.cbar.set_label('Power (dB)')
        
        # Animation
        self.animation: Optional[animation.FuncAnimation] = None
        self.last_update_time = time.time()
        self.running = False
        
        # Statistics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
    
    def add_samples(self, iq_samples: np.ndarray):
        """
        Add IQ samples to the buffer.
        
        Args:
            iq_samples: Complex IQ samples (numpy array)
        """
        with self.buffer_lock:
            self.sample_buffer.append(iq_samples.copy())
    
    def _process_buffer(self):
        """Process accumulated samples and update waterfall."""
        with self.buffer_lock:
            if not self.sample_buffer:
                return
            
            # Concatenate all buffered samples
            all_samples = np.concatenate(self.sample_buffer)
            self.sample_buffer.clear()
        
        # Process samples in FFT-sized chunks
        num_chunks = len(all_samples) // self.fft_size
        
        if num_chunks == 0:
            return
        
        # Process each chunk
        for i in range(num_chunks):
            chunk = all_samples[i * self.fft_size:(i + 1) * self.fft_size]
            
            # Compute FFT
            fft_result = np.fft.fft(chunk, n=self.fft_size)
            fft_magnitude = np.abs(fft_result)
            
            # Convert to dB and shift for display
            fft_db = 20 * np.log10(fft_magnitude + 1e-10)  # Add small value to avoid log(0)
            fft_db_shifted = np.fft.fftshift(fft_db)
            
            # Roll waterfall data up and add new line at bottom
            self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
            self.waterfall_data[-1, :] = fft_db_shifted
    
    def _update_display(self, frame):
        """Update function for matplotlib animation."""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        # Process buffer if enough time has passed
        if elapsed >= self.update_interval:
            self._process_buffer()
            
            # Update image data
            self.im.set_array(self.waterfall_data)
            
            # Update color scale (autoscale to current data range)
            vmin = np.percentile(self.waterfall_data, 5)
            vmax = np.percentile(self.waterfall_data, 95)
            self.im.set_clim(vmin, vmax)
            
            # Update FPS counter
            self.frame_count += 1
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
                self.ax.set_title(f"Waterfall Display - {self.fps:.1f} FPS")
            
            self.last_update_time = current_time
        
        return [self.im]
    
    def start(self):
        """Start the waterfall display animation."""
        if self.running:
            return
        
        self.running = True
        self.last_update_time = time.time()
        self.last_fps_time = time.time()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig,
            self._update_display,
            interval=int(self.update_interval * 1000),  # Convert to milliseconds
            blit=True,
            cache_frame_data=False
        )
        
        plt.show(block=False)
    
    def stop(self):
        """Stop the waterfall display."""
        self.running = False
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
    
    def close(self):
        """Close the display window."""
        self.stop()
        plt.close(self.fig)
    
    def set_center_frequency(self, freq_hz: float):
        """
        Update center frequency display.
        
        Args:
            freq_hz: Center frequency in Hz
        """
        self.center_freq = freq_hz
        self.freq_axis = np.fft.fftshift(
            np.fft.fftfreq(self.fft_size, 1.0 / self.sample_rate)
        ) + self.center_freq
        
        # Update image extent
        self.im.set_extent([
            self.freq_axis[0] / 1e6,
            self.freq_axis[-1] / 1e6,
            0,
            self.num_lines * self.update_interval
        ])
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_waterfall(
    sample_rate: float,
    center_freq: float = 0.0,
    fft_size: int = 2048,
    num_lines: int = 200,
    update_rate: float = 10.0,
    colormap: str = 'viridis'
) -> WaterfallDisplay:
    """
    Convenience function to create a waterfall display.
    
    Args:
        sample_rate: Sample rate in Hz
        center_freq: Center frequency in Hz
        fft_size: FFT size
        num_lines: Number of time lines
        update_rate: Update rate in FPS
        colormap: Colormap name
        
    Returns:
        WaterfallDisplay instance
    """
    return WaterfallDisplay(
        sample_rate=sample_rate,
        center_freq=center_freq,
        fft_size=fft_size,
        num_lines=num_lines,
        update_rate=update_rate,
        colormap=colormap
    )

