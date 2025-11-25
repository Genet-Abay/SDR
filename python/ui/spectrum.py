"""
Real-time Spectrum Display

Displays FFT magnitude spectrum with autoscaling and peak marker.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Tuple
import time
import threading


class SpectrumDisplay:
    """
    Real-time spectrum display for IQ samples.
    
    Shows FFT magnitude with autoscaling and peak frequency marker.
    """
    
    def __init__(
        self,
        sample_rate: float,
        center_freq: float = 0.0,
        fft_size: int = 2048,
        update_rate: float = 10.0,
        title: str = "Spectrum Display",
        show_peak: bool = True
    ):
        """
        Initialize spectrum display.
        
        Args:
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz (for display)
            fft_size: FFT size for frequency resolution
            update_rate: Target update rate in FPS
            title: Window title
            show_peak: If True, show peak frequency marker
        """
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.fft_size = fft_size
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        self.show_peak = show_peak
        
        # Frequency axis
        self.freq_axis = np.fft.fftshift(
            np.fft.fftfreq(fft_size, 1.0 / sample_rate)
        ) + center_freq
        
        # Initialize display data
        self.spectrum_data = np.zeros(fft_size, dtype=np.float32)
        self.sample_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.canvas.manager.set_window_title(title)
        
        # Create line plot
        self.line, = self.ax.plot(
            self.freq_axis / 1e6,  # Convert to MHz
            self.spectrum_data,
            'b-',
            linewidth=1.0,
            label='Magnitude'
        )
        
        # Peak marker
        self.peak_marker = None
        self.peak_text = None
        if self.show_peak:
            self.peak_marker, = self.ax.plot(
                [0], [0], 'ro', markersize=10, label='Peak'
            )
            self.peak_text = self.ax.text(
                0, 0, '', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
            )
        
        # Labels and formatting
        self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel('Magnitude (dB)')
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Autoscale settings
        self.autoscale_enabled = True
        self.ymin = None
        self.ymax = None
        
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
        """Process accumulated samples and update spectrum."""
        with self.buffer_lock:
            if not self.sample_buffer:
                return
            
            # Concatenate all buffered samples
            all_samples = np.concatenate(self.sample_buffer)
            self.sample_buffer.clear()
        
        # Take enough samples for at least one FFT
        if len(all_samples) < self.fft_size:
            return
        
        # Use the most recent FFT_size samples
        samples = all_samples[-self.fft_size:]
        
        # Compute FFT
        fft_result = np.fft.fft(samples, n=self.fft_size)
        fft_magnitude = np.abs(fft_result)
        
        # Convert to dB
        fft_db = 20 * np.log10(fft_magnitude + 1e-10)  # Add small value to avoid log(0)
        fft_db_shifted = np.fft.fftshift(fft_db)
        
        # Update spectrum data
        self.spectrum_data = fft_db_shifted
    
    def _find_peak(self) -> Tuple[float, float]:
        """
        Find peak frequency and magnitude.
        
        Returns:
            (peak_freq_mhz, peak_magnitude_db)
        """
        peak_idx = np.argmax(self.spectrum_data)
        peak_freq = self.freq_axis[peak_idx]
        peak_mag = self.spectrum_data[peak_idx]
        return peak_freq / 1e6, peak_mag
    
    def _update_display(self, frame):
        """Update function for matplotlib animation."""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        # Process buffer if enough time has passed
        if elapsed >= self.update_interval:
            self._process_buffer()
            
            # Update line data
            self.line.set_ydata(self.spectrum_data)
            
            # Autoscale Y axis
            if self.autoscale_enabled:
                # Use percentile-based scaling to avoid outliers
                ymin = np.percentile(self.spectrum_data, 1)
                ymax = np.percentile(self.spectrum_data, 99)
                
                # Add some margin
                yrange = ymax - ymin
                ymin -= yrange * 0.1
                ymax += yrange * 0.1
                
                self.ax.set_ylim(ymin, ymax)
                self.ymin = ymin
                self.ymax = ymax
            
            # Update peak marker
            if self.show_peak and np.any(self.spectrum_data > -np.inf):
                peak_freq_mhz, peak_mag = self._find_peak()
                
                self.peak_marker.set_data([peak_freq_mhz], [peak_mag])
                
                # Update peak text
                self.peak_text.set_text(
                    f'Peak: {peak_freq_mhz:.3f} MHz\n{peak_mag:.1f} dB'
                )
                self.peak_text.set_position((peak_freq_mhz, peak_mag))
            
            # Update FPS counter
            self.frame_count += 1
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
                self.ax.set_title(f"Spectrum Display - {self.fps:.1f} FPS")
            
            self.last_update_time = current_time
        
        return [self.line, self.peak_marker, self.peak_text] if self.show_peak else [self.line]
    
    def start(self):
        """Start the spectrum display animation."""
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
        """Stop the spectrum display."""
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
        ) + freq_hz
        
        # Update line data
        self.line.set_xdata(self.freq_axis / 1e6)
    
    def set_autoscale(self, enabled: bool):
        """
        Enable or disable autoscaling.
        
        Args:
            enabled: If True, enable autoscaling
        """
        self.autoscale_enabled = enabled
        if not enabled and self.ymin is not None and self.ymax is not None:
            self.ax.set_ylim(self.ymin, self.ymax)
    
    def set_ylim(self, ymin: float, ymax: float):
        """
        Set Y-axis limits manually.
        
        Args:
            ymin: Minimum Y value
            ymax: Maximum Y value
        """
        self.autoscale_enabled = False
        self.ymin = ymin
        self.ymax = ymax
        self.ax.set_ylim(ymin, ymax)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_spectrum(
    sample_rate: float,
    center_freq: float = 0.0,
    fft_size: int = 2048,
    update_rate: float = 10.0,
    show_peak: bool = True
) -> SpectrumDisplay:
    """
    Convenience function to create a spectrum display.
    
    Args:
        sample_rate: Sample rate in Hz
        center_freq: Center frequency in Hz
        fft_size: FFT size
        update_rate: Update rate in FPS
        show_peak: Show peak marker
        
    Returns:
        SpectrumDisplay instance
    """
    return SpectrumDisplay(
        sample_rate=sample_rate,
        center_freq=center_freq,
        fft_size=fft_size,
        update_rate=update_rate,
        show_peak=show_peak
    )

