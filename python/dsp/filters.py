"""
Digital Signal Processing Filters

Provides FIR filters, frequency shifting, and resampling functions.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
import warnings


class FIRLowpass:
    """
    FIR Lowpass Filter using scipy.signal.firwin
    
    Implements a finite impulse response lowpass filter with configurable
    cutoff frequency, transition bandwidth, and window function.
    """
    
    def __init__(
        self,
        cutoff: float,
        sample_rate: float,
        numtaps: Optional[int] = None,
        width: Optional[float] = None,
        window: str = 'hamming',
        pass_zero: bool = True
    ):
        """
        Initialize FIR lowpass filter.
        
        Args:
            cutoff: Cutoff frequency in Hz
            sample_rate: Sample rate in Hz
            numtaps: Number of filter taps (auto-calculated if None)
            width: Transition bandwidth in Hz (used to calculate numtaps if numtaps is None)
            window: Window function ('hamming', 'hann', 'blackman', etc.)
            pass_zero: If True, pass zero frequency (lowpass), else stop zero (highpass)
        """
        self.cutoff = cutoff
        self.sample_rate = sample_rate
        self.window = window
        self.pass_zero = pass_zero
        
        # Normalize cutoff to Nyquist frequency (0-1 range)
        nyquist = sample_rate / 2.0
        if cutoff >= nyquist:
            raise ValueError(f"Cutoff frequency {cutoff} Hz must be less than Nyquist {nyquist} Hz")
        
        self.normalized_cutoff = cutoff / nyquist
        
        # Calculate numtaps if not provided
        if numtaps is None:
            if width is None:
                # Default: use 5% of cutoff as transition width
                width = cutoff * 0.05
            # Estimate numtaps based on transition width
            # Rule of thumb: numtaps ≈ 4 / (transition_width / sample_rate)
            transition_ratio = width / sample_rate
            numtaps = int(4.0 / transition_ratio)
            # Ensure odd number for Type I filter
            if numtaps % 2 == 0:
                numtaps += 1
            # Clamp to reasonable range
            numtaps = max(11, min(numtaps, 10001))
        
        self.numtaps = numtaps
        
        # Design the filter
        self.taps = signal.firwin(
            numtaps,
            self.normalized_cutoff,
            window=window,
            pass_zero=pass_zero
        )
        
        # Create filter state for streaming
        self.zi = None
        self.reset_state()
    
    def reset_state(self):
        """Reset filter state (useful for streaming)."""
        self.zi = signal.lfilter_zi(self.taps, 1.0)
    
    def filter(self, x: np.ndarray, reset: bool = False) -> np.ndarray:
        """
        Apply filter to signal.
        
        Args:
            x: Input signal (complex or real)
            reset: If True, reset filter state before filtering
            
        Returns:
            Filtered signal (same dtype as input)
        """
        if reset:
            self.reset_state()
        
        if np.iscomplexobj(x):
            # Filter real and imaginary parts separately
            real_filtered, self.zi = signal.lfilter(
                self.taps, 1.0, x.real, zi=self.zi
            )
            imag_filtered, self.zi = signal.lfilter(
                self.taps, 1.0, x.imag, zi=self.zi
            )
            return real_filtered + 1j * imag_filtered
        else:
            # Real signal
            filtered, self.zi = signal.lfilter(
                self.taps, 1.0, x, zi=self.zi
            )
            return filtered
    
    def __call__(self, x: np.ndarray, reset: bool = False) -> np.ndarray:
        """Allow filter to be called as a function."""
        return self.filter(x, reset)


def frequency_shift(
    signal_in: np.ndarray,
    frequency_hz: float,
    sample_rate: float
) -> np.ndarray:
    """
    Frequency shift (mix) a signal by a given frequency offset.
    
    This multiplies the signal by a complex exponential to shift it in frequency.
    Positive frequency_hz shifts up, negative shifts down.
    
    Args:
        signal_in: Input signal (complex or real)
        frequency_hz: Frequency offset in Hz (positive = up, negative = down)
        sample_rate: Sample rate in Hz
        
    Returns:
        Frequency-shifted signal (complex if input is complex, complex if input is real)
    """
    # Generate complex exponential for frequency shift
    t = np.arange(len(signal_in)) / sample_rate
    shift_phasor = np.exp(1j * 2.0 * np.pi * frequency_hz * t)
    
    # Multiply signal by phasor
    if np.iscomplexobj(signal_in):
        return signal_in * shift_phasor
    else:
        # Convert real signal to complex for mixing
        return signal_in.astype(np.complex64) * shift_phasor


class Resampler:
    """
    Signal resampler using scipy.signal.resample_poly or resample.
    
    Provides efficient resampling with rational or arbitrary rate changes.
    """
    
    def __init__(
        self,
        input_rate: float,
        output_rate: float,
        method: str = 'poly'
    ):
        """
        Initialize resampler.
        
        Args:
            input_rate: Input sample rate in Hz
            output_rate: Output sample rate in Hz
            method: Resampling method ('poly' for rational, 'fft' for arbitrary)
        """
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.method = method
        
        # Calculate resampling ratio
        self.ratio = output_rate / input_rate
        
        if method == 'poly':
            # For polyphase resampling, find rational approximation
            # This is more efficient for streaming
            from fractions import Fraction
            try:
                frac = Fraction(self.ratio).limit_denominator(1000)
                self.up = frac.numerator
                self.down = frac.denominator
            except:
                # Fallback to simple ratio
                self.up = int(np.round(self.ratio * 1000))
                self.down = 1000
        else:
            self.up = None
            self.down = None
    
    def resample(self, x: np.ndarray) -> np.ndarray:
        """
        Resample signal.
        
        Args:
            x: Input signal
            
        Returns:
            Resampled signal (same dtype as input)
        """
        if self.method == 'poly' and self.up is not None and self.down is not None:
            # Polyphase resampling (efficient for rational ratios)
            return signal.resample_poly(x, self.up, self.down)
        else:
            # FFT-based resampling (works for any ratio)
            num_samples = int(len(x) * self.ratio)
            return signal.resample(x, num_samples)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow resampler to be called as a function."""
        return self.resample(x)


def create_lowpass(
    cutoff: float,
    sample_rate: float,
    numtaps: Optional[int] = None,
    width: Optional[float] = None,
    window: str = 'hamming'
) -> FIRLowpass:
    """
    Convenience function to create a FIR lowpass filter.
    
    Args:
        cutoff: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        numtaps: Number of filter taps
        width: Transition bandwidth in Hz
        window: Window function
        
    Returns:
        FIRLowpass filter instance
    """
    return FIRLowpass(cutoff, sample_rate, numtaps, width, window)


def create_resampler(
    input_rate: float,
    output_rate: float,
    method: str = 'poly'
) -> Resampler:
    """
    Convenience function to create a resampler.
    
    Args:
        input_rate: Input sample rate in Hz
        output_rate: Output sample rate in Hz
        method: Resampling method
        
    Returns:
        Resampler instance
    """
    return Resampler(input_rate, output_rate, method)

