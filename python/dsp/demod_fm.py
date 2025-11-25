"""
FM Demodulation

Implements FM (Frequency Modulation) demodulation using phase differentiation,
with de-emphasis filtering and normalized audio output.
"""

import numpy as np
from scipy import signal
from typing import Optional
from .filters import FIRLowpass


class FMDemodulator:
    """
    FM Demodulator using phase differentiation.
    
    Implements:
    - Phase differentiation for FM demodulation
    - De-emphasis filtering (75 μs for North America, 50 μs for Europe)
    - Audio normalization to float32
    """
    
    def __init__(
        self,
        sample_rate: float,
        deemphasis_tau: float = 75e-6,
        audio_cutoff: Optional[float] = None,
        audio_gain: float = 1.0
    ):
        """
        Initialize FM demodulator.
        
        Args:
            sample_rate: Input sample rate in Hz
            deemphasis_tau: De-emphasis time constant in seconds
                            (75e-6 for North America, 50e-6 for Europe)
            audio_cutoff: Audio lowpass cutoff frequency in Hz (default: sample_rate/2.56)
            audio_gain: Gain factor for output audio
        """
        self.sample_rate = sample_rate
        self.deemphasis_tau = deemphasis_tau
        self.audio_gain = audio_gain
        
        # Default audio cutoff is typically sample_rate / 2.56 for FM
        # This gives ~15 kHz audio bandwidth at 38.4 kHz sample rate
        if audio_cutoff is None:
            audio_cutoff = sample_rate / 2.56
        self.audio_cutoff = audio_cutoff
        
        # Previous sample for phase differentiation
        self.prev_sample: Optional[complex] = None
        
        # De-emphasis filter
        # De-emphasis is a first-order lowpass filter: H(s) = 1 / (1 + s*tau)
        # Convert to discrete-time using bilinear transform
        self._setup_deemphasis()
        
        # Optional audio lowpass filter (anti-aliasing)
        self.audio_filter = FIRLowpass(
            cutoff=audio_cutoff,
            sample_rate=sample_rate,
            numtaps=101,
            window='hamming'
        )
    
    def _setup_deemphasis(self):
        """Setup de-emphasis filter using bilinear transform."""
        # Continuous-time transfer function: H(s) = 1 / (1 + s*tau)
        # Bilinear transform: s = 2*fs * (z-1)/(z+1)
        # This gives us a digital IIR filter
        
        # Pre-warp the frequency for bilinear transform
        w_c = 1.0 / self.deemphasis_tau  # Cutoff frequency in rad/s
        fs = self.sample_rate
        
        # Bilinear transform coefficients
        # H(z) = (b0 + b1*z^-1) / (1 + a1*z^-1)
        # For de-emphasis: b0 = 1/(1+2*fs*tau), b1 = 1/(1+2*fs*tau)
        #                  a1 = (1-2*fs*tau)/(1+2*fs*tau)
        
        alpha = 2.0 * fs * self.deemphasis_tau
        self.deemph_b = np.array([1.0 / (1.0 + alpha), 1.0 / (1.0 + alpha)])
        self.deemph_a = np.array([1.0, (1.0 - alpha) / (1.0 + alpha)])
        
        # Initialize filter state
        self.deemph_zi = signal.lfilter_zi(self.deemph_b, self.deemph_a)
    
    def reset(self):
        """Reset demodulator state."""
        self.prev_sample = None
        self.deemph_zi = signal.lfilter_zi(self.deemph_b, self.deemph_a)
        self.audio_filter.reset_state()
    
    def demodulate(self, iq_samples: np.ndarray, apply_deemphasis: bool = True) -> np.ndarray:
        """
        Demodulate FM signal from IQ samples.
        
        Args:
            iq_samples: Complex IQ samples
            apply_deemphasis: If True, apply de-emphasis filter
            
        Returns:
            Demodulated audio signal as float32 array, normalized to [-1, 1]
        """
        if len(iq_samples) == 0:
            return np.array([], dtype=np.float32)
        
        # Phase differentiation for FM demodulation
        # FM signal: s(t) = A * exp(j * 2*pi * (fc*t + k*integral(m(t))))
        # Demodulation: m(t) = (1/(2*pi*k)) * d/dt(phase(s(t)))
        # We compute: arg(s[n] * conj(s[n-1])) = phase difference
        
        # Normalize samples to unit magnitude (remove amplitude variations)
        magnitude = np.abs(iq_samples)
        # Avoid division by zero
        magnitude = np.where(magnitude > 1e-10, magnitude, 1.0)
        normalized = iq_samples / magnitude
        
        # Compute phase difference
        if self.prev_sample is not None:
            # Multiply current sample by conjugate of previous
            phase_diff = normalized * np.conj(self.prev_sample)
        else:
            # First sample: use zero phase difference
            phase_diff = np.ones_like(normalized, dtype=complex)
        
        # Extract phase (argument)
        # np.angle gives phase in radians, which is proportional to frequency deviation
        audio = np.angle(phase_diff)
        
        # Store last sample for next call
        self.prev_sample = normalized[-1]
        
        # Apply de-emphasis filter
        if apply_deemphasis:
            audio, self.deemph_zi = signal.lfilter(
                self.deemph_b, self.deemph_a, audio, zi=self.deemph_zi
            )
        
        # Apply audio lowpass filter (anti-aliasing)
        audio = self.audio_filter.filter(audio, reset=False)
        
        # Normalize to [-1, 1] range
        # FM deviation is typically ±75 kHz for broadcast FM
        # We normalize based on the maximum expected phase difference
        # For a properly tuned signal, phase differences should be in [-pi, pi]
        # But we'll normalize based on actual range to handle any signal
        max_val = np.max(np.abs(audio))
        if max_val > 1e-10:
            audio = audio / max_val * self.audio_gain
        else:
            audio = audio * self.audio_gain
        
        # Clip to [-1, 1] range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to float32
        return audio.astype(np.float32)
    
    def __call__(self, iq_samples: np.ndarray, apply_deemphasis: bool = True) -> np.ndarray:
        """Allow demodulator to be called as a function."""
        return self.demodulate(iq_samples, apply_deemphasis)


def demodulate_fm(
    iq_samples: np.ndarray,
    sample_rate: float,
    deemphasis_tau: float = 75e-6,
    audio_cutoff: Optional[float] = None,
    audio_gain: float = 1.0
) -> np.ndarray:
    """
    Convenience function to demodulate FM signal.
    
    Args:
        iq_samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        deemphasis_tau: De-emphasis time constant in seconds
        audio_cutoff: Audio lowpass cutoff frequency in Hz
        audio_gain: Gain factor for output audio
        
    Returns:
        Demodulated audio signal as float32 array, normalized to [-1, 1]
    """
    demod = FMDemodulator(
        sample_rate=sample_rate,
        deemphasis_tau=deemphasis_tau,
        audio_cutoff=audio_cutoff,
        audio_gain=audio_gain
    )
    return demod.demodulate(iq_samples)

