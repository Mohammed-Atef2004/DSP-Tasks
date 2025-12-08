import numpy as np
from typing import Tuple, Dict, List
import cmath
from .signal import Signal


def compute_dft(signal_dict: Dict[float, float], sampling_freq: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Discrete Fourier Transform of a signal.
    
    Args:
        signal_dict: Dictionary with indices as keys and samples as values
        sampling_freq: Sampling frequency in Hz
        
    Returns:
        Tuple of (frequencies, magnitudes, phases)
    """
    # Extract samples and sort by index
    indices = list(signal_dict.keys())
    samples = list(signal_dict.values())
    
    # Sort by index
    sorted_pairs = sorted(zip(indices, samples))
    indices_sorted = [p[0] for p in sorted_pairs]
    samples_sorted = [p[1] for p in sorted_pairs]
    
    N = len(samples_sorted)
    
    # Initialize DFT arrays
    dft_real = np.zeros(N)
    dft_imag = np.zeros(N)
    
    # Compute DFT
    for k in range(N):  # Frequency index
        sum_real = 0.0
        sum_imag = 0.0
        for n in range(N):  # Time index
            angle = -2 * np.pi * k * n / N
            sum_real += samples_sorted[n] * np.cos(angle)
            sum_imag += samples_sorted[n] * np.sin(angle)
        
        dft_real[k] = sum_real
        dft_imag[k] = sum_imag
    
    # Calculate frequencies (only up to Nyquist frequency)
    frequencies = np.arange(N) * sampling_freq / N
    
    # Calculate magnitude and phase
    magnitudes = np.sqrt(dft_real**2 + dft_imag**2)
    phases = np.arctan2(dft_imag, dft_real)  # in radians
    
    return frequencies[:N//2 + 1], magnitudes[:N//2 + 1], phases[:N//2 + 1]


def compute_idft(real_parts: List[float], imag_parts: List[float]) -> List[float]:
    """
    Compute Inverse Discrete Fourier Transform.
    
    Args:
        real_parts: Real components of frequency domain
        imag_parts: Imaginary components of frequency domain
        
    Returns:
        Reconstructed time-domain signal
    """
    N = len(real_parts)
    reconstructed = np.zeros(N)
    
    for n in range(N):  # Time index
        sum_real = 0.0
        for k in range(N):  # Frequency index
            angle = 2 * np.pi * k * n / N
            sum_real += real_parts[k] * np.cos(angle) - imag_parts[k] * np.sin(angle)
        
        reconstructed[n] = sum_real / N
    
    return reconstructed.tolist()


def smart_dft_idft(signal_dict: Dict[float, float], sampling_freq: float = 1.0, 
                  inverse: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smart function that can compute both DFT and IDFT.
    
    Args:
        signal_dict: Input signal (time domain for DFT, frequency domain for IDFT)
        sampling_freq: Sampling frequency in Hz
        inverse: If True, compute IDFT; if False, compute DFT
        
    Returns:
        For DFT: (frequencies, magnitudes, phases)
        For IDFT: (indices, reconstructed_samples, None)
    """
    if not inverse:
        # Compute DFT
        return compute_dft(signal_dict, sampling_freq)
    else:
        # For IDFT, we expect signal_dict to have complex values
        # But our dict format stores real values, so we need to adapt
        # We'll assume the input is magnitude/phase representation
        indices = list(signal_dict.keys())
        values = list(signal_dict.values())
        
        N = len(values)
        
        # Convert from polar to rectangular
        # Assuming values are magnitudes, we need phases too
        # For simplicity, let's assume 0 phase
        real_parts = values
        imag_parts = [0.0] * N
        
        # Compute IDFT
        reconstructed = compute_idft(real_parts, imag_parts)
        
        # Return indices and reconstructed signal
        return np.arange(N), np.array(reconstructed), None


def fourier_transform_signal(signal: Signal, sampling_freq: float) -> Tuple[Signal, Signal]:
    """
    Apply Fourier Transform to a signal and return magnitude and phase signals.
    
    Args:
        signal: Input time-domain signal
        sampling_freq: Sampling frequency in Hz
        
    Returns:
        Tuple of (magnitude_signal, phase_signal)
    """
    signal_dict = signal.to_dict()
    frequencies, magnitudes, phases = compute_dft(signal_dict, sampling_freq)
    
    # Create magnitude signal
    mag_signal = Signal(
        indices=list(range(len(magnitudes))),
        samples=magnitudes.tolist(),
        name=f"Mag_{signal.name}_fs{sampling_freq}Hz"
    )
    
    # Create phase signal (in degrees for better visualization)
    phases_deg = np.degrees(phases)
    phase_signal = Signal(
        indices=list(range(len(phases_deg))),
        samples=phases_deg.tolist(),
        name=f"Phase_{signal.name}_fs{sampling_freq}Hz"
    )
    
    return mag_signal, phase_signal


def inverse_fourier_transform(magnitude_signal: Signal, phase_signal: Signal = None) -> Signal:
    """
    Reconstruct signal from magnitude (and optionally phase).
    
    Args:
        magnitude_signal: Magnitude spectrum signal
        phase_signal: Phase spectrum signal (optional, defaults to 0 phase)
        
    Returns:
        Reconstructed time-domain signal
    """
    N = len(magnitude_signal.samples)
    
    if phase_signal is None:
        # Assume zero phase
        phases = np.zeros(N)
    else:
        # Convert phase from degrees back to radians
        phases = np.radians(phase_signal.samples)
    
    # Convert from polar to rectangular
    magnitudes = np.array(magnitude_signal.samples)
    real_parts = magnitudes * np.cos(phases)
    imag_parts = magnitudes * np.sin(phases)
    
    # Compute IDFT
    reconstructed = compute_idft(real_parts.tolist(), imag_parts.tolist())
    
    # Create reconstructed signal
    return Signal(
        indices=list(range(N)),
        samples=reconstructed,
        name=f"Reconstructed_{magnitude_signal.name}"
    )