import numpy as np
from typing import Tuple, Dict, List
import math
from .signal import Signal


def compute_dft_full(signal_dict: Dict[float, float], sampling_freq: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Extract samples and sort by index
    indices = list(signal_dict.keys())
    samples = list(signal_dict.values())
    
    # Sort by index
    sorted_pairs = sorted(zip(indices, samples))
    indices_sorted = [p[0] for p in sorted_pairs]
    samples_sorted = np.array([p[1] for p in sorted_pairs])
    
    N = len(samples_sorted)
    
    # Pre-allocate arrays
    frequencies = np.arange(N) * sampling_freq / N
    magnitudes = np.zeros(N)
    phases = np.zeros(N)
    
    # Compute DFT for ALL frequencies (0 to N-1)
    for k in range(N):  # ALL frequency indices
        sum_real = 0.0
        sum_imag = 0.0
        
        for n in range(N):  # Time index
            angle = -2 * math.pi * k * n / N
            sum_real += samples_sorted[n] * math.cos(angle)
            sum_imag += samples_sorted[n] * math.sin(angle)
        
        magnitudes[k] = math.sqrt(sum_real**2 + sum_imag**2)
        phases[k] = math.atan2(sum_imag, sum_real)
    
    return frequencies, magnitudes, phases


def compute_dft(signal_dict: Dict[float, float], sampling_freq: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, magnitudes, phases = compute_dft_full(signal_dict, sampling_freq)
    
    # Return only up to Nyquist frequency (N/2)
    N = len(frequencies)
    return frequencies[:N//2 + 1], magnitudes[:N//2 + 1], phases[:N//2 + 1]


def compute_idft(real_parts: List[float], imag_parts: List[float]) -> List[float]:
    N = len(real_parts)
    reconstructed = np.zeros(N)
    
    for n in range(N):  # Time index
        sum_real = 0.0
        for k in range(N):  # Frequency index
            angle = 2 * math.pi * k * n / N
            sum_real += real_parts[k] * math.cos(angle) - imag_parts[k] * math.sin(angle)
        
        reconstructed[n] = sum_real / N
    
    return reconstructed.tolist()


def compute_idft_full(magnitudes: List[float], phases: List[float]) -> List[float]:
    N = len(magnitudes)
    reconstructed = np.zeros(N)
    
    for n in range(N):  # Time index
        sum_val = 0.0
        for k in range(N):  # ALL frequency indices
            # Convert from polar to rectangular
            real = magnitudes[k] * math.cos(phases[k])
            imag = magnitudes[k] * math.sin(phases[k])
            
            # IDFT formula
            angle = 2 * math.pi * k * n / N
            sum_val += real * math.cos(angle) - imag * math.sin(angle)
        
        reconstructed[n] = sum_val / N
    
    return reconstructed.tolist()


def smart_dft_idft(signal_dict: Dict[float, float], sampling_freq: float = 1.0, 
                  inverse: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not inverse:
        # Compute DFT
        return compute_dft(signal_dict, sampling_freq)
    else:
        # For IDFT, we need to handle the input format
        indices = list(signal_dict.keys())
        values = list(signal_dict.values())
        
        # Sort by index
        sorted_pairs = sorted(zip(indices, values))
        sorted_indices = [p[0] for p in sorted_pairs]
        sorted_values = np.array([p[1] for p in sorted_pairs])
        
        N = len(sorted_values)
        
        # For IDFT from magnitude only (assuming 0 phase)
        real_parts = sorted_values.tolist()
        imag_parts = [0.0] * N
        
        # Compute IDFT
        reconstructed = compute_idft(real_parts, imag_parts)
        
        # Return as arrays
        return np.arange(N), np.array(reconstructed), None


def fourier_transform_signal(signal: Signal, sampling_freq: float) -> Tuple[Signal, Signal]:
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
    N = len(magnitude_signal.samples)
    
    if phase_signal is None:
        # Assume zero phase
        phases = np.zeros(N)
    else:
        phases = np.array(phase_signal.samples)
        # Check if phases are in degrees (typical range -180 to 180)
        if np.max(np.abs(phases)) > math.pi:
            # Convert from degrees to radians
            phases = np.radians(phases)
    
    magnitudes = np.array(magnitude_signal.samples)
    
    if N <= 9:
        # Likely a full spectrum, use as is
        reconstructed = compute_idft_full(magnitudes.tolist(), phases.tolist())
    else:
        is_conjugate_symmetric = True
        for i in range(1, min(5, N//2)):
            if abs(magnitudes[i] - magnitudes[N-i]) > 0.001:
                is_conjugate_symmetric = False
                break
        
        if is_conjugate_symmetric and N % 2 == 0:
            # Already looks like full spectrum
            reconstructed = compute_idft_full(magnitudes.tolist(), phases.tolist())
        else:
            # Probably only positive frequencies, create full spectrum
            full_N = (N - 1) * 2
            
            full_magnitudes = np.zeros(full_N)
            full_phases = np.zeros(full_N)
            
            # Copy positive frequencies
            full_magnitudes[:N] = magnitudes
            full_phases[:N] = phases
            
            # Create conjugate symmetric part
            for k in range(1, N-1):
                full_magnitudes[full_N - k] = magnitudes[k]
                full_phases[full_N - k] = -phases[k]
            
            magnitudes = full_magnitudes
            phases = full_phases
            N = full_N
            
            reconstructed = compute_idft_full(magnitudes.tolist(), phases.tolist())
    
    # Create reconstructed signal
    return Signal(
        indices=list(range(len(reconstructed))),
        samples=reconstructed,
        name=f"Reconstructed_{magnitude_signal.name}"
    )