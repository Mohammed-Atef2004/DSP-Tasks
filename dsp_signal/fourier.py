import numpy as np
import math
from .signal import Signal

def dft_logic(samples, inverse=False):
    N = len(samples)
    output = np.zeros(N, dtype=complex)
    
    # Sign of the exponent: -1 for DFT, 1 for IDFT
    sign = 1 if inverse else -1
    
    for k in range(N):
        sum_val = 0j
        for n in range(N):
            angle = sign * 2 * math.pi * k * n / N
            sum_val += samples[n] * (math.cos(angle) + 1j * math.sin(angle))
        output[k] = sum_val
    
    if inverse:
        return (output / N).real  # IDFT returns real time-domain signal
    return output

def fourier_transform_signal(signal: Signal, sampling_freq: float):
    samples = np.array(signal.samples)
    dft_complex = dft_logic(samples, inverse=False)
    
    # Frequency bins calculation: f = k * (fs / N)
    N = len(samples)
    freq_bins = [k * (sampling_freq / N) for k in range(N)]
    
    magnitudes = np.abs(dft_complex)
    phases = np.angle(dft_complex) # Radians
    
    mag_sig = Signal(freq_bins, magnitudes.tolist(), f"Mag_{signal.name}")
    phase_sig = Signal(freq_bins, np.degrees(phases).tolist(), f"Phase_{signal.name}")
    
    return mag_sig, phase_sig

def inverse_fourier_transform(magnitude_signal: Signal, phase_signal: Signal):
    mags = np.array(magnitude_signal.samples)
    # Convert phase back to radians
    phases = np.radians(np.array(phase_signal.samples))
    
    # Reconstruct complex form: A * e^(j*theta)
    complex_spectrum = mags * (np.cos(phases) + 1j * np.sin(phases))
    
    reconstructed_samples = dft_logic(complex_spectrum, inverse=True)
    
    return Signal(list(range(len(reconstructed_samples))), 
                  reconstructed_samples.tolist(), "Reconstructed")