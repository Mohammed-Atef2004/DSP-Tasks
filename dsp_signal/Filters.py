import numpy as np
import math
from .signal import Signal
from .operations import convolve_signals
from .fourier import dft_logic

def window_function(attenuation, n, N):
    """
    Returns the window weight w(n) for a given index n and length N
    based on the required stopband attenuation.
    """
    # Using n + (N-1)/2 to shift window to be centered at 0 if needed, 
    # but here we use n directly as the symmetric index from -(N-1)/2 to (N-1)/2
    if attenuation <= 21:
        return 1  # Rectangular
    elif attenuation <= 44:
        return 0.5 + 0.5 * math.cos(2 * math.pi * n / N)  # Hanning
    elif attenuation <= 53:
        return 0.54 + 0.46 * math.cos(2 * math.pi * n / N)  # Hamming
    elif attenuation <= 74:
        return 0.42 + 0.5 * math.cos(2 * math.pi * n / (N - 1)) + \
               0.08 * math.cos(4 * math.pi * n / (N - 1))  # Blackman
    return 1

def design_fir_filter(filter_type, fs, fc=None, f1=None, f2=None, attenuation=44, transband=500):
    """
    Designs FIR filter coefficients h(n) using the Window Method.
    """
    # 1. Normalize transition band
    delta_f = transband / fs

    # 2. Determine N based on Attenuation
    if attenuation <= 21: N = int(np.ceil(0.9 / delta_f))
    elif attenuation <= 44: N = int(np.ceil(3.1 / delta_f))
    elif attenuation <= 53: N = int(np.ceil(3.3 / delta_f))
    elif attenuation <= 74: N = int(np.ceil(5.5 / delta_f))
    else: N = int(np.ceil(3.1 / delta_f))

    # 3. Ensure N is odd (Faculty requirement)
    if N % 2 == 0:
        N += 1

    half_n = (N - 1) // 2
    indices = list(range(-half_n, half_n + 1))
    h_coeffs = []

    # 4. Compute coefficients
    for n in indices:
        w = window_function(attenuation, n, N)
        
        if filter_type.lower() == 'low':
            fc_norm = (fc + transband / 2) / fs  # Adjust with half transition
            if n == 0:
                hd = 2 * fc_norm
            else:
                hd = 2 * fc_norm * (math.sin(n * 2 * math.pi * fc_norm) / (n * 2 * math.pi * fc_norm))

        elif filter_type.lower() == 'high':
            fc_norm = (fc - transband / 2) / fs
            if n == 0:
                hd = 1 - 2 * fc_norm
            else:
                hd = -2 * fc_norm * (math.sin(n * 2 * math.pi * fc_norm) / (n * 2 * math.pi * fc_norm))

        elif filter_type.lower() == 'bandpass':
            f1_norm = (f1 - transband / 2) / fs
            f2_norm = (f2 + transband / 2) / fs
            if n == 0:
                hd = 2 * (f2_norm - f1_norm)
            else:
                term2 = 2 * f2_norm * (math.sin(n * 2 * math.pi * f2_norm) / (n * 2 * math.pi * f2_norm))
                term1 = 2 * f1_norm * (math.sin(n * 2 * math.pi * f1_norm) / (n * 2 * math.pi * f1_norm))
                hd = term2 - term1

        elif filter_type.lower() == 'bandstop':
            f1_norm = (f1 + transband / 2) / fs
            f2_norm = (f2 - transband / 2) / fs
            if n == 0:
                hd = 1 - 2 * (f2_norm - f1_norm)
            else:
                term1 = 2 * f1_norm * (math.sin(n * 2 * math.pi * f1_norm) / (n * 2 * math.pi * f1_norm))
                term2 = 2 * f2_norm * (math.sin(n * 2 * math.pi * f2_norm) / (n * 2 * math.pi * f2_norm))
                hd = term1 - term2
        
        h_coeffs.append(hd * w)

    return Signal(indices, h_coeffs, f"FIR_{filter_type}")

def apply_fast_filtering(input_signal, filter_signal):
    """
    Fast Method: Frequency Domain Filtering.
    Multiplication in frequency domain using your DFT logic.
    """
    x = input_signal.samples
    h = filter_signal.samples
    
    # Linear convolution size
    N_total = len(x) + len(h) - 1
    
    # Zero Padding
    x_padded = np.pad(x, (0, N_total - len(x)))
    h_padded = np.pad(h, (0, N_total - len(h)))
    
    # DFT -> Multiply -> IDFT
    X_freq = dft_logic(x_padded, inverse=False)
    H_freq = dft_logic(h_padded, inverse=False)
    
    Y_freq = X_freq * H_freq
    
    y_time = dft_logic(Y_freq, inverse=True).real
    
    # Calculate starting index (usually 0 for linear convolution result)
    indices = list(range(len(y_time)))
    
    return Signal(indices, y_time.tolist(), "Fast_Filtered_Signal")