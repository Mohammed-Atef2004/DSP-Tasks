import numpy as np
import math
from .signal import Signal

def compute_correlation_faculty(signal1_dict, signal2_dict):
    # Convert dicts to arrays
    x = np.array(list(signal1_dict.values()))
    y = np.array(list(signal2_dict.values()))
    
    N = len(x)
    r_xy = []
    
    # Normalization: 1/N * sum(x*y) / (sqrt(Ex*Ey)/N)
    # Simplified: sum(x*y) / sqrt(Ex*Ey)
    norm = math.sqrt(np.sum(x**2) * np.sum(y**2))
    
    if norm == 0:
        return {float(i): 0.0 for i in range(N)}

    for j in range(N):
        sum_val = 0
        for n in range(N):
            # Circular correlation
            sum_val += x[n] * y[(n + j) % N]
        r_xy.append(sum_val / norm)
    
    return {float(i): val for i, val in enumerate(r_xy)}

def correlate_signals(signal1, signal2, faculty_format=True):
    """Wrapper to return a Signal object for the GUI."""
    corr_dict = compute_correlation_faculty(signal1.to_dict(), signal2.to_dict())
    indices = list(corr_dict.keys())
    samples = list(corr_dict.values())
    return Signal(indices, samples, f"Corr_{signal1.name}")

def estimate_time_delay(signal1_dict, signal2_dict, fs):
    """Computes lag and time delay."""
    corr_dict = compute_correlation_faculty(signal1_dict, signal2_dict)
    samples = list(corr_dict.values())
    max_lag = np.argmax(samples)
    delay = max_lag / fs
    return max_lag, delay, corr_dict