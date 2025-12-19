import numpy as np
import math
from .signal import Signal

def compute_correlation_values(x, y):
    N = len(x)
    # Ensure signals are same length for circular/direct correlation
    r_xy = []
    
    # Faculty Normalization Factor
    norm = math.sqrt(np.sum(x**2) * np.sum(y**2)) / N
    
    for j in range(N):
        sum_val = 0
        for n in range(N):
            # Using periodic/circular indexing for direct correlation
            sum_val += x[n] * y[(n + j) % N]
        r_xy.append((sum_val / N) / norm)
    
    return r_xy

def classify_signal(signal, class_a_template, class_b_template):
    """Classifies signal based on max correlation with category templates."""
    corr_a = compute_correlation_values(np.array(signal.samples), np.array(class_a_template.samples))
    corr_b = compute_correlation_values(np.array(signal.samples), np.array(class_b_template.samples))
    
    max_a = max(corr_a)
    max_b = max(corr_b)
    
    return "Class A" if max_a > max_b else "Class B"

def estimate_time_delay(signal1, signal2, fs):
    """Estimates delay in seconds based on max correlation peak."""
    r_xy = compute_correlation_values(np.array(signal1.samples), np.array(signal2.samples))
    max_lag = np.argmax(r_xy)
    delay = max_lag / fs
    return max_lag, delay