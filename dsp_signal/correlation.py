import numpy as np
from typing import Tuple, Dict, List, Optional
import math
from .signal import Signal


def compute_correlation_faculty(signal1_dict: Dict[float, float], 
                               signal2_dict: Dict[float, float]) -> Dict[float, float]:
    # Convert dictionaries to sorted lists
    indices1 = list(signal1_dict.keys())
    samples1 = list(signal1_dict.values())
    indices2 = list(signal2_dict.keys())
    samples2 = list(signal2_dict.values())
    
    # Sort by indices
    pairs1 = sorted(zip(indices1, samples1))
    pairs2 = sorted(zip(indices2, samples2))
    
    x = np.array([p[1] for p in pairs1])  # Signal 1
    y = np.array([p[1] for p in pairs2])  # Signal 2
    
    N = len(x)
    
    # Compute FULL signal energy (for all samples, not just overlapping part)
    energy_x_full = np.sum(x ** 2)
    energy_y_full = np.sum(y ** 2)
    
    # Check for zero energy
    if energy_x_full == 0 or energy_y_full == 0:
        return {k: 0.0 for k in range(N)}
    
    norm_factor_full = math.sqrt(energy_x_full * energy_y_full)
    
    correlation = {}
    
    # For each lag k from 0 to N-1
    for k in range(N):
        sum_val = 0.0
        
        for n in range(N - k):
            sum_val += x[n + k] * y[n]
        
        correlation[k] = sum_val / norm_factor_full
    
    return correlation


def compute_correlation(signal1_dict: Dict[float, float], 
                       signal2_dict: Dict[float, float],
                       normalize: bool = True,
                       faculty_format: bool = True) -> Dict[float, float]:
    if faculty_format:
        return compute_correlation_faculty(signal1_dict, signal2_dict)
    
    # Original implementation for completeness
    # Convert dictionaries to sorted lists
    indices1 = list(signal1_dict.keys())
    samples1 = list(signal1_dict.values())
    indices2 = list(signal2_dict.keys())
    samples2 = list(signal2_dict.values())
    
    # Sort by indices
    pairs1 = sorted(zip(indices1, samples1))
    pairs2 = sorted(zip(indices2, samples2))
    
    x = np.array([p[1] for p in pairs1])
    y = np.array([p[1] for p in pairs2])
    
    N1 = len(x)
    N2 = len(y)
    N = max(N1, N2)
    
    # Pad signals to same length
    x_padded = np.pad(x, (0, N - N1), 'constant')
    y_padded = np.pad(y, (0, N - N2), 'constant')
    
    # Compute correlation for all lags
    correlation = {}
    max_lag = N - 1
    
    for lag in range(-max_lag, max_lag + 1):
        sum_val = 0.0
        
        for n in range(N):
            idx1 = n
            idx2 = n - lag
            
            if 0 <= idx1 < N and 0 <= idx2 < N:
                sum_val += x_padded[idx1] * y_padded[idx2]
        
        correlation[lag] = float(sum_val)
    
    # Normalize if requested
    if normalize:
        energy_x = np.sum(x_padded ** 2)
        energy_y = np.sum(y_padded ** 2)
        
        if energy_x > 0 and energy_y > 0:
            norm_factor = np.sqrt(energy_x * energy_y)
            for lag in correlation:
                correlation[lag] /= norm_factor
    
    return correlation


def compute_autocorrelation(signal_dict: Dict[float, float], 
                           normalize: bool = True) -> Dict[float, float]:
    return compute_correlation(signal_dict, signal_dict, normalize, faculty_format=True)


def find_correlation_peak(correlation_dict: Dict[float, float]) -> Tuple[float, float]:
    if not correlation_dict:
        return 0.0, 0.0
    
    # Find maximum value (for positive correlation)
    max_value = -float('inf')
    max_lag = 0
    
    for lag, value in correlation_dict.items():
        if value > max_value:
            max_value = value
            max_lag = lag
    
    return float(max_lag), max_value


def estimate_time_delay(signal1_dict: Dict[float, float], 
                       signal2_dict: Dict[float, float],
                       sampling_freq: float = 1.0) -> Tuple[float, float, Dict[float, float]]:
    # Compute cross-correlation using faculty format
    correlation_dict = compute_correlation_faculty(signal1_dict, signal2_dict)
    
    # Find peak (excluding lag 0 if possible, but include it)
    max_corr = -float('inf')
    max_lag = 0
    
    for lag, corr in correlation_dict.items():
        if corr > max_corr:
            max_corr = corr
            max_lag = lag
    
    # Convert lag to time delay
    delay_seconds = max_lag / sampling_freq
    
    return max_lag, delay_seconds, correlation_dict


def correlate_signals(signal1: Signal, signal2: Signal, faculty_format: bool = True) -> Signal:
    signal1_dict = signal1.to_dict()
    signal2_dict = signal2.to_dict()
    
    correlation_dict = compute_correlation(signal1_dict, signal2_dict, 
                                         faculty_format=faculty_format)
    
    # Convert to Signal object
    indices = list(correlation_dict.keys())
    samples = list(correlation_dict.values())
    
    return Signal(indices, samples, f"Corr_{signal1.name}_{signal2.name}")


def autocorrelate_signal(signal: Signal) -> Signal:
    signal_dict = signal.to_dict()
    correlation_dict = compute_autocorrelation(signal_dict, normalize=True)
    
    indices = list(correlation_dict.keys())
    samples = list(correlation_dict.values())
    
    return Signal(indices, samples, f"AutoCorr_{signal.name}")