import numpy as np
from typing import Tuple, Dict, List, Optional
import math
from .signal import Signal


def estimate_snr(signal_dict: Dict[float, float], 
                noise_dict: Optional[Dict[float, float]] = None) -> float:
    # Convert to arrays
    indices = list(signal_dict.keys())
    samples = list(signal_dict.values())
    
    pairs = sorted(zip(indices, samples))
    samples_sorted = np.array([p[1] for p in pairs])
    
    if noise_dict is None:
        # Assume last half of signal is noise
        N = len(samples_sorted)
        signal_part = samples_sorted[:N//2]
        noise_part = samples_sorted[N//2:]
    else:
        # Use provided noise
        noise_indices = list(noise_dict.keys())
        noise_samples = list(noise_dict.values())
        noise_pairs = sorted(zip(noise_indices, noise_samples))
        noise_sorted = np.array([p[1] for p in noise_pairs])
        
        signal_part = samples_sorted
        noise_part = noise_sorted
    
    # Calculate power
    signal_power = np.mean(signal_part ** 2) if len(signal_part) > 0 else 0
    noise_power = np.mean(noise_part ** 2) if len(noise_part) > 0 else 0
    
    if noise_power == 0:
        return float('inf')
    
    # Calculate SNR in dB
    snr_linear = signal_power / noise_power
    snr_db = 10 * math.log10(snr_linear) if snr_linear > 0 else -float('inf')
    
    return snr_db


def estimate_signal_energy(signal_dict: Dict[float, float]) -> float:
    samples = list(signal_dict.values())
    return float(np.sum(np.array(samples) ** 2))


def estimate_signal_power(signal_dict: Dict[float, float]) -> float:
    samples = list(signal_dict.values())
    if len(samples) == 0:
        return 0.0
    return float(np.mean(np.array(samples) ** 2))


def estimate_sampling_frequency(signal1_dict: Dict[float, float],
                               signal2_dict: Dict[float, float],
                               known_delay: float = 1.0) -> float:
    # Estimate delay in samples using correlation
    from .correlation import estimate_time_delay
    
    # First, try with sampling frequency = 1 to get lag
    lag_samples, _, _ = estimate_time_delay(signal1_dict, signal2_dict, sampling_freq=1.0)
    
    if lag_samples == 0:
        return 0.0
    
    # Estimate sampling frequency: Fs = lag_samples / known_delay
    estimated_fs = abs(lag_samples) / known_delay
    
    return estimated_fs


def analyze_signal_statistics(signal_dict: Dict[float, float]) -> Dict[str, float]:
    samples = np.array(list(signal_dict.values()))
    
    if len(samples) == 0:
        return {}
    
    stats = {
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
        'variance': float(np.var(samples)),
        'min': float(np.min(samples)),
        'max': float(np.max(samples)),
        'range': float(np.max(samples) - np.min(samples)),
        'energy': float(np.sum(samples ** 2)),
        'power': float(np.mean(samples ** 2)),
        'rms': float(np.sqrt(np.mean(samples ** 2)))
    }
    
    return stats


def detect_peaks(signal_dict: Dict[float, float], 
                threshold: float = 0.5) -> List[Tuple[int, float]]:
    indices = list(signal_dict.keys())
    samples = list(signal_dict.values())
    
    if len(samples) < 3:
        return []
    
    peaks = []
    max_val = max(samples)
    min_height = threshold * max_val
    
    for i in range(1, len(samples) - 1):
        if samples[i] > samples[i-1] and samples[i] > samples[i+1]:
            if samples[i] >= min_height:
                peaks.append((int(indices[i]), float(samples[i])))
    
    return peaks