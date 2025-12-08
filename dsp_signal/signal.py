import numpy as np
from typing import List, Tuple, Dict


class Signal:
    def __init__(self, indices=None, samples=None, name=""):
        self.indices = indices if indices is not None else []
        self.samples = samples if samples is not None else []
        self.name = name

    def __str__(self):
        return f"Signal '{self.name}': {len(self.indices)} samples"

    def get_plot_data(self):
        """Return data suitable for matplotlib plotting"""
        return self.indices, self.samples

    def get_time_series(self, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return time vector and samples sampled at rate fs for continuous plotting convenience."""
        if not self.indices:
            return np.array([]), np.array([])
        n = np.array(self.indices)
        t = n / fs
        y = np.array(self.samples)
        return t, y

    def to_dict(self) -> Dict[float, float]:
        """Convert signal to dictionary format for the new functions"""
        return {float(idx): float(sample) for idx, sample in zip(self.indices, self.samples)}

    @classmethod
    def from_dict(cls, signal_dict: Dict[float, float], name: str = ""):
        """Create Signal object from dictionary"""
        indices = list(signal_dict.keys())
        samples = list(signal_dict.values())
        return cls(indices, samples, name)