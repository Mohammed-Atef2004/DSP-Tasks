from typing import Dict


def derivative_signal(signal: Dict[float, float], derivative_level: int):
    if derivative_level == 1:
        signal_values = list(signal.values())
        y_n = {}
        for n in range(0, len(signal) - 1):
            y_n[n] = signal_values[n + 1] - signal_values[n]
        return y_n
    elif derivative_level == 2:
        first_derivative = derivative_signal(signal, 1)
        second_derivative = derivative_signal(first_derivative, 1)
        return second_derivative
    else:
        raise ValueError("Only first and second derivatives are supported.")


def convolve_signals(signal: Dict[float, float], h: Dict[float, float]):
    h_keys = list(h.keys())
    h_values = list(h.values())
    signal_keys = list(signal.keys())

    y_start = h_keys[0] + signal_keys[0]
    y_end = h_keys[-1] + signal_keys[-1]
    y = {}

    for n in range(int(y_start), int(y_end) + 1):
        y_n = 0
        for k in range(len(h)):
            signal_index = n - h_keys[k]
            if signal_index in signal_keys:
                signal_value = signal[signal_index]
            else:
                signal_value = 0
            y_n += h_values[k] * signal_value
        y[n] = y_n
    return y


def moving_average_signal(signal: Dict[float, float], window_size: int):
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0 for moving average.")

    signal_values = list(signal.values())
    y_n = {}

    for n in range(0, len(signal) - window_size + 1):
        window = signal_values[n:n + window_size]
        y_n[n] = sum(window) / window_size
    return y_n