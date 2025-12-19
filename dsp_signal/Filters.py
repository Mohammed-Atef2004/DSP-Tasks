import numpy as np


def window_function(attenuation, n, N):

    if attenuation <= 21:
        return 1  # Rectangular
    elif attenuation <= 44:
        return 0.5 + 0.5 * np.cos(2 * np.pi * n / N)  # Hanning
    elif attenuation <= 53:
        return 0.54 + 0.46 * np.cos(2 * np.pi * n / N)  # Hamming
    elif attenuation <= 74:
        return 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))  # Blackman
    else:
        return 1  # fallback


def design_fir_filter(filter_type, fs, fc=None, f1=None, f2=None, attenuation=44, transband=0.05):


    # Compute normalized transition width
    deltaF = transband / fs

    # Compute filter length N based on attenuation
    if attenuation <= 21:
        N = int(np.ceil(0.9 / deltaF))
    elif attenuation <= 44:
        N = int(np.ceil(3.1 / deltaF))
    elif attenuation <= 53:
        N = int(np.ceil(3.3 / deltaF))
    elif attenuation <= 74:
        N = int(np.ceil(5.5 / deltaF))
    else:
        N = int(np.ceil(3.1 / deltaF))

    if N % 2 == 0:
        N += 1

    # Symmetric indices
    half = (N - 1) // 2
    indices = list(range(-half, half + 1))


    h = {}
    for n in indices:
        w = window_function(attenuation, n, N)

        if filter_type.lower() == 'low':
            f_c = (fc + transband / 2) / fs  # normalized with half transition
            if n == 0:
                h_d = 2 * f_c
            else:
                h_d = 2 * f_c * np.sin(2 * np.pi * f_c * n) / (2 * np.pi * f_c * n)

        elif filter_type.lower() == 'high':
            f_c = (fc - transband / 2) / fs
            if n == 0:
                h_d = 1 - 2 * f_c
            else:
                h_d = - 2 * f_c * np.sin(2 * np.pi * f_c * n) / (2 * np.pi * f_c * n)

        elif filter_type.lower() == 'bandpass':
            f1_norm = (f1 - transband / 2) / fs
            f2_norm = (f2 + transband / 2) / fs
            if n == 0:
                h_d = 2 * (f2_norm - f1_norm)
            else:
                h_d = (2 * f2_norm * np.sin(2 * np.pi * f2_norm * n) / (2 * np.pi * f2_norm * n)) - \
                      (2 * f1_norm * np.sin(2 * np.pi * f1_norm * n) / (2 * np.pi * f1_norm * n))

        elif filter_type.lower() == 'bandstop':
            f1_norm = (f1 + transband / 2) / fs
            f2_norm = (f2 - transband / 2) / fs
            if n == 0:
                h_d = 1 - 2 * (f2_norm - f1_norm)
            else:
                h_d = (2 * f1_norm * np.sin(2 * np.pi * f1_norm * n) / (2 * np.pi * f1_norm * n)) - \
                      (2 * f2_norm * np.sin(2 * np.pi * f2_norm * n) / (2 * np.pi * f2_norm * n))

        else:
            raise ValueError("Unknown filter type")

        h[n] = h_d * w

    return h
