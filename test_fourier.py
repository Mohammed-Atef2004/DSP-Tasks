#!/usr/bin/env python3
"""
Simple direct test of faculty Fourier Transform files
"""

import sys
import os
import math
import numpy as np

print("Faculty Fourier Transform Test")
print("=" * 60)

# Read the input signal
print("\n1. Reading input signal...")
with open("input_Signal_DFT.txt", 'r') as f:
    lines = f.readlines()
    # Skip first 3 lines: 0, 0, N
    N = int(lines[2].strip())
    print(f"   Number of samples: N = {N}")
    
    signal = []
    for i in range(3, 3 + N):
        parts = lines[i].strip().split()
        if len(parts) >= 2:
            signal.append(float(parts[1]))
    
    print(f"   Signal: {signal}")

# Read expected DFT output
print("\n2. Reading expected DFT output...")
with open("Output_Signal_DFT_A,Phase.txt", 'r') as f:
    lines = f.readlines()
    # Skip first 3 lines
    expected_magnitudes = []
    expected_phases = []
    
    for i in range(3, len(lines)):
        line = lines[i].strip()
        if line:
            # Remove 'f' suffix if present
            line = line.replace('f', '')
            parts = line.split()
            if len(parts) >= 2:
                expected_magnitudes.append(float(parts[0]))
                expected_phases.append(float(parts[1]))
    
    print(f"   Expected {len(expected_magnitudes)} frequency points")

# Compute DFT manually
print("\n3. Computing DFT manually...")
N = len(signal)
computed_magnitudes = np.zeros(N)
computed_phases = np.zeros(N)

for k in range(N):
    real_sum = 0
    imag_sum = 0
    for n in range(N):
        angle = -2 * math.pi * k * n / N
        real_sum += signal[n] * math.cos(angle)
        imag_sum += signal[n] * math.sin(angle)
    
    computed_magnitudes[k] = math.sqrt(real_sum**2 + imag_sum**2)
    computed_phases[k] = math.atan2(imag_sum, real_sum)

print(f"   Computed DFT for {N} frequency points")

# Compare with expected
print("\n4. Comparing with expected results...")
print("-" * 60)
print("Index | Expected Mag | Computed Mag | Diff | Expected Phase | Computed Phase | Diff")
print("-" * 100)

all_match = True
for i in range(min(len(expected_magnitudes), len(computed_magnitudes))):
    mag_diff = abs(expected_magnitudes[i] - computed_magnitudes[i])
    phase_diff = abs(expected_phases[i] - computed_phases[i])
    
    mag_ok = mag_diff < 0.001
    phase_ok = phase_diff < 0.001
    
    if not (mag_ok and phase_ok):
        all_match = False
    
    print(f"{i:5d} | {expected_magnitudes[i]:12.6f} | {computed_magnitudes[i]:12.6f} | {mag_diff:8.6f} {'✓' if mag_ok else '✗'} | "
          f"{expected_phases[i]:14.6f} | {computed_phases[i]:14.6f} | {phase_diff:8.6f} {'✓' if phase_ok else '✗'}")

# Test IDFT
print("\n5. Testing IDFT...")
print("-" * 60)

# Use the expected magnitudes and phases to reconstruct
reconstructed = np.zeros(N)
for n in range(N):
    sum_val = 0.0
    for k in range(N):
        real = expected_magnitudes[k] * math.cos(expected_phases[k])
        imag = expected_magnitudes[k] * math.sin(expected_phases[k])
        
        angle = 2 * math.pi * k * n / N
        sum_val += real * math.cos(angle) - imag * math.sin(angle)
    
    reconstructed[n] = sum_val / N

print("Index | Original | Reconstructed | Diff")
print("-" * 60)

reconstruction_ok = True
for i in range(N):
    diff = abs(signal[i] - reconstructed[i])
    ok = diff < 0.001
    if not ok:
        reconstruction_ok = False
    print(f"{i:5d} | {signal[i]:9.6f} | {reconstructed[i]:13.6f} | {diff:8.6f} {'✓' if ok else '✗'}")

# Final results
print("\n" + "=" * 60)
print("FINAL RESULTS:")
print("=" * 60)

if all_match:
    print("DFT Test: PASSED - Your DFT matches expected results")
else:
    print("DFT Test: FAILED - Check DFT implementation")

if reconstruction_ok:
    print("IDFT Test: PASSED - Signal reconstructed correctly")
else:
    print("IDFT Test: FAILED - Check IDFT implementation")

if all_match and reconstruction_ok:
    print("\nALL TESTS PASSED! Your implementation is correct.")
    sys.exit(0)
else:
    print("\nSome tests failed. Check implementation.")
    sys.exit(1)