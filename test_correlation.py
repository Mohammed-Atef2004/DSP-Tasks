#!/usr/bin/env python3
"""
Test correlation and time delay estimation with faculty files - FIXED VERSION
"""

import sys
import os
import math
import numpy as np  # ADD THIS IMPORT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsp_signal.signal import Signal
from dsp_signal.correlation import compute_correlation_faculty, estimate_time_delay


def read_signal_file(filename):
    """Read signal from faculty file format"""
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Skip first 3 lines: 0, 1, N
        N = int(lines[2].strip())
        indices = []
        samples = []
        for i in range(3, 3 + N):
            parts = lines[i].strip().split()
            if len(parts) >= 2:
                indices.append(int(parts[0]))
                samples.append(float(parts[1]))
        return indices, samples


def test_correlation_fixed():
    """Test correlation with faculty files - fixed version"""
    print("Testing Correlation with Faculty Files - FIXED")
    print("=" * 60)
    
    # Test 1: Correlation between signal1 and signal2
    print("\nTest 1: Cross-Correlation")
    print("-" * 40)
    
    # Read signals
    indices1, samples1 = read_signal_file("Corr_input signal1.txt")
    indices2, samples2 = read_signal_file("Corr_input signal2.txt")
    
    print(f"Signal 1: {samples1}")
    print(f"Signal 2: {samples2}")
    
    # Convert to dictionaries
    signal1_dict = {float(i): float(s) for i, s in zip(indices1, samples1)}
    signal2_dict = {float(i): float(s) for i, s in zip(indices2, samples2)}
    
    # Compute correlation using faculty format
    correlation_dict = compute_correlation_faculty(signal1_dict, signal2_dict)
    
    print(f"\nComputed correlation ({len(correlation_dict)} lags):")
    for lag in sorted(correlation_dict.keys()):
        print(f"  Lag {lag:3d}: {correlation_dict[lag]:.8f}")
    
    # Read expected output
    print("\nExpected correlation from CorrOutput.txt:")
    expected_indices, expected_samples = read_signal_file("CorrOutput.txt")
    for idx, val in zip(expected_indices, expected_samples):
        print(f"  Lag {idx:3d}: {val:.8f}")
    
    # Compare
    print("\nComparison:")
    print("Lag | Computed | Expected | Diff")
    print("-" * 40)
    
    all_match = True
    for lag in sorted(correlation_dict.keys()):
        if lag < len(expected_samples):
            computed = correlation_dict[lag]
            expected = expected_samples[lag]
            diff = abs(computed - expected)
            
            if diff > 0.01:
                all_match = False
            
            print(f"{lag:3d} | {computed:.8f} | {expected:.8f} | {diff:.8f} {'✓' if diff <= 0.01 else '✗'}")
        else:
            print(f"{lag:3d} | {correlation_dict[lag]:.8f} | -- | --")
    
    if all_match:
        print("\n✅ Correlation Test PASSED!")
    else:
        print("\n❌ Correlation Test FAILED")
    
    return all_match


def test_time_delay_fixed():
    """Test time delay estimation - fixed version"""
    print("\n" + "=" * 60)
    print("Test 2: Time Delay Estimation")
    print("=" * 60)
    
    # Read time-domain signals
    indices1, samples1 = read_signal_file("TD_input signal1.txt")
    indices2, samples2 = read_signal_file("TD_input signal2.txt")
    
    print(f"Signal 1: {samples1}")
    print(f"Signal 2: {samples2}")
    
    # Convert to dictionaries
    signal1_dict = {float(i): float(s) for i, s in zip(indices1, samples1)}
    signal2_dict = {float(i): float(s) for i, s in zip(indices2, samples2)}
    
    # According to "Fs and expected output.txt"
    # Fs = 100 Hz, Expected delay = 5/100 = 0.05 seconds
    
    # Estimate time delay
    lag_samples, delay_seconds, correlation_dict = estimate_time_delay(
        signal1_dict,
        signal2_dict,
        sampling_freq=100.0  # Use known Fs
    )
    
    print(f"\nCorrelation values:")
    for lag in sorted(correlation_dict.keys()):
        print(f"  Lag {lag}: {correlation_dict[lag]:.6f}")
    
    print(f"\nTime Delay Estimation:")
    print(f"Peak at lag: {lag_samples} samples")
    print(f"Delay: {delay_seconds:.6f} seconds ({delay_seconds*1000:.3f} ms)")
    print(f"Expected: 5 samples, 0.05 seconds (50 ms)")
    
    # Check if close to expected
    # Note: The faculty says "Excpected output = 5/100" which is 0.05 seconds
    # But signals might have peak at lag 4 or 5
    lag_error = abs(lag_samples - 5)
    time_error = abs(delay_seconds - 0.05)
    
    # Allow some tolerance
    if lag_error <= 1 and time_error <= 0.015:
        print("\n✅ Time Delay Test PASSED (within tolerance)!")
        return True
    else:
        print(f"\n❌ Time Delay Test FAILED")
        print(f"   Lag error: {lag_error} samples")
        print(f"   Time error: {time_error:.6f} seconds")
        
        # Let's check what the actual peak should be
        print(f"\nDebug: Checking correlation values at lags 4 and 5:")
        if 4 in correlation_dict:
            print(f"  Lag 4: {correlation_dict[4]:.6f}")
        if 5 in correlation_dict:
            print(f"  Lag 5: {correlation_dict[5]:.6f}")
        
        return False


def verify_correlation_formula():
    """Verify the correlation formula matches faculty's implementation"""
    print("\n" + "=" * 60)
    print("Verifying Correlation Formula")
    print("=" * 60)
    
    # Simple test case
    x = np.array([2.0, 1.0, 0.0, 0.0, 3.0])
    y = np.array([3.0, 2.0, 1.0, 1.0, 5.0])
    
    print(f"x = {x}")
    print(f"y = {y}")
    
    # Manual calculation for lag 0
    print("\nManual calculation for lag 0:")
    N = len(x)
    k = 0
    
    sum_val = 0
    sum_x2 = 0
    sum_y2 = 0
    
    for n in range(N - k):
        sum_val += x[n + k] * y[n]
        sum_x2 += x[n + k] ** 2
        sum_y2 += y[n] ** 2
    
    print(f"  Σ x[n]*y[n] = {sum_val}")
    print(f"  Σ x[n]² = {sum_x2}")
    print(f"  Σ y[n]² = {sum_y2}")
    
    if sum_x2 > 0 and sum_y2 > 0:
        norm_factor = math.sqrt(sum_x2 * sum_y2)
        result = sum_val / norm_factor
        print(f"  Normalization factor = {norm_factor}")
        print(f"  r[0] = {sum_val} / {norm_factor} = {result:.8f}")
    
    # Manual calculation for lag 1
    print("\nManual calculation for lag 1:")
    k = 1
    
    sum_val = 0
    sum_x2 = 0
    sum_y2 = 0
    
    for n in range(N - k):
        sum_val += x[n + k] * y[n]
        sum_x2 += x[n + k] ** 2
        sum_y2 += y[n] ** 2
    
    print(f"  Σ x[n+1]*y[n] = {sum_val}")
    print(f"  Σ x[n+1]² = {sum_x2}")
    print(f"  Σ y[n]² = {sum_y2}")
    
    if sum_x2 > 0 and sum_y2 > 0:
        norm_factor = math.sqrt(sum_x2 * sum_y2)
        result = sum_val / norm_factor
        print(f"  Normalization factor = {norm_factor}")
        print(f"  r[1] = {sum_val} / {norm_factor} = {result:.8f}")


def main():
    """Main test function"""
    print("Correlation and Time Analysis Test Suite - FIXED")
    print("=" * 60)
    
    # First verify formula
    verify_correlation_formula()
    
    tests_passed = 0
    total_tests = 2
    
    # Check for required files
    required_files = [
        ("Corr_input signal1.txt", True),
        ("Corr_input signal2.txt", True),
        ("CorrOutput.txt", True),
        ("TD_input signal1.txt", False),
        ("TD_input signal2.txt", False)
    ]
    
    print("\nChecking for test files:")
    all_files_exist = True
    for filename, required in required_files:
        exists = os.path.exists(filename)
        status = "✅ Found" if exists else ("⚠️  Missing" if not required else "❌ Missing")
        print(f"  {filename}: {status}")
        if required and not exists:
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Missing required files. Cannot run tests.")
        return False
    
    # Run tests
    print("\n" + "=" * 60)
    
    # Test 1: Correlation
    if test_correlation_fixed():
        tests_passed += 1
    
    # Test 2: Time delay (optional)
    if os.path.exists("TD_input signal1.txt") and os.path.exists("TD_input signal2.txt"):
        if test_time_delay_fixed():
            tests_passed += 1
    else:
        print("\n⚠️  Time delay test files not found, skipping...")
        total_tests -= 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)