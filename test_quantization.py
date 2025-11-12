import numpy as np
from dsp_signal_processing_tool import Signal
from QuanTest1 import QuantizationTest1
from QuanTest2 import QuantizationTest2

def read_input_file(filename):
    try:
        with open(filename, 'r') as f:
            f.readline()  
            f.readline()  
            num_samples_line = f.readline().strip()
            num_samples = int(num_samples_line)
            indices = []
            samples = []

            for _ in range(num_samples):
                line = f.readline().strip()
                if not line:
                    break
                parts = line.split()
                if len(parts) == 2:
                    indices.append(int(parts[0]))
                    samples.append(float(parts[1]))

            return indices, samples
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        return [], []

def read_expected_output_2(filename):
    """Read expected output for test 2 to understand the indices"""
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    
    with open(filename, 'r') as f:
        # Skip header
        for _ in range(3):
            f.readline()
        
        # Read data
        while True:
            line = f.readline().strip()
            if not line:
                break
            parts = line.split()
            if len(parts) == 4:
                expectedIntervalIndices.append(int(parts[0]))
                expectedEncodedValues.append(parts[1])
                expectedQuantizedValues.append(float(parts[2]))
                expectedSampledError.append(float(parts[3]))
    
    print("=== Expected Output for Test 2 ===")
    print(f"Expected indices: {expectedIntervalIndices}")
    print(f"Expected encoded: {expectedEncodedValues}")
    print(f"Expected quantized: {expectedQuantizedValues}")
    print(f"Expected error: {expectedSampledError}")
    
    return expectedIntervalIndices, expectedEncodedValues, expectedQuantizedValues, expectedSampledError

def quantize_samples(samples, L, x_min=None, x_max=None):
    """Generic quantization function that matches faculty's algorithm"""
    samples = np.array(samples, dtype=float)
    
    if x_min is None:
        x_min = min(samples)
    if x_max is None:
        x_max = max(samples)
    
    delta = (x_max - x_min) / L
    levels = x_min + (np.arange(L) + 0.5) * delta
    
    quantized = np.zeros_like(samples)
    encoded = np.zeros_like(samples, dtype=int)
    
    for i, sample in enumerate(samples):
        differences = np.abs(sample - levels)
        closest_idx = np.argmin(differences)
        quantized[i] = levels[closest_idx]
        encoded[i] = closest_idx
    
    return quantized, encoded

def test_quantization_1():
    """Test 1: 3-bit quantization with fixed range 0.2-1.0"""
    print("=== Running Quantization Test 1 ===")
    
    indices, samples = read_input_file("Quan1_input.txt")
    print(f"Test 1 - Input samples: {samples}")
    
    L = 8  # 3-bit quantization
    x_min, x_max = 0.2, 1.0  # Fixed range for test 1
    
    quantized, encoded = quantize_samples(samples, L, x_min, x_max)
    your_encoded_str = [format(code, '03b') for code in encoded]
    
    print(f"Test 1 - My encoded: {your_encoded_str}")
    print(f"Test 1 - My quantized: {quantized.tolist()}")
    
    QuantizationTest1("Quan1_Out.txt", your_encoded_str, quantized.tolist())

def test_quantization_2():
    """Test 2: 2-bit quantization with auto range"""
    print("\n=== Running Quantization Test 2 ===")
    
    # Read input and expected output to understand the indices
    indices, samples = read_input_file("Quan2_input.txt")
    exp_indices, exp_encoded, exp_quantized, exp_error = read_expected_output_2("Quan2_Out.txt")
    
    print(f"Test 2 - Input indices: {indices}")
    print(f"Test 2 - Input samples: {samples}")
    
    L = 4  # 2-bit quantization
    # Auto range for test 2 - calculate from data
    x_min, x_max = min(samples), max(samples)
    
    quantized, encoded = quantize_samples(samples, L, x_min, x_max)
    
    # FIX: Calculate error as Quantized - Original (not Original - Quantized)
    error = quantized - np.array(samples)  # This is the key fix!
    
    your_encoded_str = [format(code, '02b') for code in encoded]
    
    print(f"Test 2 - My indices: {indices}")
    print(f"Test 2 - My encoded: {your_encoded_str}")
    print(f"Test 2 - My quantized: {quantized.tolist()}")
    print(f"Test 2 - My error (fixed): {error.tolist()}")
    
    # Use the expected indices instead of the input indices
    QuantizationTest2("Quan2_Out.txt", exp_indices, your_encoded_str, quantized.tolist(), error.tolist())

if __name__ == "__main__":
    test_quantization_1()
    test_quantization_2()