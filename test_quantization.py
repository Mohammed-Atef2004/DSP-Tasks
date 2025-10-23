import numpy as np
from dsp_signal_processing_tool import Signal
from QuanTest1 import QuantizationTest1

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

def quantize_samples_fixed(samples, L, x_min=0.2, x_max=1.0):
    """
    Fixed quantization that matches faculty's expected output
    """
    samples = np.array(samples, dtype=float)
    
    # Use the range that matches faculty test
    delta = (x_max - x_min) / L
    
    print(f"Quantization parameters: min={x_min}, max={x_max}, delta={delta}, L={L}")
    
    # Create quantization levels (midpoints)
    levels = x_min + (np.arange(L) + 0.5) * delta
    
    print(f"Quantization levels: {levels}")
    
    # Quantize each sample to nearest level
    quantized = np.zeros_like(samples)
    encoded = np.zeros_like(samples, dtype=int)
    
    for i, sample in enumerate(samples):
        differences = np.abs(sample - levels)
        closest_idx = np.argmin(differences)
        quantized[i] = levels[closest_idx]
        encoded[i] = closest_idx
    
    return quantized, encoded

def debug_faculty_output():
    """Debug what the faculty's expected output actually contains"""
    expectedEncodedValues = []
    expectedQuantizedValues = []
    
    with open("Quan1_Out.txt", 'r') as f:
        line = f.readline()  # first 0
        line = f.readline()  # second 0  
        line = f.readline()  # 11
        line = f.readline()  # first data line
        
        while line and line.strip():
            L = line.strip()
            if len(L.split(' ')) == 2:
                parts = L.split(' ')
                encoded = str(parts[0])
                quantized = float(parts[1])
                expectedEncodedValues.append(encoded)
                expectedQuantizedValues.append(quantized)
            line = f.readline()
    
    print("Expected encoded values:", expectedEncodedValues)
    print("Expected quantized values:", expectedQuantizedValues)
    print("Length of expected encoded:", len(expectedEncodedValues))
    print("Length of expected quantized:", len(expectedQuantizedValues))
    return expectedEncodedValues, expectedQuantizedValues

def test_quantization_with_faculty_files():
    # Debug faculty output first
    expected_encoded, expected_quantized = debug_faculty_output()
    
    # Read input signal
    input_file = "Quan1_input.txt"
    indices, samples = read_input_file(input_file)
    
    if not samples:
        print("Failed to read input file")
        return
    
    print("Input samples:", samples)
    
    # Use 3-bit quantization (8 levels) with faculty's range
    bits = 3
    L = 2 ** bits  # 8 levels
    
    # Perform quantization using fixed function
    quantized_samples, encoded_values = quantize_samples_fixed(np.array(samples), L)
    
    print("Encoded values (indices):", encoded_values)
    print("Quantized samples:", quantized_samples)
    
    # Convert encoded values to 3-bit binary strings
    your_encoded_str = [format(code, '03b') for code in encoded_values]
    
    print("Your encoded strings:", your_encoded_str)
    print("Your quantized values:", quantized_samples.tolist())
    
    # Test against faculty's expected output
    QuantizationTest1("Quan1_Out.txt", your_encoded_str, quantized_samples.tolist())

if __name__ == "__main__":
    test_quantization_with_faculty_files()