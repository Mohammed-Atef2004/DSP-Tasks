def QuantizationTest1(file_name, Your_EncodedValues, Your_QuantizedValues):
    expectedEncodedValues = []
    expectedQuantizedValues = []
    
    print(f"Testing with file: {file_name}")
    print(f"My encoded values length: {len(Your_EncodedValues)}")
    print(f"My quantized values length: {len(Your_QuantizedValues)}")
    
    with open(file_name, 'r') as f:
        # Skip the first three lines (header)
        for _ in range(3):
            line = f.readline()
        
        # Read the data lines
        while True:
            line = f.readline().strip()
            if not line:  # End of file
                break
                
            parts = line.split()
            if len(parts) == 2:
                encoded_val = str(parts[0])
                quantized_val = float(parts[1])
                expectedEncodedValues.append(encoded_val)
                expectedQuantizedValues.append(quantized_val)

    print(f"Expected encoded values length: {len(expectedEncodedValues)}")
    print(f"Expected quantized values length: {len(expectedQuantizedValues)}")
    
    # Debug: Print first few values
    print("\nFirst 3 values comparison:")
    for i in range(min(3, len(Your_EncodedValues))):
        print(f"Encoded: Me={Your_EncodedValues[i]}, Expected={expectedEncodedValues[i]}")
        print(f"Quantized: Me={Your_QuantizedValues[i]}, Expected={expectedQuantizedValues[i]}")
    
    # Check lengths
    if len(Your_EncodedValues) != len(expectedEncodedValues):
        print(f"QuantizationTest1 Test case failed, My encoded signal has different length ({len(Your_EncodedValues)}) from the expected one ({len(expectedEncodedValues)})")
        return
        
    if len(Your_QuantizedValues) != len(expectedQuantizedValues):
        print(f"QuantizationTest1 Test case failed, My quantized signal has different length ({len(Your_QuantizedValues)}) from the expected one ({len(expectedQuantizedValues)})")
        return
    
    # Check encoded values
    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            print(f"QuantizationTest1 Test case failed at index {i}: My EncodedValue '{Your_EncodedValues[i]}' != expected '{expectedEncodedValues[i]}'")
            return
    
    # Check quantized values with tolerance
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) >= 0.01:
            print(f"QuantizationTest1 Test case failed at index {i}: My QuantizedValue {Your_QuantizedValues[i]} != expected {expectedQuantizedValues[i]} (diff: {abs(Your_QuantizedValues[i] - expectedQuantizedValues[i])})")
            return
    
    print("QuantizationTest1 Test case passed successfully")