from dsp_signal.file_io import ReadSignalFile


def compare_signals(Your_indices, Your_samples, file_name):
    """Compare generated signal with expected signal from file"""
    expected_indices, expected_samples = ReadSignalFile(file_name)

    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return False

    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            print("Test case failed, your signal have different indices from the expected one")
            return False

    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return False

    print("Comparison test case passed successfully")
    return True

# test cases: 1, 3
def Compare_filters(file_name,Your_indices,Your_samples):      
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return
    print("Test case passed successfully")


import math

def SignalComapreAmplitude(SignalInput, SignalOutput):
    if len(SignalInput) != len(SignalOutput):
        return False
    for i in range(len(SignalInput)):
        # Tolerance of 0.001 as per faculty requirement
        if abs(SignalInput[i] - SignalOutput[i]) > 0.001:
            return False
    return True

def SignalComaprePhaseShift(SignalInput, SignalOutput):
    if len(SignalInput) != len(SignalOutput):
        return False
    for i in range(len(SignalInput)):
        # Rounding and tolerance of 0.0001 as per faculty requirement
        A = round(SignalInput[i], 4)
        B = round(SignalOutput[i], 4)
        if abs(A - B) > 0.0001:
            return False
    return True