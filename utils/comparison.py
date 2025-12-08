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