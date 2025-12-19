def ReadSignalFile(file_name):
    """Read signal from file (compatible with original format)"""
    try:
        with open(file_name, 'r') as f:
            f.readline()  # skip first two lines
            f.readline()
            num_samples_line = f.readline().strip()
            if not num_samples_line:
                raise ValueError("Invalid file format")
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
        raise ValueError(f"Error reading file: {str(e)}")