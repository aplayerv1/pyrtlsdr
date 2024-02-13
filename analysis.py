import numpy as np
import argparse
import os
import re
import time as tm
from scipy.signal import find_peaks

def extract_date_from_filename(filename):
    # Extract date and time from filename using regular expression
    pattern = r'(\d{8})_(\d{6})'  # Assuming the format is YYYYMMDD_HHMMSS
    match = re.search(pattern, filename)
    if match:
        date = match.group(1)
        time = match.group(2)
        return date, time
    else:
        return None, None

def detect_signals(data, threshold):
    # Find peaks in the data above the specified threshold
    peaks, _ = find_peaks(data, height=threshold)

    # Return the indices of detected peaks
    return peaks

def main(args):
    # Extract date from the input filename
    filename = os.path.basename(args.input)
    date, time = extract_date_from_filename(filename)
    start_time = tm.time()

    if date:
        # Read the RTL-SDR binary file
        with open(args.input, 'rb') as f:
            # Read the binary data and store it in a numpy array
            binary_data = np.fromfile(f, dtype=np.uint8)
            print("Binary data shape:", binary_data.shape)

        # Perform signal detection
        detected_peaks = detect_signals(binary_data, args.threshold)
        print("Detected peaks:", detected_peaks)

        end_time = tm.time()
        total_time = end_time - start_time
        print(f"Total time taken: {total_time} seconds")
    else:
        print("Unable to extract date from the filename.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTL-SDR binary data and detect signals.')
    parser.add_argument('-i', '--input', type=str, help='Path to RTL-SDR binary file', required=True)
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for signal detection')
    args = parser.parse_args()

    main(args)
