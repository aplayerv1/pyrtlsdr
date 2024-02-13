import argparse
import os
import numpy as np
import time
import logging
from rtlsdr import RtlSdr

# Set up logging
logging.basicConfig(filename='signal_processing.log', level=logging.ERROR)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Astronomical Signal Processing')
parser.add_argument('-f', '--frequency', type=float, help='Frequency of observation', required=True)
parser.add_argument('-t', '--time', type=int, help='Duration of signal processing in seconds', required=True)
parser.add_argument('--output-dir', type=str, help='Directory to save the output file', default='./')
parser.add_argument('--sample-rate', type=float, help='Sample rate in Hz', default=2.4e6)
parser.add_argument('--gain', type=str, help='Gain setting of the device', default='auto')
args = parser.parse_args()

# Set the frequency and duration of signal processing
frequency = args.frequency
duration_seconds = args.time
output_dir = args.output_dir
sample_rate = args.sample_rate
gain = args.gain

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Generate a filename based on the current timestamp
timestamp = time.strftime('%Y%m%d_%H%M%S')
output_filename = os.path.join(output_dir, f'raw_data_{timestamp}.bin')

# Configure RTL-SDR device
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = frequency

# Convert gain to integer or float
gain = int(gain)  # or float(gain) depending on your requirements

# Set the gain on the RTL-SDR device
sdr.gain = gain

# Define PPL lock parameters
ppl_lock_frequency = 1e3  # Frequency of the PPL lock loop
ppl_lock_time = 0.1  # Time to wait for PPL lock in seconds

# Main data acquisition loop
start_time = time.time()
# Assuming sdr is your RtlSdr object
tuned_frequency = sdr.center_freq
print("Tuned Frequency:", tuned_frequency)


try:
    with open(output_filename, 'wb') as f:
        # Wait for PPL lock
        time.sleep(ppl_lock_time)
        
        while (time.time() - start_time) < duration_seconds:
            # Read samples from the RTL-SDR device
            samples = sdr.read_samples(1024 * 16)  # Read 16 KB of samples
            # Convert samples to raw binary and write to file
            raw_bytes = np.array(samples, dtype=np.complex64).tobytes()
            f.write(raw_bytes)
finally:
    # Close RTL-SDR device
    sdr.close()
