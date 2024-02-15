import argparse
import os
import numpy as np
import time
from rtlsdr import RtlSdr

def capture_data(start_freq, end_freq, sample_rate, duration_seconds, output_dir):
    # Configure RTL-SDR device
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate

    # Data capture loop
    data = []
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_filename = os.path.join(output_dir, f'binary_raw_{timestamp}.bin')

    start_time = time.time()
    while (time.time() - start_time) < duration_seconds:
        for freq in np.arange(start_freq, end_freq, 1e6):  # Step of 1 MHz
            sdr.center_freq = freq
            samples = sdr.read_samples(1024 * 256)  # Read 256 KB of samples
            data.append(samples)

    # Close RTL-SDR device
    sdr.close()

    # Save data to binary file
    with open(output_filename, 'wb') as f:
        for samples in data:
            f.write(samples.tobytes())

    print(f"Data saved to: {output_filename}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RTL-SDR Data Capture')
    parser.add_argument('--start-freq', type=float, help='Start frequency in Hz', required=True)
    parser.add_argument('--end-freq', type=float, help='End frequency in Hz', required=True)
    parser.add_argument('--sample-rate', type=float, help='Sample rate in Hz', default=2.4e6)
    parser.add_argument('--duration', type=int, help='Duration of capture in seconds', default=60)
    parser.add_argument('--output-dir', type=str, help='Directory to save the output file', default='./')
    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Capture data
    capture_data(args.start_freq, args.end_freq, args.sample_rate, args.duration, args.output_dir)
