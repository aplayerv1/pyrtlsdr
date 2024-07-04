import os
import re
import argparse
import numpy as np
import time as tm
from astropy.io import fits
from scipy import signal
from tqdm import tqdm
import gc
from datetime import datetime
import concurrent.futures

gc.enable()

# Constants
DEFAULT_SAMPLING_RATE = 2.4e6
DEFAULT_CENTER_FREQUENCY = 1420.40e6
DEFAULT_LNB_OFFSET = 9750e6
DEFAULT_GAIN_FACTOR = 50
DEFAULT_BANDWIDTH = 100e6

def extract_date_from_filename(filename):
    # Extract date and time from filename using regular expression
    match = re.search(r'(\d{8})_(\d{6})', filename)  # Assuming the format is YYYYMMDD_HHMMSS
    return match.groups() if match else (None, None)

def extract_observation_start_time(fits_filename):
    with fits.open(fits_filename, ignore_missing_simple=True) as hdul:
        header = hdul[0].header
        # Extract observation start time from the header (adjust the keyword as per your FITS file)
        start_date_str = header['DATE']
        # Append default time (midnight) to the date string
        start_time_str = start_date_str + 'T00:00:00'
        # Convert the string to a datetime object
        start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%S')
    return start_time

def bandpass_filter(signal_data, sampling_rate, center_frequency, bandwidth, filter_order):
    nyquist_rate = sampling_rate / 2
    low_cutoff = (center_frequency - bandwidth / 2) / nyquist_rate
    high_cutoff = (center_frequency + bandwidth / 2) / nyquist_rate
    sos = signal.butter(filter_order, [low_cutoff, high_cutoff], btype='bandpass', output='sos', fs=sampling_rate)
    return signal.sosfiltfilt(sos, signal_data)

def compute_fft_chunk(signal_chunk, sampling_rate, chunk_size):
    frequencies = np.fft.fftfreq(chunk_size, 1 / sampling_rate)
    return np.fft.fft(signal_chunk), frequencies

def compute_fft(signal_data, sampling_rate, chunk_size=1024):
    num_samples = len(signal_data)
    num_chunks = num_samples // chunk_size
    
    # Create a ThreadPoolExecutor with a reasonable number of threads
    num_threads = min(num_chunks, os.cpu_count())  # Use available CPU cores
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(compute_fft_chunk, (signal_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)),
                                        [sampling_rate] * num_chunks,
                                        [chunk_size] * num_chunks),
                            total=num_chunks,
                            desc='Computing FFT',
                            unit='chunk'))
    
    # Extract frequencies and fft_results from results
    frequencies_all, fft_result_all = [], []
    for fft_result, frequencies in results:
        frequencies_all.append(frequencies)
        fft_result_all.append(fft_result)
    
    return frequencies_all, fft_result_all

def save_fft_to_file_in_chunks(frequencies, fft_result, filename):
    with open(filename + '.txt', 'w') as file:
        file.write("Frequency(Hz), Real Part, Imaginary Part\n")
        for freq_chunk, result_chunk in zip(frequencies, fft_result):
            for freq, value in zip(freq_chunk, result_chunk):
                file.write(f"{freq}, {value.real}, {value.imag}\n")
    print(f"FFT result saved to {filename}.txt")

def remove_lnb_offset(signal_data, sampling_frequency, lnb_offset_frequency):
    nyquist = sampling_frequency / 2
    lnb_normalized_frequency = np.clip(lnb_offset_frequency / nyquist, 0.01, 0.99)
    b, a = signal.butter(5, lnb_normalized_frequency, btype='high')
    return signal.filtfilt(b, a, signal_data)

def remove_dc_offset(signal_data):
    return signal_data - np.mean(signal_data)

def main(args):
    filename = os.path.basename(args.input)
    date, time = extract_date_from_filename(filename)
    start_time = extract_observation_start_time(args.input)
    if date and start_time:
        os.makedirs(args.output, exist_ok=True)
        hdul = fits.open(args.input, ignore_missing_simple=True)
        data = hdul[0].data
        hdul.close()
        
        # Remove DC offset from the data
        data = remove_dc_offset(data)
        
        # Remove LNB offset and apply bandpass filter
        data = remove_lnb_offset(data, args.sampling_rate, args.lnb_offset)
        data = bandpass_filter(data, args.sampling_rate, args.center_frequency, args.bandwidth, 5)
        
        # Compute FFT using multithreading
        frequencies, fft_result = compute_fft(data, args.sampling_rate)
        
        # Save FFT results to file
        save_fft_to_file_in_chunks(frequencies, fft_result, os.path.join(args.output, filename))
    else:
        print("Unable to extract date from the filename.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTL-SDR binary data and generate heatmap and signal strength plots.')
    parser.add_argument('-i', '--input', type=str, help='Path to RTL-SDR binary file')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory for FFT files (default: output)')
    parser.add_argument('-s', '--sampling_rate', type=float, default=DEFAULT_SAMPLING_RATE, help='Sampling rate in Hz (default: 2.4e6)')
    parser.add_argument('-c', '--center_frequency', type=float, default=DEFAULT_CENTER_FREQUENCY, help='Center frequency in Hz (default: 1420.40e6)')
    parser.add_argument('-l', '--lnb_offset', type=float, default=DEFAULT_LNB_OFFSET, help='LNB offset frequency in Hz (default: 9750e6)')
    parser.add_argument('-g', '--gain-factor', type=float, default=DEFAULT_GAIN_FACTOR, help='Digital gain factor (default: 50)')
    parser.add_argument('-b', '--bandwidth', type=float, default=DEFAULT_BANDWIDTH, help='Bandwidth in Hz (default: 100e6)')
    args = parser.parse_args()

    main(args)

gc.collect()
