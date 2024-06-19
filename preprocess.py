import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
from scipy import signal
from tqdm import tqdm
import pywt
import time as tm
from scipy.signal import butter, sosfiltfilt
from astropy.io import fits
import cv2
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta
from scipy import ndimage
from skimage import exposure
import gc

gc.enable()

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

def extract_observation_start_time(fits_filename):
    with fits.open(fits_filename, ignore_missing_simple=True) as hdul:
        header = hdul[0].header
        # Extract observation start time from the header (adjust the keyword as per your FITS file)
        start_time_str = header['DATE']
        # Convert the string to a datetime object
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d')
    return start_time



def bandpass_filter(signal_data, sampling_rate, center_frequency, bandwidth, filter_order):
    preprocessed_signal = signal_data
    # Define the Nyquist frequency
    nyquist_rate = sampling_rate / 2
    # Define the normalized cutoff frequencies for the bandpass filter
    low_cutoff = ((center_frequency - 10e6) - bandwidth / 2) / nyquist_rate  # Adjusted for LNB frequency offset
    high_cutoff = ((center_frequency - 10e6) + bandwidth / 2) / nyquist_rate  # Adjusted for LNB frequency offset

    # Normalize the cutoff frequencies
    low_cutoff_normalized = low_cutoff / nyquist_rate
    high_cutoff_normalized = high_cutoff / nyquist_rate

    # Apply bandpass filter
    b, a = signal.butter(filter_order, [low_cutoff_normalized, high_cutoff_normalized], 'bandpass')
    sos = butter(filter_order, [low_cutoff, high_cutoff], btype='bandpass', output='sos', fs=sampling_rate)
    signal_data_filtered = sosfiltfilt(sos, preprocessed_signal)
    return signal_data_filtered

def compute_fft(signal_data, sampling_rate, chunk_size=1024):
    num_samples = len(signal_data)
    num_chunks = num_samples // chunk_size

    frequencies_all = []
    fft_result_all = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        # Extract the chunk of signal data
        signal_chunk = signal_data[start_idx:end_idx]

        # Compute the FFT of the chunk
        frequencies_chunk, fft_result_chunk = np.fft.fftfreq(len(signal_chunk), 1 / sampling_rate), np.fft.fft(signal_chunk)

        # Append the FFT results to the lists
        frequencies_all.append(frequencies_chunk)
        fft_result_all.append(fft_result_chunk)

    return frequencies_all, fft_result_all

def save_fft_to_file_in_chunks(frequencies, fft_result, filename):
    fft_filename = filename + '.txt'

    with open(fft_filename, 'w') as file:
        file.write("Frequency(Hz), Real Part, Imaginary Part\n")

        for freq_chunk, result_chunk in zip(frequencies, fft_result):
            for freq, value in zip(freq_chunk, result_chunk):
                file.write(f"{freq}, {value.real}, {value.imag}\n")

    print(f"FFT result saved to {fft_filename}")


def remove_lnb_offset(signal_data, sampling_frequency, lnb_offset_frequency):
    nyquist = sampling_frequency / 2
    lnb_normalized_frequency = lnb_offset_frequency / nyquist
    if lnb_normalized_frequency >= 1:
        lnb_normalized_frequency = 0.99
    elif lnb_normalized_frequency <= 0:
        lnb_normalized_frequency = 0.01
    b, a = signal.butter(5, lnb_normalized_frequency, btype='high')
    filtered_data = signal.filtfilt(b, a, signal_data)
    return filtered_data

def main(args):
    # Extract date from the input filename
    filename = os.path.basename(args.input)
    date, time = extract_date_from_filename(filename)
    start_time_begin = tm.time()
    # Extract observation start time from FITS header
    start_time = extract_observation_start_time(args.input)

    if date:
        # Read the data from the FITS file
        hdul = fits.open(args.input, ignore_missing_simple=True)
        data = hdul[0].data
        hdul.close()
        
        # Create the output directory if it does not exist
        os.makedirs(args.output, exist_ok=True)
        
        # Preprocess the data
        binary_data_no_lnb = remove_lnb_offset(data,args.sampling_rate,args.lnb_offset)
        
        apply_bpf = bandpass_filter(binary_data_no_lnb,args.sampling_rate,args.center_frequency, args.bandwidth, 5)
        data_out = apply_bpf

        # Compute FFT of the signal data
        frequencies, fft_result = compute_fft(data_out, args.sampling_rate)
        file_n= os.path.join(args.output,filename + "_fft")
        # Save FFT result to file
        save_fft_to_file_in_chunks(frequencies, fft_result, file_n)


        output_filename = os.path.join(args.output, filename)
        with fits.open(output_filename, mode='append') as hdul_out:
            hdu_out = fits.ImageHDU(data_out)
            hdul_out.append(hdu_out)

        end_time = tm.time()
        total_time = (end_time - start_time_begin)/60
        print(f"Total time taken: {total_time} minutes")
        
    else:
        print("Unable to extract date from the filename.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTL-SDR binary data and generate heatmap and signal strength plots.')
    parser.add_argument('-i', '--input', type=str, help='Path to RTL-SDR binary file')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory for PNG files (default: output)')
    parser.add_argument('-s', '--sampling_rate', type=float, default=2.4e6, help='Sampling rate in Hz (default: 2.4e6)')
    parser.add_argument('-c', '--center_frequency', type=float, default=1420.40e6, help='Center frequency in Hz (default: 1420.30e6)')
    parser.add_argument('-l', '--lnb-offset', type=float, default=9750e6, help='LNB offset frequency in Hz')
    parser.add_argument('-g', '--gain-factor', type=float, default=50, help='Digital gain factor')
    parser.add_argument('-b', '--bandwidth', type=float,default=100e6,help='Bandwidth in float 100e6 = 100mhz')
    args = parser.parse_args()

    main(args)
    
gc.collect()