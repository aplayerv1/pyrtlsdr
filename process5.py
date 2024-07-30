import os, re, gc, argparse
from datetime import datetime
import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy import signal, ndimage
from scipy.signal import lfilter, find_peaks, medfilt

from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import cProfile
import librosa
import librosa.display
import pywt

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import logging
from logging.handlers import RotatingFileHandler
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import logging

from image_gen.spectra import save_spectra,save_enhanced_spectra, clip_indices
from image_gen.waterfall import save_waterfall_image
from image_gen.psd import calculate_and_save_psd
from image_gen.lofar import generate_lofar_image
from image_gen.save_plots import spectrogram_plot, analyze_signal_strength
from image_gen.spectral_line import create_spectral_line_image, brightness_temp_plot, calculate_brightness_temperature, create_spectral_line_profile, plot_observation_position
from image_gen.energy import run_fft_processing
from image_gen.intensity_map import create_intensity_map
from image_gen.simulate_rotational_vibrational_transitions import simulate_rotational_vibrational_transitions
from image_gen.position_velocity import create_position_velocity_diagram
from image_gen.gyrosynchrotron import process_frequency_range
gc.enable()

# Setup logging with a file size limit
log_file = "output.log"
max_log_size = 10 * 1024 * 1024  # 10 MB
backup_count = 5

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('astropy').setLevel(logging.WARNING)
logging.getLogger('scipy').setLevel(logging.WARNING)
logging.getLogger('pywt').setLevel(logging.WARNING)
logging.getLogger('librosa').setLevel(logging.WARNING)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Define the speed of light in meters per second
speed_of_light = 299792458  # meters per second
EARTH_ROTATION_RATE = 15  # degrees per hour
tolerance = 1e6
# Low band LO frequency in MHz
notch_freq = 9750
# Notch width in MHz
notch_width = 30
magnetic_field_strength=1
k_B = 1.38e-23 

class VLA:
    def __init__(self, initial_size=1024):
        self.data = np.zeros(initial_size)
        self.size = 0

    def append(self, value):
        if self.size >= self.data.size:
            self.data = np.resize(self.data, self.data.size * 2)
        self.data[self.size] = value
        self.size += 1

    def get_data(self):
        return self.data[:self.size]

def remove_lnb_effect(signal, fs, notch_freq, notch_width):
    logging.debug("Starting remove_lnb_effect function")
    
    signal = np.asarray(signal, dtype=np.float64)
    logging.debug(f"Original signal size: {signal.size}")
    logging.debug(f"Original signal mean: {np.mean(signal)}")
    
    t = np.tan(np.pi * notch_width / fs)
    logging.debug(f"Computed tan: {t}")
    
    beta = (1 - t) / (1 + t)
    logging.debug(f"Computed beta: {beta}")
    
    gamma = -np.cos(2 * np.pi * notch_freq / fs)
    logging.debug(f"Computed gamma: {gamma}")
    
    b = [1, gamma * (1 + beta), beta]
    a = [1, gamma * (1 - beta), -beta]
    
    logging.debug(f"Filter coefficients (b): {b}")
    logging.debug(f"Filter coefficients (a): {a}")
    
    processed_signal = lfilter(b, a, signal)
    logging.debug("Applied notch filter to the signal")
    
    logging.debug(f"Processed signal size: {processed_signal.size}")
    logging.debug(f"Processed signal mean: {np.mean(processed_signal)}")
    
    return processed_signal

def extract_date_from_filename(filename):
    # Extract date and time from filename using regular expression
    pattern = r'(\d{8})_(\d{6})'  # Assuming the format is YYYYMMDD_HHMMSS
    match = re.search(pattern, filename)
    header = fits.Header()
    if match:
        date = match.group(1)
        time = match.group(2)
    else:
        # If pattern not found, use current date and time
        current_datetime = datetime.now()
        date = current_datetime.strftime('%Y%m%d')
        time = current_datetime.strftime('%H%M%S')
        logging.warning(f"Date/time pattern not found in filename: {filename}. Using current date/time.")

    with fits.open(filename, ignore_missing_simple=True) as hdul:
        header['DATE'] = date
        hdu = fits.PrimaryHDU(header=header)
        hdul.append(hdu)
    return date, time

def extract_observation_start_time(fits_filename):
    with fits.open(fits_filename, ignore_missing_simple=True) as hdul:
        header = hdul[0].header
        start_time_str = header['DATE']
        try:
            # Try parsing with time information
            start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            try:
                # If that fails, try parsing just the date
                start_time = datetime.strptime(start_time_str, '%Y-%m-%d')
            except ValueError:
                # If both fail, log an error and return None or raise an exception
                logging.error(f"Unable to parse date from FITS header: {start_time_str}")
                return None
    return start_time

def calculate_rotation_angle(start_time):
    elapsed_time = datetime.now() - start_time
    rotation_angle = int((elapsed_time.total_seconds() / 3600) * EARTH_ROTATION_RATE) % 4
    return rotation_angle

def apply_rotation(data, rotation_angle):
    data_2d = data[:, np.newaxis]
    rotated_data_2d = ndimage.rotate(data_2d, rotation_angle, reshape=False, mode='nearest')
    rotated_data = rotated_data_2d.flatten()
    return rotated_data


def save_spectrogram_and_signal_strength(freq, fft_values, sampling_rate, output_dir, date, time, lat, lon, duration_hours):
    logging.info(f"Generating spectrogram for {date} {time}")
    logging.debug(f"Spectrogram generated with shape {freq.shape}")
    spectrogram_plot(freq, fft_values,sampling_rate, output_dir, date, time, lat, lon, duration_hours)
    logging.info(f"Spectrogram and signal strength plots saved for {date} {time}")

def load_fits_file(filename):
    with fits.open(filename) as hdul:
        data = hdul[0].data
    return data

def denoise_signal(data, wavelet='db1', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(new_coeffs, wavelet)

def amplify_signal(data, factor=1):
    return data * factor

def process_and_save_chunk(chunk, output_file):
    if len(chunk) == 0:
        logging.warning("Received an empty chunk for processing.")
        return chunk

    # Compute the FFT of the chunk
    fft_chunk = np.fft.fft(chunk)
    logging.debug(f"FFT of chunk: {fft_chunk}")

    try:
        with open(output_file, 'ab') as f:
            np.save(f, fft_chunk)
    except Exception as e:
        logging.error(f"Failed to save chunk: {str(e)}")

    # Log the size of the processed chunk
    logging.info(f"Chunk size: {fft_chunk.nbytes} bytes")
    return fft_chunk

def process_fft_file(input_fits, output_dir, fs, date, time, lat, lon, duration, chunk_size=1024):
    temp_output_file = os.path.join(output_dir, "full_processed_data.npy")
    logging.debug(f"Full processed data file path: {temp_output_file}")

    try:
        with fits.open(input_fits) as hdul:
            data = hdul[0].data + 1j * hdul[1].data
            total_rows = data.shape[0]
            
        # Trim data to be a multiple of chunk_size
        valid_rows = (total_rows // chunk_size) * chunk_size
        data = data[:valid_rows]
        
        full_data = np.memmap(temp_output_file, dtype='complex64', mode='w+', shape=(valid_rows, 2))

        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []

            for chunk_start in range(0, valid_rows, chunk_size):
                chunk_end = chunk_start + chunk_size
                chunk = data[chunk_start:chunk_end]
                futures.append(executor.submit(process_chunk, chunk, fs, full_data, chunk_start, 0.1))

            for future in futures:
                future.result()

        full_data.flush()

        generate_full_dataset_images(temp_output_file, output_dir, fs, date, time, lat, lon, duration)

    except Exception as e:
        logging.error(f"An error occurred while processing {input_fits}: {str(e)}")
    finally:
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)
            logging.debug(f"Temporary file {temp_output_file} deleted.")

def process_chunk(chunk, sample_rate, full_data, start_row, threshold):
    # Perform FFT on the chunk
    fft_result = np.fft.fft(chunk)
    freq_bins = np.fft.fftfreq(len(chunk), d=1/sample_rate)
    logging.debug(f"FFT output shape: {fft_result.shape}")

    # Filter the FFT result
    mask = np.abs(fft_result) > threshold
    filtered_fft = fft_result[mask]
    filtered_freqs = freq_bins[mask]
    
    logging.debug(f"Filtered data - mean: {np.mean(filtered_fft)}, std: {np.std(filtered_fft)}")
    
    if len(filtered_fft) == 0:
        return 0

    end_row = start_row + len(filtered_fft)
    
    if end_row > full_data.shape[0]:
        logging.error("Filtered FFT result exceeds full_data capacity.")
        return 0

    full_data[start_row:end_row, 0] = filtered_freqs
    full_data[start_row:end_row, 1] = filtered_fft

    return len(filtered_fft)


def generate_full_dataset_images(data_file, output_dir, fs, date, time, lat, lon, duration):
    file_size = os.path.getsize(data_file)
    num_rows = file_size // (np.dtype('complex64').itemsize * 2)

    full_data = np.memmap(data_file, dtype='complex64', mode='r', shape=(num_rows, 2))
    logging.debug("Before compute bandwidth:")

    if full_data.ndim == 2:
        initial_bandwidth, low_cutoff, high_cutoff, filtered_freq, filtered_fft_values = compute_bandwidth_and_cutoffs(full_data, args.center_frequency)
    else:
        logging.error(f"Unexpected data shape: {full_data.shape}")
    logging.debug(f"Filtered frequency range After Compute Bandwidth: {filtered_freq.min()} to {filtered_freq.max()} Hz")
    logging.debug("After Compute Bandwidth:")
    # Handle the error appropriately
    peaks, troughs = find_emission_absorption_lines(filtered_freq, filtered_fft_values, low_cutoff, high_cutoff)
    logging.debug("After find emissions:")
    logging.debug(f"peaks: {peaks}, troughs: {troughs}")
    logging.debug(f"Bandwidth: {initial_bandwidth}, lowcutoff: {low_cutoff}, highcutoff: {high_cutoff}")
    logging.debug(f"Filtered frequency range: {filtered_freq.min()} to {filtered_freq.max()} Hz")
    logging.debug(f"Filtered FFT values range: {np.min(np.abs(filtered_fft_values))} to {np.max(np.abs(filtered_fft_values))}")
    logging.debug(f"Number of filtered frequency points: {len(filtered_freq)}")
    logging.debug(f"Number of filtered FFT values: {len(filtered_fft_values)}")

    if args.center_frequency < 1000e6:
        logging.debug(f"frequency is {filtered_freq.max} < 1000e6")
        process_frequency_range(filtered_freq,filtered_fft_values,args.center_frequency,output_dir,date,time)
        if args.center_frequency < 100e6:
            generate_lofar_image(filtered_fft_values, output_dir, date, time, lat, lon, duration)
        
    # Generate waterfall image
    save_waterfall_image(filtered_freq, filtered_fft_values, output_dir, date, time, duration, args.center_frequency, initial_bandwidth, peaks, troughs, low_cutoff, high_cutoff)
    
    create_intensity_map(filtered_freq,filtered_fft_values, fs, output_dir, date, time,temperature=2.7)

    save_spectrogram_and_signal_strength(filtered_freq, filtered_fft_values, fs, output_dir, date, time, lat, lon, duration)
    
    calculate_and_save_psd(filtered_freq, filtered_fft_values, fs, output_dir, date, time, args.center_frequency, initial_bandwidth,low_cutoff,high_cutoff)

    save_spectra(filtered_freq, filtered_fft_values, peaks, troughs, output_dir, date, time)
    save_enhanced_spectra(filtered_freq, filtered_fft_values, peaks, troughs, output_dir, date, time)

    #  # Analyze and save signal strength
    analyze_signal_strength(filtered_freq, filtered_fft_values, output_dir, date, time)

    create_spectral_line_image(filtered_fft_values, filtered_fft_values, peaks, troughs, output_dir, date, time)

    brightness_temp_plot(filtered_freq,filtered_fft_values, peaks, troughs, output_dir, date, time, lat, lon, duration)
    plot_observation_position(output_dir,date,time,lat,lon,duration)

    logging.debug("Processing Energy Brightness Flux")
    run_fft_processing(filtered_freq,filtered_fft_values,args.center_frequency, output_dir, date, time)

    simulate_rotational_vibrational_transitions(full_data, fs, args.center_frequency, initial_bandwidth, output_dir, date, time)

    create_position_velocity_diagram(full_data, fs, output_dir, date, time)


    # Close the memmap
    del full_data

def find_emission_absorption_lines(filtered_freq, filtered_fft_values, low_cutoff, high_cutoff):
    # Apply median filter to smooth the data
    smoothed_fft = medfilt(filtered_fft_values, kernel_size=5)
    
    # Calculate dynamic threshold
    mean_fft = np.mean(smoothed_fft)
    std_fft = np.std(smoothed_fft)
    threshold = mean_fft + 2 * std_fft
    
    # Find peaks
    peak_indices, _ = find_peaks(smoothed_fft, height=threshold, distance=20)
    peaks = filtered_freq[peak_indices]
    
    # Find troughs
    trough_indices, _ = find_peaks(-smoothed_fft, height=-threshold, distance=20)
    troughs = filtered_freq[trough_indices]
    
    # Remove 0 Hz peaks if present
    peaks = peaks[peaks != 0]
    
    return peaks, troughs
   

def remove_dc_offset(signal):
    mean_val = np.mean(signal)
    logging.debug(f"DC Offset Mean: {mean_val}")
    return signal - mean_val

def apply_notch_filter(data, center_freq, sampling_rate, quality_factor=30):
    logging.debug(f"Applying notch filter - center_freq: {center_freq}, sampling_rate: {sampling_rate}, quality_factor: {quality_factor}")
    logging.debug(f"Input signal shape: {data.shape}, dtype: {data.dtype}")
    b, a = signal.iirnotch(center_freq / (sampling_rate / 2), quality_factor)
    filtered_data = signal.filtfilt(b, a, data)
    logging.debug(f"Filtered data shape: {filtered_data.shape}, dtype: {filtered_data.dtype}")
    return filtered_data

def bandpass_filter(data, fs, lowcut, highcut, order=5):
    logging.debug(f"Bandpass filter input - fs: {fs}, lowcut: {lowcut}, highcut: {highcut}, order: {order}")
    nyq = 0.5 * fs
    center_freq = (lowcut + highcut) / 2
    bandwidth = highcut - lowcut
    
    # Shift to baseband
    t = np.arange(len(data)) / fs
    shifted_data = data * np.exp(-2j * np.pi * center_freq * t)
    
    # Apply lowpass filter
    cutoff = bandwidth / (2 * nyq)
    b, a = signal.butter(order, cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, shifted_data)
    
    # Shift back to original frequency
    output_data = filtered_data * np.exp(2j * np.pi * center_freq * t)
    
    logging.debug(f"Filtered data - mean: {np.mean(output_data)}, std: {np.std(output_data)}")
    return output_data.real

def compute_fft(signal_data, sampling_rate, center_frequency, duration, notch_freq, notch_width):
    logging.debug(f"Starting FFT computation - sampling_rate: {sampling_rate}, center_frequency: {center_frequency}")
    logging.debug(f"Input signal_data shape: {signal_data.shape}, dtype: {signal_data.dtype}")

    # Calculate rotation angle
    duration_hours = duration / 3600
    rotation_angle = duration_hours * EARTH_ROTATION_RATE
    logging.debug(f"Calculated rotation angle: {rotation_angle} degrees")

    # Apply rotation
    signal_data = apply_rotation(signal_data, rotation_angle)

    # Remove DC offset
    signal_data = remove_dc_offset(signal_data)

    # Denoise
    signal_data = denoise_signal(signal_data)

    # Remove LNB effect
    signal_data = remove_lnb_effect(signal_data, sampling_rate, notch_freq, notch_width)

    # Existing preprocessing steps
    signal_data = amplify_signal(signal_data, 2)
    signal_data = apply_notch_filter(signal_data, 60, sampling_rate)
    signal_data = bandpass_filter(signal_data, sampling_rate, center_frequency - 0.5e6, center_frequency + 0.5e6)

    n = len(signal_data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    actual_freq = freq * sampling_rate
    fft_values = np.fft.fft(signal_data)

    # Combine frequency and FFT values into a 2D array
    result = np.column_stack((actual_freq, np.abs(fft_values)))

    logging.debug(f"FFT output shape: {result.shape}")

    return result

def compute_bandwidth_and_cutoffs(full_data, center_frequency):
    freq = full_data[:, 0].real  # Extract real part of frequencies
    fft_values = np.abs(full_data[:, 1])  # Use magnitude of FFT values

    logging.debug(f"Frequency range: {freq.min()} to {freq.max()} Hz")
    logging.debug(f"FFT values range: {fft_values.min()} to {fft_values.max()}")

    threshold = np.max(fft_values) * 0.1
    significant_freqs = freq[fft_values > threshold]
    
    logging.debug(f"Threshold: {threshold}")
    logging.debug(f"Number of significant frequencies: {len(significant_freqs)}")

    if len(significant_freqs) < 2:
        initial_bandwidth = max(freq.max() - freq.min(), 1000)  # Minimum 1 kHz bandwidth
    else:
        initial_bandwidth = max(abs(significant_freqs[-1] - significant_freqs[0]), 1000)

    logging.debug(f"Initial bandwidth: {initial_bandwidth} Hz")

    low_cutoff = max(freq.min(), center_frequency - initial_bandwidth / 2)
    high_cutoff = min(freq.max(), center_frequency + initial_bandwidth / 2)
    
    if low_cutoff >= high_cutoff:
        low_cutoff, high_cutoff = freq.min(), freq.max()

    logging.debug(f"Low cutoff: {low_cutoff} Hz")
    logging.debug(f"High cutoff: {high_cutoff} Hz")

    mask = (freq >= low_cutoff) & (freq <= high_cutoff)
    filtered_freq = freq[mask]
    filtered_fft_values = fft_values[mask]

    if len(filtered_freq) == 0:
        logging.warning("No frequencies left after filtering. Reverting to full frequency range.")
        filtered_freq = freq
        filtered_fft_values = fft_values

    logging.debug(f"Number of frequencies after filtering: {len(filtered_freq)}")
    logging.debug(f"Frequency range After Bandwidth and Cutoff: {filtered_freq.min()} to {filtered_freq.max()} Hz")

    return initial_bandwidth, low_cutoff, high_cutoff, filtered_freq, filtered_fft_values


def main(args):
    try:
 
        input_fits = args.input_fits
        output_dir = args.output_dir
        sample_rate = args.fs
        lat = args.latitude
        lon = args.longitude
        duration = args.duration

        logging.debug(f"Input FITS file: {input_fits}")
        logging.debug(f"Output directory: {output_dir}")
        logging.debug(f"Sample rate: {sample_rate} Hz")
        logging.debug(f"Latitude: {lat} degrees")
        logging.debug(f"Longitude: {lon} degrees")
        logging.debug(f"Duration: {duration} seconds")

        os.makedirs(output_dir, exist_ok=True)
                
        date, time = extract_date_from_filename(input_fits)
        # start_time = extract_observation_start_time(input_fits)
        start_time = extract_observation_start_time(input_fits)
        
        process_fft_file(input_fits, output_dir, sample_rate, date, time, lat, lon, duration)

        print("Signal processing and image generation completed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process FITS files and generate images.')
    parser.add_argument('-i', '--input_fits', type=str, required=True, help='Path to the input FITS file.')
    parser.add_argument('-f', '--fft_file', type=str, required=False, help='Path to the input FFT file.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--fs', type=float, default=20e6, help='Sampling frequency in Hz.')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Chunk size for processing data.')
    parser.add_argument('--tolerance', type=float, default=1e6, help='Tolerance for bandpass filter frequency check.')
    parser.add_argument('--center-frequency', type=float, default=1420.4e6, help='Center Frequency.')
    parser.add_argument('--latitude', type=float, required=True, help='Latitude of the observation site in degrees')
    parser.add_argument('--longitude', type=float, required=True, help='Longitude of the observation site in degrees')
    parser.add_argument('--duration', type=float, default=24.0, help='Duration of the observation in hours')

    args = parser.parse_args()
    
    main(args)