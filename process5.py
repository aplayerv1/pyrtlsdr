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

from image_gen.spectra import save_spectra, save_spectra2
from image_gen.waterfall import save_waterfall_image
from image_gen.psd import calculate_and_save_psd
from image_gen.lofar import generate_lofar_image
from image_gen.save_plots import save_plots_to_png, analyze_signal_strength
from image_gen.spectral_line import create_spectral_line_image, brightness_temp_plot, calculate_brightness_temperature, create_spectral_line_profile
from image_gen.energy import create_energy_level_diagram
from image_gen.intensity_map import create_intensity_map
from image_gen.simulate_rotational_vibrational_transitions import simulate_rotational_vibrational_transitions
from image_gen.position_velocity import create_position_velocity_diagram

gc.enable()

# Setup logging with a file size limit
log_file = "output.log"
max_log_size = 10 * 1024 * 1024  # 10 MB
backup_count = 5

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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
delta_lambda = 1e-2  # Change in wavelength in 
lambda_0 = 0.211      # Rest wavelength in meters (500 nanometers)
EARTH_ROTATION_RATE = 15  # degrees per hour
tolerance = 1e6

# Sampling frequency in Hz
# Low band LO frequency in MHz
notch_freq = 9750
# Notch width in MHz
notch_width = 30

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

def compute_spectrogram(samples, fs):
    logging.debug(f"Input samples range: {np.min(samples)} to {np.max(samples)}")
    nfft = 256
    noverlap = 128
    spectrogram, frequencies, times, _ = plt.specgram(samples, NFFT=nfft, Fs=fs, noverlap=noverlap)
    spectrogram = np.where(spectrogram == 0, 1e-10, spectrogram)
    logging.debug(f"Spectrogram range: {np.min(spectrogram)} to {np.max(spectrogram)}")
    return spectrogram, frequencies, times

def save_spectrogram_and_signal_strength(signal_data, sampling_rate, output_dir, date, time, lat, lon, duration_hours):
    spectrogram, frequencies, times = compute_spectrogram(signal_data, sampling_rate)
    save_plots_to_png(spectrogram, signal_data, frequencies, times, output_dir, date, time, lat, lon, duration_hours)

def find_emission_absorption_lines(freq, fft_values, height_threshold=0.1, distance=20):
    magnitude = np.abs(fft_values)
    peaks, _ = find_peaks(magnitude, height=height_threshold, distance=distance)
    troughs, _ = find_peaks(-magnitude, height=height_threshold, distance=distance)
    return peaks, troughs

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
            
        # Create a memory-mapped file for the full dataset
        full_data = np.memmap(temp_output_file, dtype='complex64', mode='w+', shape=(total_rows,))

        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []

            for start_row in range(0, total_rows, chunk_size):
                end_row = min(start_row + chunk_size, total_rows)
                chunk = data[start_row:end_row]
                futures.append(executor.submit(process_chunk, chunk, fs, full_data, start_row))

            # Wait for all futures to complete
            for future in futures:
                future.result()

        # Flush the memmap to ensure all data is written to disk
        full_data.flush()

        # Generate full dataset images
        generate_full_dataset_images(temp_output_file, output_dir, fs, date, time, lat, lon, duration)

    except Exception as e:
        logging.error(f"An error occurred while processing {input_fits}: {str(e)}")
    finally:
        # Delete the temporary file
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)
            logging.debug(f"Temporary file {temp_output_file} deleted.")

def process_chunk(chunk,fs, full_data, start_row):
    processed_chunk = compute_fft(chunk, fs, args.center_frequency)
    if isinstance(processed_chunk, tuple):
        # Assuming the first element of the tuple is the data we want
        processed_chunk = processed_chunk[0]
    full_data[start_row:start_row+processed_chunk.size] = processed_chunk.ravel()

def generate_full_dataset_images(data_file, output_dir, fs, date, time, lat, lon, duration):
    full_data = np.memmap(data_file, dtype='complex64', mode='r')
    logging.info(f"Processing data with duration: {args.duration} seconds")
    duration_hours = args.duration / 3600  # Convert seconds to hours
    logging.debug(f"Duration in hours: {duration_hours}")

    rotation_angle = duration_hours * EARTH_ROTATION_RATE
    logging.debug(f"Calculated rotation angle: {rotation_angle} degrees")

    logging.info("Applying rotation to data")
    full_data = apply_rotation(full_data, rotation_angle)

    logging.info("Removing DC offset")
    full_data = remove_dc_offset(full_data)

    logging.info("Denoising")
    full_data = denoise_signal(full_data)

    logging.info(f"Removing LNB effect with fs={fs}, notch_freq={notch_freq}, notch_width={notch_width}")
    full_data = remove_lnb_effect(full_data, fs, notch_freq, notch_width)

    logging.debug(f"Data shape after processing: {full_data.shape}")
    full_data = (np.abs(full_data) - np.min(np.abs(full_data))) / (np.max(np.abs(full_data)) - np.min(np.abs(full_data)))


    initial_bandwidth, low_cutoff, high_cutoff, freq, fft_values = compute_bandwidth_and_cutoffs(full_data, fs, args.center_frequency)
    peaks, troughs = find_emission_absorption_lines(freq, fft_values,low_cutoff,high_cutoff)

    # Generate waterfall image
    save_waterfall_image(full_data, output_dir, date, time, duration, args.center_frequency, initial_bandwidth,peaks,troughs,low_cutoff,high_cutoff)
    
    save_spectrogram_and_signal_strength(full_data, fs, output_dir, date, time, lat, lon, duration)
    
    calculate_and_save_psd(full_data, args.fs, output_dir, date, time, args.center_frequency, initial_bandwidth)

    save_spectra(freq, fft_values, peaks, troughs, output_dir, date, time)
    save_spectra2(freq, fft_values, peaks, troughs, output_dir, date, time)

     # Analyze and save signal strength
    analyze_signal_strength(full_data, output_dir, date, time)

    create_spectral_line_image(freq, fft_values, peaks, troughs, output_dir, date, time)

    brightness_temp_plot(freq, fft_values, peaks, troughs, output_dir, date, time, lat, lon, duration)

    logging.debug(f"Frequency {np.max(freq)}")
    if np.max(freq) < 100e6:
        generate_lofar_image(full_data, output_dir, date, time, lat, lon, duration)
        logging.debug("After Loafar")

    logging.debug("After p velocity")
    create_energy_level_diagram(args.center_frequency, output_dir, date, time)
    logging.debug("After level")
    create_intensity_map(full_data, fs, output_dir, date, time)
    logging.debug("After map")
    simulate_rotational_vibrational_transitions(full_data, fs, args.center_frequency, initial_bandwidth, output_dir, date, time)
    logging.debug("After vibra")
    create_position_velocity_diagram(full_data, fs, output_dir, date, time)
    logging.debug("After vdiag")
    # Close the memmap
    del full_data

def find_emission_absorption_lines(freq, fft_values, low_cutoff, high_cutoff, height_threshold=0.1, distance=20):
    # Filter the frequency range
    mask = (freq >= low_cutoff) & (freq <= high_cutoff)
    filtered_freq = freq[mask]
    filtered_fft = fft_values[mask]

    magnitude = np.abs(filtered_fft)
    peaks, _ = find_peaks(magnitude, height=height_threshold, distance=distance)
    troughs, _ = find_peaks(-magnitude, height=height_threshold, distance=distance)

    # Convert peak and trough indices back to original frequency array
    original_peaks = np.where(mask)[0][peaks]
    original_troughs = np.where(mask)[0][troughs]

    return original_peaks, original_troughs

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

def compute_fft(signal_data, sampling_rate, center_frequency):
    
    logging.debug(f"Starting FFT computation - sampling_rate: {sampling_rate}, center_frequency: {center_frequency}")
    logging.debug(f"Input signal_data shape: {signal_data.shape}, dtype: {signal_data.dtype}")
    # Remove DC component
    signal_data = amplify_signal(signal_data,2)
    signal_data = remove_dc_offset(signal_data)
    logging.debug(f"Signal data after DC offset removal - Mean: {np.mean(signal_data)}, Std: {np.std(signal_data)}")
    
    # Apply notch filter
    n_freq = 60  # Example: removing 60 Hz power line noise
    signal_data = apply_notch_filter(signal_data, n_freq, sampling_rate)
    logging.debug(f"Signal data after notch filter - Mean: {np.mean(signal_data)}, Std: {np.std(signal_data)}")
    
    # Apply bandpass filter
    low_cutoff = center_frequency - (0.5e6)  # Adjust this based on your requirements
    high_cutoff = center_frequency + (0.5e6)  # Adjust this based on your requirements
    signal_data = bandpass_filter(signal_data, sampling_rate, low_cutoff, high_cutoff)
    
    n = len(signal_data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    actual_freq = freq * sampling_rate  # Calculate actual frequencies in Hz
    
    logging.debug(f"Computed FFT frequencies (actual): {actual_freq[:10]}")  # Display first 10 for brevity
    
    fft_values = np.fft.fft(signal_data)
    logging.debug(f"Computed FFT values (first 10): {fft_values[:10]}")  # Display first 10 for brevity
    
    # Return only the positive half of the frequencies
    half_n = n // 2
    logging.debug(f"FFT output - freq shape: {actual_freq[:half_n].shape}, values shape: {np.abs(fft_values[:half_n]).shape}")
    return actual_freq[:half_n], np.abs(fft_values[:half_n])

def compute_bandwidth_and_cutoffs(amplified_signal, fs, center_frequency):
    # Compute FFT
    freq, fft_values = compute_fft(amplified_signal, fs, center_frequency)
    logging.debug("Computed FFT frequencies and values.")
    
    # Calculate the initial bandwidth based on significant frequencies
    threshold = np.max(fft_values) * 0.1  # Example threshold at 10% of max value
    significant_freqs = freq[fft_values > threshold]
    
    if len(significant_freqs) == 0:
        initial_bandwidth = 0
    else:
        initial_bandwidth = significant_freqs[-1] - significant_freqs[0]
    
    logging.debug(f"Initial Bandwidth: {initial_bandwidth} Hz")
    
    # Calculate low and high cutoffs
    low_cutoff = center_frequency - initial_bandwidth / 2
    high_cutoff = center_frequency + initial_bandwidth / 2
    
    logging.debug(f"Low Cutoff: {low_cutoff} Hz")
    logging.debug(f"High Cutoff: {high_cutoff} Hz")
    
    return initial_bandwidth, low_cutoff, high_cutoff, freq, fft_values


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