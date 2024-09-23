import os
import re
import gc
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from astropy.io import fits
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

from image_gen.spectra import save_spectra, save_enhanced_spectra, clip_indices
from image_gen.waterfall import save_waterfall_image
from image_gen.psd import calculate_and_save_psd
from image_gen.lofar import generate_lofar_image
from image_gen.save_plots import spectrogram_plot, analyze_signal_strength
from image_gen.spectral_line import (
    create_spectral_line_image, brightness_temp_plot, calculate_brightness_temperature, 
    create_spectral_line_profile, plot_observation_position
)
from image_gen.energy import run_fft_processing
from image_gen.intensity_map import create_intensity_map
from image_gen.simulate_rotational_vibrational_transitions import simulate_rotational_vibrational_transitions
from image_gen.position_velocity import create_position_velocity_diagram
from image_gen.gyrosynchrotron import identify_gyrosynchrotron_emission
from advanced_signal_processing import advanced_signal_processing_pipeline
from compute_bandwidth import compute_bandwidth_and_cutoffs, process_chunk, compute_gpu_fft, cyclostationary_feature_detection, classify_signals
import cupy as cp

# Setup logging
log_file = "output.log"
max_log_size = 10 * 1024 * 1024  # 10 MB
backup_count = 5

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('astropy').setLevel(logging.WARNING)
logging.getLogger('scipy').setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count)

formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Constants
speed_of_light = 299792458  # meters per second
EARTH_ROTATION_RATE = 15  # degrees per hour
tolerance = 1e6
notch_freq = 9750  # MHz
notch_width = 30  # MHz

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

def extract_date_from_filename(filename):
    pattern = r'data_(\d{6})_(\d{6})_'  # Matches YYMMDD_HHMMSS format
    match = re.search(pattern, filename)
    header = fits.Header()
    if match:
        date = match.group(1)
        time = match.group(2)
        logging.debug(f"Extracted date: {date}, time: {time} from filename: {filename}")
    else:
        current_datetime = datetime.now()
        date = current_datetime.strftime('%y%m%d')
        time = current_datetime.strftime('%H%M%S')
        logging.warning(f"Date/time pattern not found in filename: {filename}. Using current date/time.")

    with fits.open(filename, ignore_missing_simple=True) as hdul:
        header['DATE'] = f"20{date[:2]}-{date[2:4]}-{date[4:]}"
        hdu = fits.PrimaryHDU(header=header)
        hdul.append(hdu)
    return date, time

def extract_observation_start_time(fits_filename):
    with fits.open(fits_filename, ignore_missing_simple=True) as hdul:
        header = hdul[0].header
        start_time_str = header.get('DATE', '')
        try:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            try:
                start_time = datetime.strptime(start_time_str, '%Y-%m-%d')
            except ValueError:
                logging.error(f"Unable to parse date from FITS header: {start_time_str}")
                return None
    return start_time

def calculate_rotation_angle(start_time):
    elapsed_time = datetime.now() - start_time
    rotation_angle = int((elapsed_time.total_seconds() / 3600) * EARTH_ROTATION_RATE) % 360
    return rotation_angle

def save_spectrogram_and_signal_strength(freq, fft_values, sampling_rate, output_dir, date, time, lat, lon, duration_hours):
    logging.info(f"Generating spectrogram for {date} {time}")
    spectrogram_plot(freq, fft_values, sampling_rate, output_dir, date, time, lat, lon, duration_hours)
    logging.info(f"Spectrogram and signal strength plots saved for {date} {time}")

def load_fits_file(filename):
    with fits.open(filename) as hdul:
        data = hdul[0].data
    return data

def process_and_save_chunk(chunk, output_file):
    if len(chunk) == 0:
        logging.warning("Received an empty chunk for processing.")
        return chunk

    fft_chunk = np.fft.fft(chunk)
    logging.debug(f"FFT of chunk: {fft_chunk}")

    try:
        with open(output_file, 'ab') as f:
            np.save(f, fft_chunk)
    except Exception as e:
        logging.error(f"Failed to save chunk: {str(e)}")

    logging.info(f"Chunk size: {fft_chunk.nbytes} bytes")
    return fft_chunk


def process_fft_file(input_fits, output_dir, fs, date, time, lat, lon, duration,center_frequency,low_cutoff,high_cutoff ,chunk_size=1024):
    temp_output_file = os.path.join(output_dir, "full_processed_data.npy")
    logging.debug(f"Full processed data file path: {temp_output_file}")

    try:
        with fits.open(input_fits) as hdul:
            data = hdul[0].data + 1j * hdul[1].data
            if data.ndim != 1:
                raise ValueError("FITS data must be a 1D array.")
            total_rows = data.shape[0]

        valid_rows = (total_rows // chunk_size) * chunk_size
        data = data[:valid_rows]
        
        # Use np.memmap with complex64 dtype
        full_data = np.memmap(temp_output_file, dtype='complex64', mode='w+', shape=(valid_rows, 2))
        
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []
            for chunk_start in range(0, valid_rows, chunk_size):
                chunk_end = chunk_start + chunk_size
                chunk = data[chunk_start:chunk_end]
                futures.append(executor.submit(process_chunk, chunk, fs, full_data, chunk_start, 0.1))

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"An error occurred during chunk processing: {str(e)}")
        
        full_data.flush()
        generate_full_dataset_images(temp_output_file, output_dir, fs, date, time, lat, lon, duration,center_frequency,low_cutoff,high_cutoff)

    except Exception as e:
        logging.error(f"An error occurred while processing {input_fits}: {str(e)}")
    finally:
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)
            logging.debug(f"Temporary file {temp_output_file} deleted.")

def perform_gpu_fft(time_domain_signal, chunk_size=1024*1024):
    result = np.zeros_like(time_domain_signal, dtype=np.complex128)
    
    for i in range(0, len(time_domain_signal), chunk_size):
        chunk = time_domain_signal[i:i+chunk_size]
        
        try:
            # Transfer chunk to GPU
            data_gpu = cp.asarray(chunk)
            
            # Perform FFT on chunk
            fft_result = cp.fft.fft(data_gpu)
            
            # Transfer result back to CPU
            result[i:i+chunk_size] = cp.asnumpy(fft_result)
        except cp.cuda.memory.OutOfMemoryError:
            # Fallback to CPU FFT if GPU memory allocation fails
            result[i:i+chunk_size] = np.fft.fft(chunk)
        
        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()
    
    return result



def generate_full_dataset_images(data_file, output_dir, fs, date, time, lat, lon, duration,center_frequency,low_cutoff,high_cutoff):
    file_size = os.path.getsize(data_file)
    num_rows = file_size // (np.dtype('complex64').itemsize * 2)

    full_data = np.memmap(data_file, dtype='complex64', mode='r', shape=(num_rows, 2))
    logging.debug("Before compute bandwidth:")
    logging.debug(f"Center Frequency: {center_frequency}")
    if full_data.ndim == 2:
        initial_bandwidth, _, _, filtered_freq, filtered_fft_values = compute_bandwidth_and_cutoffs(full_data, center_frequency, low_cutoff, high_cutoff)
    else:
        logging.error(f"Unexpected data shape: {full_data.shape}")

    filtered_fft_values = advanced_signal_processing_pipeline(filtered_freq, filtered_fft_values, fs, center_frequency, initial_bandwidth, EARTH_ROTATION_RATE)
    logging.debug(f"Filtered frequency range After Compute Bandwidth: {filtered_freq.min()} to {filtered_freq.max()} Hz")
    # fft_nottime = perform_gpu_fft(filtered_fft_values)
    logging.debug("After No time")
    logging.debug(f"Filtered frequency range After Compute Bandwidth: {filtered_freq.min()} to {filtered_freq.max()} Hz")
    
    peaks, troughs = find_emission_absorption_lines(filtered_freq, filtered_fft_values, low_cutoff, high_cutoff)
    logging.debug(f"peaks: {peaks}, troughs: {troughs}")
    logging.debug(f"Bandwidth: {initial_bandwidth}, lowcutoff: {low_cutoff}, highcutoff: {high_cutoff}")
    logging.debug(f"Filtered frequency range: {filtered_freq.min()} to {filtered_freq.max()} Hz")
    logging.debug(f"Filtered FFT values range: {np.min(np.abs(filtered_fft_values))} to {np.max(np.abs(filtered_fft_values))}")
    logging.debug(f"Number of filtered frequency points: {len(filtered_freq)}")
    logging.debug(f"Number of filtered FFT values: {len(filtered_fft_values)}")

    if args.center_frequency < 1000e6:
        identify_gyrosynchrotron_emission(filtered_freq, filtered_fft_values, center_frequency, output_dir, date, time)
        if args.center_frequency < 250e6:
            generate_lofar_image(filtered_fft_values,filtered_fft_values, filtered_freq, output_dir, date, time, lat, lon, duration)
            
    save_waterfall_image(filtered_freq, filtered_fft_values, output_dir, date, time, duration, center_frequency, initial_bandwidth, peaks, troughs, low_cutoff, high_cutoff)
    
    create_intensity_map(filtered_freq, filtered_fft_values, fs, output_dir, date, time, temperature=2.7)
    save_spectrogram_and_signal_strength(filtered_freq, filtered_fft_values, fs, output_dir, date, time, lat, lon, duration)
    calculate_and_save_psd(filtered_freq, filtered_fft_values, fs, output_dir, date, time, center_frequency, initial_bandwidth, low_cutoff, high_cutoff)
    save_spectra(filtered_freq, filtered_fft_values, peaks, troughs, output_dir, date, time)
    save_enhanced_spectra(filtered_freq, filtered_fft_values, peaks, troughs, output_dir, date, time)
    analyze_signal_strength(filtered_freq, filtered_fft_values, output_dir, date, time)
    create_spectral_line_image(filtered_fft_values, filtered_fft_values, peaks, troughs, output_dir, date, time)
    brightness_temp_plot(filtered_freq, filtered_fft_values, peaks, troughs, output_dir, date, time, lat, lon, duration)
    plot_observation_position(output_dir, date, time, lat, lon, duration)

    logging.debug("Processing Energy Brightness Flux")
    run_fft_processing(filtered_freq, filtered_fft_values, center_frequency, output_dir, date, time)
    simulate_rotational_vibrational_transitions(filtered_fft_values, fs, center_frequency, initial_bandwidth, output_dir, date, time)
    create_position_velocity_diagram(filtered_fft_values, fs, output_dir, date, time)
    
    del full_data


def find_emission_absorption_lines(filtered_freq, filtered_fft_values, low_cutoff, high_cutoff):
    logging.debug(f"Input shapes: filtered_freq: {filtered_freq.shape}, filtered_fft_values: {filtered_fft_values.shape}")

    # Separate real and imaginary parts
    real_part = np.real(filtered_fft_values)
    imag_part = np.imag(filtered_fft_values)

    logging.debug(f"Real part shape: {real_part.shape}, Imag part shape: {imag_part.shape}")

    # Apply median filter to real and imaginary parts separately
    smoothed_real = medfilt(real_part, kernel_size=5)
    smoothed_imag = medfilt(imag_part, kernel_size=5)

    logging.debug(f"Smoothed real shape: {smoothed_real.shape}, Smoothed imag shape: {smoothed_imag.shape}")

    # Recombine the smoothed parts
    smoothed_fft = smoothed_real + 1j * smoothed_imag

    # Calculate magnitude of the smoothed FFT
    magnitude_fft = np.abs(smoothed_fft)

    logging.debug(f"Magnitude FFT shape: {magnitude_fft.shape}")

    mean_fft = np.mean(magnitude_fft)
    std_fft = np.std(magnitude_fft)
    threshold = mean_fft + 2 * std_fft

    logging.debug(f"Threshold: {threshold}")

    try:
        peak_indices, _ = find_peaks(magnitude_fft, height=threshold, distance=20)
        peaks = filtered_freq[peak_indices]

        trough_indices, _ = find_peaks(-magnitude_fft, height=-threshold, distance=20)
        troughs = filtered_freq[trough_indices]

        peaks = peaks[peaks != 0]

        logging.debug(f"Number of peaks found: {len(peaks)}")
        logging.debug(f"Number of troughs found: {len(troughs)}")

    except Exception as e:
        logging.error(f"Error in find_peaks: {str(e)}", exc_info=True)
        return [], []

    return peaks, troughs

def main(args):
    import sys
    try:
        input_fits = args.input_fits
        output_dir = args.output_dir
        sample_rate = args.fs
        lat = args.latitude
        lon = args.longitude
        duration = args.duration
        center_frequency = args.center_frequency
        low_cutoff = args.low_cutoff
        high_cutoff = args.high_cutoff
        logging.debug(f"Input FITS file: {input_fits}")
        logging.debug(f"Output directory: {output_dir}")
        logging.debug(f"Sample rate: {sample_rate} Hz")
        logging.debug(f"Latitude: {lat} degrees")
        logging.debug(f"Longitude: {lon} degrees")
        logging.debug(f"Duration: {duration} seconds")
        logging.debug(f"Center Frequency: {center_frequency} Hz")
        logging.debug(f"Low Cutoff: {low_cutoff} Hz")
        logging.debug(f"High Cutoff: {high_cutoff} Hz")

        os.makedirs(output_dir, exist_ok=True)
                
        date, time = extract_date_from_filename(input_fits)
        start_time = extract_observation_start_time(input_fits)
        
        if start_time:
            process_fft_file(input_fits, output_dir, sample_rate, date, time, lat, lon, duration, center_frequency, low_cutoff, high_cutoff)
            print("Signal processing and image generation completed successfully.")
        else:
            print("Failed to extract start time from FITS file.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
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
    parser.add_argument('--low-cutoff', type=float, default=None, help='Low cutoff frequency in Hz.')
    parser.add_argument('--high-cutoff', type=float, default=None, help='High cutoff frequency in Hz.')
    args = parser.parse_args()
    
    main(args)
