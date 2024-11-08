import os
import re
import gc
import argparse
from scipy.interpolate import interp1d
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
from compute_bandwidth import process_chunk, compute_gpu_fft, cyclostationary_feature_detection, classify_signals,robust_compute_bandwidth
from check_file import check_file
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
MAGNETIC_FIELD_STRENGTH = 1

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

def extract_date_from_fits(filename):
    try:
        with fits.open(filename, ignore_missing_simple=True) as hdul:
            # Access the primary HDU's header
            header = hdul[0].header
            
            # Extract DATE and TIME from the header
            date = header.get('DATE', None)
            time = header.get('TIME', None)
            date = datetime.strptime(date, '%Y-%m-%d').strftime('%y%m%d')
            if date is None or time is None:
                logging.warning(f"DATE or TIME not found in header for file: {filename}.")
                return None, None

            logging.debug(f"Extracted date: {date}, time: {time} from header of filename: {filename}")
            return date, time

    except Exception as e:
        logging.error(f"Error extracting date/time from FITS file {filename}: {e}")
        return None, None

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

def save_spectrogram_and_signal_strength(freq, fft_values, sampling_rate, output_dir, date, time, lat, lon, duration_hours, lowcutoff, highcutoff):
    logging.info(f"Generating spectrogram for {date} {time}")
    #spectrogram_plot(frequency, fft_values, sampling_rate, png_location, date, time, lat, lon, duration_hours, lowcutoff, highcutoff):

    spectrogram_plot(freq, fft_values, sampling_rate, output_dir, date, time, lat, lon, duration_hours, lowcutoff, highcutoff)
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


def process_fft_file(input_fits, output_dir, fs, date, time, lat, lon, duration, center_frequency, low_cutoff, high_cutoff, chunk_size=1024):
    temp_output_file = os.path.join(output_dir, "full_processed_data.npy")
    logging.debug(f"Full processed data file path: {temp_output_file}")

    try:
        with fits.open(input_fits) as hdul:
            data = hdul[0].data + 1j * hdul[1].data
            if data.ndim != 1:
                raise ValueError("FITS data must be a 1D array.")
            
            signal_strength = np.abs(data).mean()
            logging.debug(f"Initial signal strength: {signal_strength}")
            if signal_strength < 1e-10:
                data *= 1e3

            nyquist_freq = fs / 2
            threshold_frequency = 1e9
            logging.debug(f"Nyquist frequency: {nyquist_freq} Hz")
            logging.debug(f"Threshold frequency for aliasing: {threshold_frequency} Hz")

            # Enhanced frequency folding calculation
            if center_frequency > threshold_frequency:
                effective_freq = (center_frequency % fs)
                if effective_freq > nyquist_freq:
                    effective_freq = fs - effective_freq
            else:
                effective_freq = center_frequency

            if low_cutoff > threshold_frequency:
                effective_low = (low_cutoff % fs)
                if effective_low > nyquist_freq:
                    effective_low = fs - effective_low
            else:
                effective_low = low_cutoff

            if high_cutoff > threshold_frequency:
                effective_high = (high_cutoff % fs)
                if effective_high > nyquist_freq:
                    effective_high = fs - effective_high
            else:
                effective_high = high_cutoff

            logging.debug(f"Effective Frequency: {effective_freq} Hz")
            logging.debug(f"Effective Low Cutoff: {effective_low} Hz")
            logging.debug(f"Effective High Cutoff: {effective_high} Hz")

            full_fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data), d=1/fs)
            
            folding_factor = int(center_frequency / fs)
            effective_freqs = freqs + (folding_factor * fs)
            
            full_data = np.memmap(temp_output_file, dtype='complex128', mode='w+', shape=(data.shape[0],))
            full_data[:] = full_fft[:]
            full_data.flush()

            generate_full_dataset_images(temp_output_file, output_dir, fs, date, time, lat, lon, duration,
                                      effective_freq, effective_low, effective_high, effective_freqs)

    except Exception as e:
        logging.error(f"An error occurred while processing {input_fits}: {str(e)}")
    finally:
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)
            logging.debug(f"Temporary file {temp_output_file} deleted.")



def generate_full_dataset_images(data_file, output_dir, fs, date, time, lat, lon, duration, effective_freq, effective_low, effective_high, effective_freqs):
    file_size = os.path.getsize(data_file)
    num_rows = file_size // np.dtype('complex128').itemsize
    logging.debug(f"File size: {file_size} bytes")

    full_data = np.memmap(data_file, dtype='complex128', mode='r', shape=(num_rows,))
    
    # Dynamic processing parameters based on frequency range
    if effective_freq > 1e9:
        gain_factor = 8
        filter_quality = 50
        wavelet_level = 2
    elif effective_freq > 100e6:
        gain_factor = 4
        filter_quality = 30
        wavelet_level = 1
    else:
        gain_factor = 2
        filter_quality = 20
        wavelet_level = 1
    
    logging.debug(f"Effective Center Frequency: {effective_freq}")
    logging.debug(f"Input FFT Values Before robust: {full_data}")
   
    # Use effective_freqs instead of computing new frequencies
    initial_bandwidth, signal_power, mask, filtered_freq, filtered_fft_values = robust_compute_bandwidth(full_data, effective_freq, effective_low, effective_high,fs)
    
    logging.debug(f"End of compute: {filtered_fft_values}")
    
    # filtered_fft_values, filtered_freq = advanced_signal_processing_pipeline(
    # effective_freqs,
    # filtered_fft_values,
    # fs,
    # effective_freq,
    # effective_low,
    # effective_high,
    # EARTH_ROTATION_RATE,
    # quality_factor=filter_quality, 
    # step_size=0.05 if effective_freq > 1e9 else 0.1,  # Adjusting based on signal freq
    # filter_length=20 if effective_freq > 1e9 else 10,  # Smaller length for lower frequencies
    # gain_factor=gain_factor * 2,  # Test increasing gain if amplification is too weak
    # wavelet_level=wavelet_level
    # )
    logging.debug(f"End of advanced fft: {filtered_fft_values}")
    logging.debug(f"End of advanced filtered freq: {filtered_freq}")
    logging.debug(f"End of advanced effective freqs: {effective_freqs}")
    logging.debug(f"End of advanced effective freq: {effective_freq}")
    # Use effective_freqs for peak detection
    peaks, troughs = find_emission_absorption_lines(
        effective_freqs,
        filtered_fft_values,
        effective_low,
        effective_high,
        effective_freq
    )
    logging.debug(f"End of peaks: {peaks}")
    logging.debug(f"End of troughs: {troughs}")
    time = time.replace(":","")
    date = date.replace(':', '')
    
    
    effective_freqs, filtered_fft_values = align_frequencies(filtered_freq, filtered_fft_values)
    
    time_domain_data = np.fft.ifft(filtered_fft_values).real
    
    # Generate images using effective_freqs
    if effective_freq < 1000e6:
        identify_gyrosynchrotron_emission(effective_freqs, filtered_fft_values, MAGNETIC_FIELD_STRENGTH, output_dir, date, time)
        if effective_freq < 250e6:
            generate_lofar_image(filtered_fft_values, time_domain_data, effective_freqs, output_dir, date, time, lat, lon, duration)

    # Update all image generation functions to use effective_freqs
    save_waterfall_image(effective_freqs, filtered_fft_values, output_dir, date, time, duration, effective_freq, initial_bandwidth, peaks, troughs, effective_low, effective_high)
    create_intensity_map(effective_freqs, filtered_fft_values, fs, output_dir, date, time, temperature=2.7)
    save_spectrogram_and_signal_strength(effective_freqs, filtered_fft_values, fs, output_dir, date, time, lat, lon, duration, effective_low, effective_high)
    calculate_and_save_psd(effective_freqs, filtered_fft_values, fs, output_dir, date, time, effective_freq, initial_bandwidth, effective_low, effective_high)
    
    # Continue with remaining visualization functions
    simulate_rotational_vibrational_transitions(filtered_fft_values, fs, effective_freq, initial_bandwidth, output_dir, date, time)
    save_spectra(effective_freqs, filtered_fft_values, peaks, troughs, output_dir, date, time)
    save_enhanced_spectra(effective_freqs, filtered_fft_values, peaks, troughs, output_dir, date, time)
    analyze_signal_strength(effective_freqs, filtered_fft_values, output_dir, date, time)
    create_spectral_line_image(filtered_fft_values, filtered_fft_values, peaks, troughs, output_dir, date, time)
    brightness_temp_plot(effective_freqs, filtered_fft_values, peaks, troughs, output_dir, date, time, lat, lon, duration)
    plot_observation_position(output_dir, date, time, lat, lon, duration)
    run_fft_processing(effective_freqs, filtered_fft_values, effective_freq, output_dir, date, time)
    create_position_velocity_diagram(filtered_fft_values, fs, output_dir, date, time)
    
    del full_data

def align_frequencies(filtered_freq, filtered_fft_values):
    # Check if either of the arrays is empty
    if len(filtered_freq) == 0 or len(filtered_fft_values) == 0:
        logging.error("One or both of the input arrays are empty. Cannot align frequencies.")
        return [], []  # Return empty arrays if either is empty
    
    # Check the lengths of the two arrays
    len_freq = len(filtered_freq)
    len_fft = len(filtered_fft_values)

    # If the lengths are equal, return them as is
    if len_freq == len_fft:
        return filtered_freq, filtered_fft_values
    
    # If the length of filtered_freq is greater, interpolate filtered_fft_values
    if len_freq > len_fft:
        logging.debug(f"Interpolating filtered_fft_values to match the length of filtered_freq ({len_freq}).")
        interpolator = interp1d(np.linspace(0, len(filtered_fft_values)-1, len_fft),
                               filtered_fft_values, kind='linear', fill_value="extrapolate")
        filtered_fft_values = interpolator(np.linspace(0, len_freq-1, len_freq))
    
    # If the length of filtered_fft_values is greater, pad filtered_freq to match filtered_fft_values
    elif len_fft > len_freq:
        logging.debug(f"Padding filtered_freq to match the length of filtered_fft_values ({len_fft}).")
        filtered_freq = np.pad(filtered_freq, (0, len_fft - len_freq), 'constant', constant_values=0)
    
    logging.debug(f"Returning aligned arrays with shapes: filtered_freq {filtered_freq.shape}, filtered_fft_values {filtered_fft_values.shape}")
    
    return filtered_freq, filtered_fft_values

def find_emission_absorption_lines(filtered_freq, filtered_fft_values, effective_low, effective_high, effective_freq):
    logging.debug(f"Input shapes: filtered_freq: {filtered_freq.shape}, filtered_fft_values: {filtered_fft_values.shape}")
    logging.debug(f"Effective frequency range: {effective_low} to {effective_high} Hz")
    logging.debug(f"Effective center frequency: {effective_freq} Hz")

    # Calculate power spectrum with overflow handling
    clamped_fft_values = np.clip(np.abs(filtered_fft_values), 0, 1e6)
    power_spectrum = clamped_fft_values ** 2
    logging.debug(f"Power spectrum range: min {power_spectrum.min()}, max {power_spectrum.max()}")

    # Dynamic parameters based on frequency range
    if effective_freq > 1e9:  # Above 1 GHz
        kernel_size = 7
        threshold_factor = 4
        peak_params = {'distance': 80, 'prominence': 0.2, 'width': 5}
    elif effective_freq > 100e6:  # Above 100 MHz
        kernel_size = 5
        threshold_factor = 3
        peak_params = {'distance': 40, 'prominence': 0.1, 'width': 3}
    else:  # Below 100 MHz
        kernel_size = 3
        threshold_factor = 2
        peak_params = {'distance': 20, 'prominence': 0.05, 'width': 2}

    # Apply median filter and handle NaNs or Infs
    smoothed_power = medfilt(power_spectrum, kernel_size=kernel_size)
    smoothed_power = np.nan_to_num(smoothed_power, nan=0.0, posinf=0.0, neginf=0.0)
    logging.debug(f"Smoothed power range: min {smoothed_power.min()}, max {smoothed_power.max()}")

    # Calculate dynamic threshold
    mean_power = np.mean(smoothed_power)
    std_power = np.std(smoothed_power)
    threshold = mean_power + threshold_factor * std_power
    logging.debug(f"Threshold set to: {threshold}")

    try:
        # Peak detection for emission lines
        peak_params['height'] = threshold
        peak_indices, _ = find_peaks(smoothed_power, **peak_params)
        
        # Trough detection for absorption lines, with different approach for positive-only data
        trough_params = peak_params.copy()
        trough_indices, _ = find_peaks(-smoothed_power, prominence=peak_params['prominence'], width=peak_params['width'])

        # Filter peaks and troughs by effective frequency range
        peaks = filtered_freq[peak_indices]
        troughs = filtered_freq[trough_indices]

        # Ensure values fall within the specified frequency range
        peaks = peaks[(peaks >= effective_low) & (peaks <= effective_high)]
        troughs = troughs[(troughs >= effective_low) & (troughs <= effective_high)]

        logging.debug(f"Number of peaks found: {len(peaks)}")
        logging.debug(f"Number of troughs found: {len(troughs)}")

        # Check if no peaks or troughs were found and log a warning
        if len(peaks) == 0:
            logging.warning("No peaks detected in the given frequency range.")
        if len(troughs) == 0:
            logging.warning("No troughs detected in the given frequency range.")

    except Exception as e:
        logging.error(f"Error in find_peaks: {str(e)}", exc_info=True)
        return [], []

    return peaks, troughs


def is_file_processed(filename):
    processed_file = 'processed.dat'
    if not os.path.exists(processed_file):
        return False
    
    with open(processed_file, 'r') as f:
        processed_files = f.read().splitlines()
    
    return filename in processed_files

def mark_file_as_processed(filename):
    processed_file = 'processed.dat'
    with open(processed_file, 'a') as f:
        f.write(f"{filename}\n")


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
    
        if is_file_processed(input_fits):
            logging.info(f"File {input_fits} has already been processed. Skipping.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        check_file()
        date, time = extract_date_from_fits(input_fits)
        start_time = extract_observation_start_time(input_fits)
        
        if start_time:
            process_fft_file(input_fits, output_dir, sample_rate, date, time, lat, lon, duration, center_frequency, low_cutoff, high_cutoff)
            mark_file_as_processed(input_fits)
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
