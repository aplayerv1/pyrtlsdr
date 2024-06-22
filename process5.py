import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
from scipy.signal import butter, sosfiltfilt, lfilter, find_peaks, medfilt
from astropy.io import fits
from scipy import ndimage
from skimage import exposure
import gc
import librosa
import librosa.display
from tqdm import tqdm
import pywt
from datetime import datetime
import io
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging 
gc.enable()



# Define the speed of light in meters per second
speed_of_light = 299792458  # meters per second
delta_lambda = 1e-2  # Change in wavelength in 
lambda_0 = 0.211      # Rest wavelength in meters (500 nanometers)
EARTH_ROTATION_RATE = 15  # degrees per hour
tolerance = 1e6

# Sampling frequency in Hz
fs = 2.4e6
# Low band LO frequency in MHz
notch_freq = 9750
# Notch width in MHz
notch_width = 30


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
    signal = np.asarray(signal, dtype=np.float64)
    t = np.tan(np.pi * notch_width / fs)
    beta = (1 - t) / (1 + t)
    gamma = -np.cos(2 * np.pi * notch_freq / fs)
    b = [1, gamma * (1 + beta), beta]
    a = [1, gamma * (1 - beta), -beta]
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def extract_date_from_filename(filename):
    # Extract date and time from filename using regular expression
    pattern = r'(\d{8})_(\d{6})'  # Assuming the format is YYYYMMDD_HHMMSS
    match = re.search(pattern, filename)
    header = fits.Header()
    if match:
        date = match.group(1)
        time = match.group(2)
        with fits.open(filename, ignore_missing_simple=True) as hdul:
            header['DATE'] = date
            hdu = fits.PrimaryHDU(header=header)
            hdul.append(hdu)
        return date, time
    else:
        return None, None

def extract_observation_start_time(fits_filename):
    with fits.open(fits_filename, ignore_missing_simple=True) as hdul:
        header = hdul[0].header
        start_time_str = header['DATE']
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d')
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

def calculate_velocity(delta_lambda, lambda_0, is_redshift=True):
    sign = 1 if is_redshift else -1
    velocity = sign * (delta_lambda / lambda_0) * speed_of_light
    return velocity

def apply_redshift(data, delta_lambda, lambda_0):
    velocity = calculate_velocity(delta_lambda, lambda_0, is_redshift=True)
    if velocity is not None:
        if isinstance(data, np.ndarray):
            shifted_data = data * np.sqrt((1 - velocity / speed_of_light) / (1 + velocity / speed_of_light))
            return shifted_data
        else:
            print("Error: data is not a NumPy array")
            return None
    else:
        return data

def apply_blueshift(data):
    shifted_data = data
    return shifted_data

def image_reconstruction(signal_data, shift_type='none', delta_lambda=None, lambda_0=None, apply_fft=False, inverse=False):
    if shift_type == 'redshift':
        shifted_signal = apply_redshift(signal_data, delta_lambda, lambda_0)
    elif shift_type == 'blueshift':
        shifted_signal = apply_blueshift(signal_data)
    elif shift_type == 'both':
        redshifted_signal = apply_redshift(signal_data, delta_lambda, lambda_0)
        shifted_signal = apply_blueshift(redshifted_signal)
    else:
        shifted_signal = signal_data

    if apply_fft:
        if len(shifted_signal.shape) == 1:
            shifted_signal_2d = np.reshape(shifted_signal, (1, len(shifted_signal)))
        else:
            shifted_signal_2d = shifted_signal

        fft_result = np.fft.fft2(shifted_signal_2d)
        reconstructed_image = np.fft.ifft2(fft_result).real

        if inverse:
            reconstructed_signal_2d = np.fft.ifft2(fft_result).real
            reconstructed_signal = np.ravel(reconstructed_signal_2d)
            return reconstructed_signal
        else:
            return reconstructed_image
    else:
        return shifted_signal

def save_visualized_image2(reconstructed_image, output_dir, date, time, shift_type='none'):
    with tqdm(total=1, desc='Generating Reconstructed Image:') as pbar:
        os.makedirs(output_dir, exist_ok=True)
        image_filename = os.path.join(output_dir, f'reconstructed_image_{date}_{time}_{shift_type}.png')

        if len(reconstructed_image.shape) == 1:
            total_elements = len(reconstructed_image)
            sqrt_elements = int(np.sqrt(total_elements))
            for i in range(sqrt_elements, 0, -1):
                if total_elements % i == 0:
                    width = i
                    height = total_elements // i
                    break
            reconstructed_image = np.reshape(reconstructed_image, (height, width))

        plt.figure(figsize=(10, 8))
        plt.imshow(reconstructed_image, cmap='gray')
        plt.colorbar(label='Intensity')
        plt.title(f'Reconstructed Image ({shift_type.capitalize()} Shift)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(image_filename, bbox_inches='tight')
        plt.close()
        pbar.update(1)

def compute_spectrogram(samples, fs):
    nfft = 256
    noverlap = 128
    spectrogram, frequencies, times, _ = plt.specgram(samples, NFFT=nfft, Fs=fs, noverlap=noverlap)
    spectrogram = np.where(spectrogram == 0, 1e-10, spectrogram)
    return spectrogram, frequencies, times

def find_emission_absorption_lines(freq, fft_values, height_threshold=0.1, distance=20):
    magnitude = np.abs(fft_values)
    peaks, _ = find_peaks(magnitude, height=height_threshold, distance=distance)
    troughs, _ = find_peaks(-magnitude, height=height_threshold, distance=distance)
    return peaks, troughs

def downsample_data(data, max_columns=2**23):
    if data.shape[1] > max_columns:
        factor = data.shape[1] // max_columns
        return data[:, ::factor]
    return data

def save_spectra(freq, fft_values, peaks, troughs, output_dir, date, time):
  with tqdm(total=1, desc='Generating Spectra:') as pbar:
  
    os.makedirs(output_dir, exist_ok=True)
    magnitude = np.abs(fft_values)
    plt.figure(figsize=(14, 7))
    plt.plot(freq, magnitude, label='Spectrum', color='black')
    if peaks.size > 0:
        plt.plot(freq[peaks], magnitude[peaks], 'ro', label='Emission Lines')
    if troughs.size > 0:
        plt.plot(freq[troughs], magnitude[troughs], 'bo', label='Absorption Lines')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Emission and Absorption Spectra')
    plt.legend(loc="upper right")
    plt.xlim([0, max(freq)])
    output_path = os.path.join(output_dir, f'spectra_{date}_{time}.png')
    plt.savefig(output_path)
    plt.close()
    pbar.update(1)

def save_spectra2(freq, fft_values, peaks, troughs, output_dir, date, time):
    with tqdm(total=1, desc='Generating Spectra:') as pbar:
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply preprocessing to the FFT values
        enhanced_fft_values = preprocess_fft_values(fft_values)
        
        plt.figure(figsize=(14, 7))
        plt.plot(freq, enhanced_fft_values, label='Spectrum', color='black')
        
        if peaks.size > 0:
            plt.plot(freq[peaks], enhanced_fft_values[peaks], 'ro', label='Emission Lines')
        if troughs.size > 0:
            plt.plot(freq[troughs], enhanced_fft_values[troughs], 'bo', label='Absorption Lines')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Emission and Absorption Spectra')
        plt.legend(loc="upper right")
        plt.xlim([0, max(freq)])
        output_path = os.path.join(output_dir, f'spectra2_{date}_{time}.png')
        plt.savefig(output_path)
        plt.close()
        pbar.update(1)



def save_plots_to_png(spectrogram, signal_strength, frequencies, times, png_location, date, time):
    with tqdm(total=1, desc='Generating Spectrogram & Signal_Strength:') as pbar:
        try:
            spectrogram = downsample_data(spectrogram)
            plt.figure()
            plt.imshow(10 * np.log10(spectrogram.T), aspect='auto', cmap='viridis', origin='lower',
                       extent=[times[0], times[-1], frequencies[0], frequencies[-1]])
            plt.title('Spectrogram')
            plt.colorbar(label='Intensity (dB)')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.savefig(f'{png_location}/spectrogram_{date}_{time}.png')
            plt.close()

            if signal_strength.ndim == 1:
                plt.figure()
                plt.plot(signal_strength)
                plt.title('Signal Strength')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.savefig(f'{png_location}/signal_strength_{date}_{time}.png')
                plt.close()
            elif signal_strength.ndim == 2:
                signal_strength = downsample_data(signal_strength)
                plt.figure()
                plt.imshow(signal_strength.T, aspect='auto', cmap='viridis', origin='lower')
                plt.title('Signal Strength')
                plt.colorbar(label='Amplitude')
                plt.xlabel('Time')
                plt.ylabel('Sample Index')
                plt.savefig(f'{png_location}/signal_strength_{date}_{time}.png')
                plt.close()
            pbar.update(1)
        except Exception as e:
            print(f"An error occurred while saving plots: {str(e)}")


def preprocess_signal(signal_data, start_time, end_time):
    time_vector = np.linspace(0, len(signal_data) / fs, num=len(signal_data))
    mask = (time_vector >= start_time) & (time_vector <= end_time)
    processed_signal = signal_data[mask]
    filtered_signal = remove_lnb_effect(processed_signal, fs, notch_freq, notch_width)
    return filtered_signal

def load_fits_file(filename):
    with fits.open(filename) as hdul:
        data = hdul[0].data
    return data

def load_fft_file(filename):
    data = np.loadtxt(filename, delimiter=',')
    freq = data[:, 0]
    fft_values = data[:, 1]
    return freq, fft_values

def compute_mfcc(signal_data, sample_rate=44100, num_mfcc=13):
    mfcc = librosa.feature.mfcc(y=signal_data, sr=sample_rate, n_mfcc=num_mfcc)
    return mfcc

def save_mfcc_plot(mfcc, png_location, date, time):
    with tqdm(total=1, desc='Generating MFCC Plot:') as pbar:
        try:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc, x_axis='time')
            plt.colorbar()
            plt.title('MFCC')
            plt.tight_layout()
            plt.savefig(f'{png_location}/mfcc_{date}_{time}.png')
            plt.close()
            pbar.update(1)
        except Exception as e:
            print(f"An error occurred while saving MFCC plot: {str(e)}")

def save_spectrogram_and_signal_strength(signal_data, sampling_rate, output_dir, date, time):
    spectrogram, frequencies, times = compute_spectrogram(signal_data, sampling_rate)
    save_plots_to_png(spectrogram, signal_data, frequencies, times, output_dir, date, time)

def save_cqt_spectrogram(signal_data, sampling_rate, output_dir, date, time):
    with tqdm(total=1, desc='Generating CQT Plot:') as pbar:
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute the Constant-Q Transform
        cqt = librosa.cqt(signal_data, sr=sampling_rate)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        # Plot and save the Constant-Q spectrogram
        plt.figure(figsize=(10, 8))
        librosa.display.specshow(cqt_db, sr=sampling_rate, x_axis='time', y_axis='cqt_hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(f'{output_dir}/cqt_spectrogram_{date}_{time}.png')
        plt.close()
        pbar.update(1)

def save_vqt_spectrogram(signal_data, sampling_rate, output_dir, date, time):
    with tqdm(total=1, desc='Generating VQT Plot:') as pbar:
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute the Variable-Q Transform
        vqt = librosa.vqt(signal_data, sr=sampling_rate)
        vqt_db = librosa.amplitude_to_db(np.abs(vqt), ref=np.max)
        
        # Plot and save the Variable-Q spectrogram
        plt.figure(figsize=(10, 8))
        librosa.display.specshow(vqt_db, sr=sampling_rate, x_axis='time', y_axis='cqt_hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Variable-Q Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(f'{output_dir}/vqt_spectrogram_{date}_{time}.png')
        plt.close()
        pbar.update(1)

def enhance_fft_values(fft_values):
    # Apply logarithmic scaling to enhance FFT values
    enhanced_fft_values = np.log10(np.abs(fft_values) + 1)
    return enhanced_fft_values

def preprocess_fft_values(fft_values, kernel_size=3):
    # Denoise FFT values using median filter
    denoised_fft_values = medfilt(np.abs(fft_values), kernel_size=kernel_size)
    
    # Apply enhancement to denoised FFT values
    enhanced_fft_values = enhance_fft_values(denoised_fft_values)
    
    return enhanced_fft_values

def denoise_signal(data, wavelet='db1', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(new_coeffs, wavelet)

def amplify_signal(data, factor=10):
    return data * factor

def bandpass_filter(signal_data, sampling_rate, center_frequency, bandwidth=400e3, filter_order=5, tolerance=1e6):
    nyquist_rate = sampling_rate / 2.0
    if abs(center_frequency - args.center_frequency) < tolerance:
        low_cutoff = ((center_frequency - 10e6) - bandwidth / 2) / nyquist_rate
        high_cutoff = ((center_frequency - 10e6) + bandwidth / 2) / nyquist_rate
    else:
        low_cutoff = (center_frequency - bandwidth / 2) / nyquist_rate
        high_cutoff = (center_frequency + bandwidth / 2) / nyquist_rate
    # Convert cutoff frequencies to the range [0, 1]
    high_cutoff = high_cutoff / nyquist_rate
    low_cutoff = low_cutoff / nyquist_rate
    # Print normalized cutoff frequencies
    print("Low Cutoff (Normalized):", low_cutoff)
    print("High Cutoff (Normalized):", high_cutoff)
    print("Center Frequency:", center_frequency)
    sos = butter(filter_order, [low_cutoff, high_cutoff], btype='band', output='sos')
    filtered_signal = sosfiltfilt(sos, signal_data)
    return filtered_signal

# def compute_fft(signal_data, sampling_rate):
#     # Remove DC component
#     signal_data = signal_data - np.mean(signal_data)
    
#     n = len(signal_data)
#     freq = np.fft.fftfreq(n, d=1/sampling_rate)
#     fft_values = np.fft.fft(signal_data)
    
#     # Return only the positive half of the frequencies
#     half_n = n // 2
#     return freq[:half_n], np.abs(fft_values[:half_n])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_fft_file(input_fits, output_dir, fs, date, time, chunk_size=1024):
    print(fs)
    try:
        temp_output_file = os.path.join(output_dir, "temp_processed_data.npy")
        logging.debug(f"Temporary output file path: {temp_output_file}")

        # Create or clear the temporary output file
        with open(temp_output_file, 'wb') as f:
            np.save(f, np.array([]))  # Save an empty array to initialize the file

        def process_and_save_chunk(chunk, output_file):
            logging.debug(f"Processing chunk with length: {len(chunk)}")
            if len(chunk) == 0:
                logging.warning("Received an empty chunk for processing.")
                return chunk

            # Process the chunk (example: amplification)
            amplified_chunk = chunk * 2  # Example processing
            logging.debug(f"Amplified chunk: {amplified_chunk}")

            try:
                with open(output_file, 'ab') as f:
                    np.save(f, amplified_chunk)
            except Exception as e:
                logging.error(f"Failed to save chunk: {str(e)}")

            return amplified_chunk

        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []

            with fits.open(input_fits) as hdul:
                data = hdul[0].data
                logging.info(f"Data shape: {data.shape}")

                if not isinstance(data.shape, tuple):
                    logging.error(f"Unexpected type for data.shape: {type(data.shape)}")
                    raise ValueError("data.shape is not a tuple")

                for dim in data.shape:
                    if not isinstance(dim, int):
                        logging.error(f"Unexpected type for data.shape element: {type(dim)}")
                        raise ValueError("data.shape elements are not integers")

                if data.ndim == 1:
                    logging.debug("Data is 1-dimensional")
                    total_rows = data.shape[0]
                    logging.debug(f"Total rows: {total_rows} (type: {type(total_rows)})")

                    # Skip the first row
                    for start_row in range(1, total_rows, chunk_size):
                        logging.debug(f"start_row: {start_row} (type: {type(start_row)})")
                        end_row = min(start_row + chunk_size, total_rows)
                        logging.debug(f"end_row: {end_row} (type: {type(end_row)})")
                        chunk = data[start_row:end_row]
                        logging.debug(f"Processing chunk from row {start_row} to {end_row}")
                        futures.append(executor.submit(process_and_save_chunk, chunk, temp_output_file))

                elif data.ndim == 2:
                    logging.debug("Data is 2-dimensional")
                    total_rows = data.shape[0]
                    logging.debug(f"Total rows: {total_rows} (type: {type(total_rows)})")

                    # Skip the first row
                    for start_row in range(1, total_rows, chunk_size):
                        logging.debug(f"start_row: {start_row} (type: {type(start_row)})")
                        end_row = min(start_row + chunk_size, total_rows)
                        logging.debug(f"end_row: {end_row} (type: {type(end_row)})")
                        chunk = data[start_row:end_row, :]
                        logging.debug(f"Processing chunk from row {start_row} to {end_row}")
                        futures.append(executor.submit(process_and_save_chunk, chunk, temp_output_file))

                else:
                    logging.error("Unsupported data dimensionality")
                    raise ValueError("Unsupported data dimensionality")

            logging.debug("Waiting for all futures to complete")
            results = [future.result() for future in futures]

            logging.debug("All futures completed")
            # Use results for further processing or logging
            for i, result in enumerate(results):
                if result is not None:
                    logging.info(f"Chunk {i} processed with shape {result.shape}")
                else:
                    logging.warning(f"Chunk {i} resulted in None")

        amplified_signal = concatenate_chunks(temp_output_file)

        if amplified_signal.size > 0:
            logging.debug(f"Amplified signal size: {amplified_signal.size}")
            freq, fft_values = compute_fft(amplified_signal, fs)
            peaks, troughs = find_emission_absorption_lines(freq, fft_values)
            create_spectral_line_image(freq, fft_values, peaks, troughs, output_dir, date, time)
            create_spectral_line_image2(freq, fft_values, peaks, troughs, output_dir, date, time)
            save_spectra2(freq, fft_values, peaks, troughs, output_dir, date, time)
            save_spectrogram_and_signal_strength(amplified_signal, fs, output_dir, date, time)

            # Add LOFAR imaging step
            generate_lofar_image(amplified_signal, output_dir, date, time)
            
        else:
            logging.warning("No valid data processed.")
    except Exception as e:
        logging.error(f"An error occurred while processing {input_fits}: {str(e)}")
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)
            logging.debug(f"Temporary file {temp_output_file} deleted.")

def generate_lofar_image(signal_data, output_dir, date, time):
    with tqdm(total=1, desc='Generating LOFAR Image:') as pbar:
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Determine the size of the image
            image_size = int(np.sqrt(len(signal_data)))
            if image_size * image_size < len(signal_data):
                image_size += 1
            
            # Create the LOFAR image data array
            lofar_image_data = np.zeros((image_size, image_size))
            lofar_image_data.flat[:len(signal_data)] = signal_data[:image_size * image_size]
            lofar_image_data = lofar_image_data[:image_size, :image_size]

            # Generate the plot
            plt.figure(figsize=(10, 8))
            plt.imshow(lofar_image_data, cmap='inferno', aspect='auto')
            plt.colorbar(label='Intensity')

            # Add contour levels
            levels = np.linspace(np.min(lofar_image_data), np.max(lofar_image_data), 10)
            plt.contour(lofar_image_data, levels=levels, colors='white', linewidths=0.5)

            # Add title and labels
            plt.title(f'LOFAR Image ({date} {time})')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')

            # Save the image
            output_file = os.path.join(output_dir, f'lofar_image_{date}_{time}.png')
            plt.savefig(output_file)
            plt.close()

            pbar.update(1)
        except Exception as e:
            logging.error(f"An error occurred while generating LOFAR image: {str(e)}")


def concatenate_chunks(temp_output_file):
    chunks = []
    with open(temp_output_file, 'rb') as f:
        while True:
            try:
                chunk = np.load(f, allow_pickle=False)
                if chunk.size > 0:  # Ignore empty arrays
                    logging.info(f"Loaded chunk with shape: {chunk.shape}")
                    chunks.append(chunk)
            except ValueError:
                break
    return np.concatenate(chunks) if len(chunks) > 0 else np.array([])

def compute_fft(signal, fs):
    n = len(signal)
    if n == 0:
        logging.error("Signal length is zero.")
        return np.array([]), np.array([])

    freq = np.fft.fftfreq(n, d=1/fs)
    fft_values = np.fft.fft(signal)
    return freq, fft_values

def find_emission_absorption_lines(freq, fft_values):
    magnitude = np.abs(fft_values)
    peaks, _ = find_peaks(magnitude)
    troughs, _ = find_peaks(-magnitude)
    return peaks, troughs

def create_spectral_line_image(freq, fft_values, peaks, troughs, output_dir, date, time):
    magnitude = np.abs(fft_values)
    plt.figure(figsize=(10, 6))
    plt.plot(freq, magnitude, label='Spectrum')
    plt.plot(freq[peaks], magnitude[peaks], 'ro', label='Emission Lines')
    plt.plot(freq[troughs], magnitude[troughs], 'bo', label='Absorption Lines')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Spectral Line Image ({date} {time})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'spectral_line_{date}_{time}.png'))
    plt.close()

def create_spectral_line_image2(freq, fft_values, peaks, troughs, output_dir, date, time):
    os.makedirs(output_dir, exist_ok=True)
    heatmap, xedges, yedges = np.histogram2d(freq[peaks], np.abs(fft_values)[peaks], bins=(100, 100), range=[[0, max(freq)], [0, max(np.abs(fft_values))]])

    plt.figure(figsize=(10, 5))
    plt.imshow(heatmap.T, extent=[0, max(freq), 0, max(np.abs(fft_values))], origin='lower', cmap='inferno')
    plt.colorbar(label='Intensity')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('All Sky Map at a Wavelength of 21 cm')
    
    output_path = os.path.join(output_dir, f'all_sky_map_{date}_{time}.png')
    plt.savefig(output_path)
    plt.close()



def main(args):
    try:
        i = 0
        input_fits = args.input_fits
        fft_file = args.fft_file
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        sample_rate = 2.4e6

        signal_data = load_fits_file(input_fits)
        
        date, time = extract_date_from_filename(input_fits)
        start_time = extract_observation_start_time(input_fits)
        rotation_angle = calculate_rotation_angle(start_time)
        
        signal_data = apply_rotation(signal_data, rotation_angle)
        
        delta_lambda = 1e-2
        lambda_0 = 0.211
        
        reconstructed_image_none = image_reconstruction(signal_data, shift_type='none')
        reconstructed_image_redshift = image_reconstruction(signal_data, shift_type='redshift', delta_lambda=delta_lambda, lambda_0=lambda_0)
        reconstructed_image_blueshift = image_reconstruction(signal_data, shift_type='blueshift')
        reconstructed_image_both = image_reconstruction(signal_data, shift_type='both', delta_lambda=delta_lambda, lambda_0=lambda_0)
        
        save_visualized_image2(reconstructed_image_none, output_dir, date, time, shift_type='none')
        save_visualized_image2(reconstructed_image_redshift, output_dir, date, time, shift_type='redshift')
        save_visualized_image2(reconstructed_image_blueshift, output_dir, date, time, shift_type='blueshift')
        save_visualized_image2(reconstructed_image_both, output_dir, date, time, shift_type='both')

        # # Generate and save CQT and VQT spectrograms
        signal_data_float = signal_data.astype(np.float32)
        save_cqt_spectrogram(signal_data_float, sample_rate, output_dir, date, time)
        save_vqt_spectrogram(signal_data_float, sample_rate, output_dir, date, time)


        # Load FFT file without unpacking
        data = np.loadtxt(fft_file, delimiter=',', skiprows=1)


        # Extract frequency and FFT values from the loaded data
        freq = data[:, 0]  # Assuming frequency is in the first column
        fft_values = data[:, 1]  # Assuming FFT values are in the second column
        fft_imagine = data[:, 2] # Assuming FFT imaginary values are in the third column


        fft_both = fft_values + 1j * fft_imagine

        processed_fft_values = preprocess_fft_values(fft_both)

        # Process signal data
        filtered_signal = preprocess_signal(signal_data, start_time=0, end_time=len(signal_data) / sample_rate)


        # # # Save spectrogram and signal strength
        # save_spectrogram_and_signal_strength(filtered_signal, sample_rate, output_dir, date, time)


        # Find peaks and troughs
        peaks, troughs = find_emission_absorption_lines(freq, processed_fft_values)


        # Save spectra and create spectral line image
        save_spectra(freq, processed_fft_values, peaks, troughs, output_dir, date, time)


        # # Compute MFCC and save plot
        mfcc = compute_mfcc(filtered_signal, sample_rate)
        save_mfcc_plot(mfcc, output_dir, date, time)

        process_fft_file(args.input_fits, args.output_dir, sample_rate, date, time)

        print("Signal processing and image generation completed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process FITS files and generate images.')
    parser.add_argument('-i', '--input_fits', type=str, required=True, help='Path to the input FITS file.')
    parser.add_argument('-f', '--fft_file', type=str, required=True, help='Path to the input FFT file.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('-s', '--start_time', type=float, default=0, help='Start time for signal preprocessing.')
    parser.add_argument('-e', '--end_time', type=float, default=None, help='End time for signal preprocessing.')
    parser.add_argument('--fs', type=float, default=2.4e6, help='Sampling frequency in Hz.')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Chunk size for processing data.')
    parser.add_argument('--tolerance', type=float, default=1e6, help='Tolerance for bandpass filter frequency check.')
    parser.add_argument('--center-frequency', type=float, default=1420.4e6, help='Center Frequency.')

    args = parser.parse_args()
    
    main(args)