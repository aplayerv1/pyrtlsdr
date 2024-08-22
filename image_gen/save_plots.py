import logging
import tempfile
from matplotlib.colors import LogNorm
from scipy import signal
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import concurrent.futures
import logging
import tempfile
import os
import numpy as np

def create_spectrogram(freq, fft_values, fs, chunk_size=1000, max_workers=4):
    logging.info(f"Starting spectrogram creation with {len(freq)} frequency points and {fs} Hz sampling rate")

    freq = np.array(freq)
    duration = len(fft_values) / fs
    times = np.linspace(0, duration, len(fft_values))

    total_size = len(freq) * len(times) * 4  # 4 bytes for float32
    if total_size > 2**31 - 1:  # Max file size for 32-bit systems
        logging.warning("Spectrogram size exceeds maximum file size. Reducing resolution.")
        downscale_factor = int(np.ceil(np.sqrt(total_size / (2**31 - 1))))
        freq = freq[::downscale_factor]
        times = times[::downscale_factor]

    spectrogram_path = os.path.join(tempfile.gettempdir(), 'spectrogram.dat')
    spectrogram = np.memmap(spectrogram_path, dtype='float32', mode='w+', shape=(len(freq), len(times)))

    logging.info(f"Created memory-mapped array for spectrogram at {spectrogram_path}")

    total_chunks = len(fft_values) // chunk_size + (1 if len(fft_values) % chunk_size else 0)
    
    def process_chunk(chunk_index):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(fft_values))
        chunk = fft_values[start:end]
        chunk_spectrogram = np.abs(chunk.reshape(-1, 1))
        chunk_spectrogram_db = 10 * np.log10(chunk_spectrogram + 1e-10)
        spectrogram[:, start:end] = chunk_spectrogram_db
        logging.debug(f"Processed chunk {chunk_index + 1} of {total_chunks}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, i) for i in range(total_chunks)]
        concurrent.futures.wait(futures)

    logging.info(f"Spectrogram creation completed. Shape: {spectrogram.shape}")
    return spectrogram, freq, times

def spectrogram_plot(frequency, fft_values, sampling_rate, png_location, date, time, lat, lon, duration_hours):
    logging.info(f"Starting basic spectrogram plot generation for {date} {time}")
    
    with tqdm(total=1, desc='Generating Basic Spectrogram:') as pbar:
        try:
            # Ensure fft_values is a numpy array
            fft_values = np.asarray(fft_values)
            
            if fft_values.ndim != 1:
                logging.error("fft_values should be a 1D array for this function")
                raise ValueError("fft_values should be a 1D array")

            # Create basic spectrogram
            plt.figure(figsize=(12, 8))
            plt.specgram(fft_values, Fs=sampling_rate, scale='linear', cmap='viridis')

            plt.colorbar(label='Power')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Basic Spectrogram {date} {time}\nLat: {lat}, Lon: {lon}')

            # Save the plot to a file
            spectrogram_filename = f'basic_spectrogram_{date}_{time}.png'
            spectrogram_path = os.path.join(png_location, spectrogram_filename)
            plt.savefig(spectrogram_path, dpi=300, format='png', bbox_inches='tight')
            
            if os.path.exists(spectrogram_path) and os.path.getsize(spectrogram_path) > 0:
                logging.info(f"Basic spectrogram saved successfully to: {spectrogram_path}")
            else:
                logging.error(f"Failed to save basic spectrogram or file is empty: {spectrogram_path}")

            plt.close()
            pbar.update(1)
            logging.info(f"Basic spectrogram generation completed for {date} {time}")

        except Exception as e:
            logging.error(f"Error in basic spectrogram plot generation: {e}")

def analyze_signal_strength(freq, fft_values, output_dir, date, time):
    magnitude = np.abs(fft_values)
    min_val, max_val, mean_val, std_val = np.min(magnitude), np.max(magnitude), np.mean(magnitude), np.std(magnitude)

    analysis_results = f"Signal strength - min: {min_val}, max: {max_val}, mean: {mean_val}, std: {std_val}\n"

    # Plot the distribution of the signal strength values
    plt.figure(figsize=(12, 6))
    plt.hist(magnitude, bins=50)
    plt.title('Signal Strength Distribution')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    distribution_plot_path = os.path.join(output_dir, f'signal_strength_distribution_{date}_{time}.png')
    plt.savefig(distribution_plot_path)
    plt.close()

    # Plot the signal strength values over time
    time_axis = np.arange(len(magnitude)) / len(magnitude)  # Normalize time to [0, 1]
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, magnitude)
    plt.title('Signal Strength Over Time')
    plt.xlabel('Time (normalized)')
    plt.ylabel('Magnitude')
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    strength_plot_path = os.path.join(output_dir, f'signal_strength_{date}_{time}.png')
    plt.savefig(strength_plot_path)
    plt.close()

    print(f"Distribution plot saved to {distribution_plot_path}")
    print(f"Signal strength plot saved to {strength_plot_path}")

    return analysis_results
