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
    logging.info(f"Starting spectrogram plot generation for {date} {time}")
    with tqdm(total=1, desc='Generating Spectrogram & Signal_Strength:') as pbar:
        try:
            # Apply Hann window
            window = signal.hann(len(fft_values))
            fft_values = fft_values * window

            # Create spectrogram
            f, t, Sxx = signal.spectrogram(fft_values, fs=sampling_rate, nperseg=1024, noverlap=512)
            
            # Apply logarithmic scaling
            Sxx = np.log10(Sxx + 1e-10)

            # Normalize the spectrogram data
            Sxx = np.clip(Sxx, 1e-10, None)  # Clip values to avoid log(0) issues
            Sxx_norm = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())

            plt.figure(figsize=(12, 8))

            # Use LogNorm for color scaling with adjusted vmin and vmax
            im = plt.pcolormesh(t, f, Sxx_norm, cmap='viridis', norm=LogNorm(vmin=1e-10, vmax=1))

            plt.colorbar(im, label='Normalized Power (dB)')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Spectrogram {date} {time}\nLat: {lat}, Lon: {lon}')

            # Add contour lines
            plt.contour(t, f, Sxx, colors='w', linewidths=0.5, alpha=0.3)

            # Save the plot to a file
            spectrogram_filename = f'spectrogram_{date}_{time}.png'
            spectrogram_path = os.path.join(png_location, spectrogram_filename)
            plt.savefig(spectrogram_path, dpi=300, format='png', bbox_inches='tight')
            plt.close()

            logging.info(f"Spectrogram saved to: {spectrogram_path}")
            pbar.update(1)
            logging.info(f"Spectrogram generation completed for {date} {time}")
        except Exception as e:
            logging.error(f"Error in plot generation: {e}")
            logging.exception("Detailed traceback:")

def analyze_signal_strength(freq, fft_values, output_dir, date, time):
    magnitude = np.abs(fft_values)
    min_val = np.min(magnitude)
    max_val = np.max(magnitude)
    mean_val = np.mean(magnitude)
    std_val = np.std(magnitude)

    analysis_results = (f"Signal strength - min: {min_val}, max: {max_val}, mean: {mean_val}, std: {std_val}\n")

    # Plot the distribution of the signal strength values
    plt.figure(figsize=(12, 6))
    plt.hist(magnitude, bins=50)
    plt.title('Signal Strength Distribution')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    distribution_plot_path = os.path.join(output_dir, f'signal_strength_distribution_{date}_{time}.png')
    plt.savefig(distribution_plot_path)
    plt.close()

    # Plot the signal strength values over frequency
    plt.figure(figsize=(12, 6))
    plt.plot(freq, magnitude)
    plt.title('Signal Strength Over Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    strength_plot_path = os.path.join(output_dir, f'signal_strength_{date}_{time}.png')
    plt.savefig(strength_plot_path)
    plt.close()
    
    print(f"Distribution plot saved to {distribution_plot_path}")
    print(f"Signal strength plot saved to {strength_plot_path}")
