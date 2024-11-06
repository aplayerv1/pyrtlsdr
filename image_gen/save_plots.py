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

def spectrogram_plot(frequency, fft_values, sampling_rate, png_location, date, time, lat, lon, duration_hours, lowcutoff, highcutoff):
    logging.info(f"Starting spectrogram plot generation for {date} {time}")
    
    with tqdm(total=1, desc='Generating Spectrogram:') as pbar:
        try:
            fft_values = np.asarray(fft_values)
            
            if fft_values.ndim != 1:
                logging.error("fft_values should be a 1D array for this function")
                raise ValueError("fft_values should be a 1D array")

            # Inside spectrogram_plot function
            min_length = min(len(frequency), len(fft_values))
            mask = (frequency[:min_length] >= lowcutoff) & (frequency[:min_length] <= highcutoff)
            filtered_frequency = frequency[:min_length][mask]
            filtered_fft_values = fft_values[:min_length][mask]


            print(f"Filtered frequency length: {len(filtered_frequency)}")
            print(f"Filtered fft_values length: {len(filtered_fft_values)}")
            print(f"Lowcutoff: {lowcutoff}, Highcutoff: {highcutoff}")
            print(f"Frequency range: {np.min(frequency)} to {np.max(frequency)}")
           
            n_fft = 1024  # or another appropriate value
            freq_range = np.linspace(lowcutoff, highcutoff, n_fft)

            plt.figure(figsize=(12, 8))
            plt.specgram(filtered_fft_values, NFFT=n_fft, Fs=sampling_rate, Fc=np.mean([lowcutoff, highcutoff]), noverlap=512, cmap='viridis')

            plt.colorbar(label='Power')
            plt.xlabel('Time (Minutes)')
            plt.ylabel('Frequency (Hz)')
            plt.ylim(lowcutoff, highcutoff)
            plt.xlim(0, duration_hours/60)
            plt.title(f'Spectrogram {date} {time}\nLat: {lat}, Lon: {lon}\nFrequency Range: {lowcutoff/1e6:.2f} MHz - {highcutoff/1e6:.2f} MHz')

            spectrogram_filename = f'spectrogram_{date}_{time}.png'
            spectrogram_path = os.path.join(png_location, spectrogram_filename)
            plt.savefig(spectrogram_path, dpi=300, format='png', bbox_inches='tight')
            
            if os.path.exists(spectrogram_path) and os.path.getsize(spectrogram_path) > 0:
                logging.info(f"Spectrogram saved successfully to: {spectrogram_path}")
            else:
                logging.error(f"Failed to save spectrogram or file is empty: {spectrogram_path}")

            plt.close()
            pbar.update(1)
            logging.info(f"Spectrogram generation completed for {date} {time}")

        except Exception as e:
            logging.error(f"Error in spectrogram plot generation: {e}")


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
