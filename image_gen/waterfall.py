import logging
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os

def enhance_fft_values(fft_values):
    return np.log10(np.abs(fft_values) + 1)

def preprocess_fft_values(fft_values, kernel_size=3):
    denoised_fft_values = signal.medfilt(np.abs(fft_values), kernel_size=kernel_size)
    return enhance_fft_values(denoised_fft_values)

def save_waterfall_image(freq, fft_values, output_dir, date_str, time_str, duration_hours, center_freq, bandwidth, peaks, troughs, low_cutoff, high_cutoff):
    logging.info(f"Starting waterfall image generation for {date_str} {time_str}")
    try:
        logging.debug(f"Initial frequency range: {freq.min()} to {freq.max()} Hz")
        logging.debug(f"Initial FFT values range: {fft_values.min()} to {fft_values.max()}")
        logging.debug(f"Duration hours before: {duration_hours}")
        duration_hours = duration_hours / 60
        logging.debug(f"Duration hours after: {duration_hours}")
        # Preprocess FFT values
        preprocessed_fft_values = preprocess_fft_values(fft_values)

        # Separate positive and negative values
        positive_mask = preprocessed_fft_values > 0
        negative_mask = preprocessed_fft_values < 0
        
        # Apply logarithmic scaling to positive values
        preprocessed_fft_values[positive_mask] = np.log10(preprocessed_fft_values[positive_mask] + 1)
        
        # Apply logarithmic scaling to negative values, keeping them negative
        preprocessed_fft_values[negative_mask] = -np.log10(-preprocessed_fft_values[negative_mask] + 1)

        sorted_indices = np.argsort(freq)
        freq = freq[sorted_indices]
        preprocessed_fft_values = preprocessed_fft_values[sorted_indices]
        
        unique_indices = np.unique(freq, return_index=True)[1]
        freq = freq[unique_indices]
        preprocessed_fft_values = preprocessed_fft_values[unique_indices]

        logging.debug(f"Frequency range after sorting and removing duplicates: {freq.min()} to {freq.max()} Hz")
        logging.debug(f"Number of unique frequency points: {len(freq)}")

        num_samples = len(freq)
        num_times = int(np.sqrt(num_samples))
        num_frequencies = num_samples // num_times

        logging.debug(f"Reshaping data to {num_times} time points and {num_frequencies} frequency points")

        reshaped_data = preprocessed_fft_values[:num_times * num_frequencies].reshape((num_times, num_frequencies))
        power_db = 10 * np.log10(np.abs(reshaped_data)**2 + 1e-10)

        logging.debug(f"Power range in dB: {power_db.min()} to {power_db.max()} dB")

        os.makedirs(output_dir, exist_ok=True)
        image_filename = os.path.join(output_dir, f'waterfall_{date_str}_{time_str}.png')

        time_axis = np.linspace(0, duration_hours, num_times)
        freq_axis = freq[:num_frequencies] / 1e6  # Convert to MHz

        plt.figure(figsize=(12, 8))
        plt.pcolormesh(time_axis, freq_axis, power_db.T, shading='auto', cmap='coolwarm')
        plt.colorbar(label='Power (dB)')

        plt.ylim(freq_axis.min(), freq_axis.max())
        
        valid_peaks = [int(p) for p in peaks if 0 <= p < len(freq_axis)]
        valid_troughs = [int(t) for t in troughs if 0 <= t < len(freq_axis)]

        logging.debug(f"Number of valid peaks: {len(valid_peaks)}")
        logging.debug(f"Number of valid troughs: {len(valid_troughs)}")

        for peak in valid_peaks:
            plt.axhline(y=freq_axis[peak], color='r', linestyle='--', alpha=0.5)
        for trough in valid_troughs:
            plt.axhline(y=freq_axis[trough], color='b', linestyle='--', alpha=0.5)

        plt.title(f'Waterfall Display \n{date_str} {time_str} (Duration: {duration_hours:.2f} minutes)')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Frequency (MHz)')
        plt.tight_layout()

        plt.savefig(image_filename, dpi=300)
        plt.close()

        logging.info(f"Waterfall display saved successfully: {image_filename}")
    except Exception as e:
        logging.error(f"An error occurred while saving waterfall image: {e}")
