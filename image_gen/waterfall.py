import logging
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os

def save_waterfall_image(freq, fft_values, output_dir, date_str, time_str, duration_hours, center_freq, bandwidth, peaks, troughs, low_cutoff, high_cutoff):
    logging.info(f"Starting waterfall image generation for {date_str} {time_str}")
    fft_values = np.abs(fft_values)
    try:
        logging.debug(f"Input Frequency Data: {freq}")
        logging.debug(f"Input FFT Values: {fft_values}")

        logging.debug(f"Initial frequency range: {freq.min()} to {freq.max()} Hz")
        logging.debug(f"Initial FFT values range: {fft_values.min()} to {fft_values.max()}")
        logging.debug(f"Duration hours before: {duration_hours}")
        
        duration_minutes = duration_hours * 60
        logging.debug(f"Duration minutes after: {duration_minutes}")

        preprocessed_fft_values = np.abs(fft_values).astype(np.float64)
        logging.debug(f"Preprocessed FFT values range: {preprocessed_fft_values.min()} to {preprocessed_fft_values.max()}")

        with np.errstate(divide='ignore', invalid='ignore'):
            preprocessed_fft_values = np.log10(preprocessed_fft_values + 1e-10)

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

        # Ensure there are no zero or negative values before taking log10
        reshaped_data = np.clip(np.abs(reshaped_data), 1e-10, None)
        power_db = 10 * np.log10(reshaped_data**2)
        
        # Check if the min and max values of power_db are equal
        if power_db.max() != power_db.min():
            power_db_normalized = (power_db - power_db.min()) / (power_db.max() - power_db.min())
        else:
            # Handle the case where all values are the same
            power_db_normalized = np.zeros_like(power_db)  # Or use a default value, depending on your needs

        # Now apply the enhancement as normal
        power_db_enhanced = np.power(power_db_normalized, 0.5)

        logging.debug(f"Power range in dB: {power_db.min()} to {power_db.max()} dB")

        os.makedirs(output_dir, exist_ok=True)
        image_filename = os.path.join(output_dir, f'waterfall_{date_str}_{time_str}.png')

        time_axis = np.linspace(0, duration_minutes, num_times)
        freq_axis = freq[:num_frequencies] / 1e6

        plt.figure(figsize=(12, 8))
        plt.pcolormesh(time_axis, freq_axis, power_db_enhanced.T, 
                      shading='auto', 
                      cmap='viridis',
                      vmin=np.percentile(power_db_enhanced, 5),
                      vmax=np.percentile(power_db_enhanced, 95))
        plt.colorbar(label='Power (dB)')
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.ylim(freq_axis.min(), freq_axis.max())
        
        valid_peaks = [int(p) for p in peaks if 0 <= p < len(freq_axis)]
        valid_troughs = [int(t) for t in troughs if 0 <= t < len(freq_axis)]

        logging.debug(f"Number of valid peaks: {len(valid_peaks)}")
        logging.debug(f"Number of valid troughs: {len(valid_troughs)}")

        for peak in valid_peaks:
            plt.axhline(y=freq_axis[peak], color='r', linestyle='--', alpha=0.5)
        for trough in valid_troughs:
            plt.axhline(y=freq_axis[trough], color='b', linestyle='--', alpha=0.5)

        plt.title(f'Waterfall Display \n{date_str} {time_str} (Duration: {duration_minutes:.2f} seconds)')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Frequency (MHz)')
        plt.tight_layout()

        plt.savefig(image_filename, dpi=300)
        plt.close()

        logging.info(f"Waterfall display saved successfully: {image_filename}")
    except Exception as e:
        logging.error(f"An error occurred while saving waterfall image: {e}")
