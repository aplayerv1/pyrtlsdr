import logging
import numpy as np
import matplotlib.pyplot as plt
import os

def save_waterfall_image(signal_data, output_dir, date_str, time_str, duration_hours, center_freq, bandwidth, peaks, troughs, low_cutoff, high_cutoff):
    try:
        signal_data = np.array(signal_data)
        num_samples = len(signal_data)
        
        num_times = int(np.sqrt(num_samples))
        num_frequencies = num_samples // num_times
        
        reshaped_data = signal_data[:num_times * num_frequencies].reshape((num_times, num_frequencies))
        power_db = 10 * np.log10(np.abs(reshaped_data)**2 + 1e-10)

        os.makedirs(output_dir, exist_ok=True)
        image_filename = os.path.join(output_dir, f'waterfall_{date_str}_{time_str}.png')
        
        time_axis = np.linspace(0, duration_hours, num_times)
        freq_axis = np.linspace(center_freq - bandwidth/2, center_freq + bandwidth/2, num_frequencies) / 1e6

        plt.figure(figsize=(12, 8))
        plt.pcolormesh(time_axis, freq_axis, power_db.T, shading='auto', cmap='viridis')
        plt.colorbar(label='Power (dB)')
        
        # Apply low and high cutoffs
        plt.ylim(low_cutoff/1e6, high_cutoff/1e6)
        
        # Plot peaks and troughs
        for peak in peaks:
            plt.axhline(y=freq_axis[peak], color='r', linestyle='--', alpha=0.5)
        for trough in troughs:
            plt.axhline(y=freq_axis[trough], color='b', linestyle='--', alpha=0.5)

        plt.title(f'Waterfall Display \n{date_str} {time_str} (Duration: {duration_hours:.2f} hours)')
        plt.xlabel('Time (hours)')
        plt.ylabel('Frequency (MHz)')
        plt.tight_layout()

        plt.savefig(image_filename, dpi=300)
        plt.close()

        logging.info(f"Waterfall display saved successfully: {image_filename}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")