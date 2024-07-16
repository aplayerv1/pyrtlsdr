import logging
import os
from matplotlib import pyplot as plt
import numpy as np

def simulate_rotational_vibrational_transitions(signal_data, sampling_rate, center_frequency, bandwidth, output_dir, date, time):
    try:
        # Ensure signal_data is a NumPy array
        signal_data = np.array(signal_data)
        num_samples = len(signal_data)
        
        if num_samples == 0:
            raise ValueError("signal_data is empty.")
        
        # Ensure bandwidth is a scalar
        if not np.isscalar(bandwidth):
            raise ValueError("bandwidth should be a scalar value.")
        
        logging.debug(f"Signal data length: {num_samples}, Sampling rate: {sampling_rate}, Center frequency: {center_frequency}, Bandwidth: {bandwidth}")

        # Calculate frequency range
        freq_range = np.fft.fftfreq(num_samples, d=1/sampling_rate)
        intensity = np.abs(signal_data)  # Assuming signal_data is already in FFT form

        # Plot the intensity of rotational/vibrational transitions
        plt.figure(figsize=(10, 6))
        plt.plot((freq_range + center_frequency) / 1e6, intensity, label='Rotational/Vibrational Transitions')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Intensity')
        plt.title('Rotational/Vibrational Transitions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Adjust x-axis limits based on center_frequency and bandwidth
        plt.xlim((center_frequency - bandwidth/2) / 1e6, (center_frequency + bandwidth/2) / 1e6)

        # Save the plot if output_dir is specified
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_filename = os.path.join(output_dir, f'rotational_vibrational_transitions_{date}_{time}.png')
            plt.savefig(image_filename, dpi=300)
            logging.info(f"Plot saved to: {image_filename}")
        
        plt.show()
    except Exception as e:
        logging.error(f"An error occurred: {e}")