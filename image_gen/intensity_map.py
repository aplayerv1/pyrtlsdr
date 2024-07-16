import logging
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, butter, filtfilt


def is_frequency_domain(data):
    # Apply inverse FFT
    time_domain_data = np.fft.ifft(data)
    # Check if the real part of the inverse FFT looks like a time-domain signal
    real_part = np.real(time_domain_data)
    # Simple heuristic: check if the real part of the IFFT has significant variations
    mean_real_part = np.mean(real_part)
    std_real_part = np.std(real_part)
    if std_real_part > 0.1 * mean_real_part:
        return True
    else:
        return False

def create_intensity_map(signal_data, sampling_rate, output_dir, date, time):
    logging.debug("Starting create_intensity_map function")
    logging.debug(f"Signal data shape: {signal_data.shape}")

    # Check if the data is in the frequency domain
    if not is_frequency_domain(signal_data):
        signal_data = np.fft.fft(signal_data)

    # Calculate the intensity map
    intensity_map = np.abs(signal_data)

    # Create time array
    time_array = np.arange(len(signal_data)) / sampling_rate

    # Calculate the frequency bins
    freq_bins = np.fft.fftshift(np.fft.fftfreq(len(signal_data), d=1/sampling_rate))

    # Reshape intensity_map to 2D (time x frequency)
    intensity_2d = intensity_map.reshape(-1, len(freq_bins))

    # Plot the intensity map
    plt.figure(figsize=(12, 8))
    plt.imshow(np.log1p(intensity_2d), aspect='auto', cmap='viridis',
               extent=[freq_bins.min(), freq_bins.max(), time_array.min(), time_array.max()])
    plt.title(f"Time-Frequency Intensity Map: {date} {time}")
    plt.colorbar(label='Log Intensity')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time (s)')

    # Save the intensity map
    output_path = os.path.join(output_dir, f"time_freq_intensity_map_{date}_{time}.png")
    plt.savefig(output_path)
    plt.close()
    logging.debug(f"Time-Frequency Intensity map saved to {output_path}")