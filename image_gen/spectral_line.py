import logging
import os
import numpy as np
import matplotlib.pyplot as plt

def amplify_signal_with_peaks_troughs(signal, peaks, troughs, amplification_factor=2):
    amplified_signal = signal.copy()
    valid_peaks = peaks[peaks < signal.shape[0]]
    valid_troughs = troughs[troughs < signal.shape[0]]
    amplified_signal[valid_peaks] *= amplification_factor
    amplified_signal[valid_troughs] /= amplification_factor
    return amplified_signal

def calculate_brightness_temperature(fft_values_scaled, freq):
    c = 299792458  # Speed of light in m/s
    k_B = 1.380649e-23  # Boltzmann constant in J/K
   
    logging.debug(f"Input FFT values range: {np.min(fft_values_scaled)} to {np.max(fft_values_scaled)}")
    logging.debug(f"Frequency range: {np.min(freq)} to {np.max(freq)} Hz")

    with np.errstate(divide='ignore', invalid='ignore'):
        wavelength = np.where(freq != 0, c / freq, 0)
        brightness_temp = np.where(freq != 0, (wavelength**2 * fft_values_scaled) / (2 * k_B), 0)
   
    logging.debug(f"Calculated wavelength range: {np.min(wavelength)} to {np.max(wavelength)} m")
    logging.debug(f"Raw brightness temperature range: {np.min(brightness_temp)} to {np.max(brightness_temp)} K")

    brightness_temp = np.nan_to_num(brightness_temp, nan=0.0, posinf=0.0, neginf=0.0)
   
    logging.debug(f"Final brightness temperature range: {np.min(brightness_temp)} to {np.max(brightness_temp)} K")

    return brightness_temp

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

def brightness_temp_plot(freq, fft_values, peaks, troughs, output_dir, date, time, lat, lon, duration_hours):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Reshape fft_values into a 2D array
        num_time_steps = int(np.sqrt(len(fft_values)))
        num_freq_steps = len(fft_values) // num_time_steps
        fft_values_2d = fft_values[:num_time_steps*num_freq_steps].reshape(num_time_steps, num_freq_steps)
        
        # Calculate brightness temperature
        amplified_fft_values = amplify_signal_with_peaks_troughs(fft_values_2d, peaks, troughs, amplification_factor=10000)
        brightness_temperature_2d = calculate_brightness_temperature(np.abs(amplified_fft_values), freq[:num_freq_steps])
        brightness_temperature_2d = np.log10(brightness_temperature_2d + 1)  # Add 1 to avoid log(0)
        
        # Create the 2D temperature plot
        fig, ax = plt.subplots(figsize=(12, 8))
        freq_min, freq_max = min(freq), max(freq)
        im = ax.imshow(brightness_temperature_2d, aspect='auto',
                    extent=[freq_min, freq_max, 0, duration_hours],
                    origin='lower', cmap='viridis', interpolation='nearest')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Time (hours)')
        ax.set_title(f'H1 Brightness Temperature: {date} {time}\nLat: {lat:.2f}°, Lon: {lon:.2f}°\nDuration: {duration_hours:.2f} hours')
        
        cbar = fig.colorbar(im, ax=ax, label='Log Brightness Temperature (K)')
        
        output_path = os.path.join(output_dir, f'brightness_temperature_2d_{date}_{time}.png')
        plt.savefig(output_path)
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in brightness_temp_plot: {str(e)}")
        logging.error(f"FFT values shape: {fft_values.shape}")
        logging.error(f"Frequency shape: {freq.shape}")

def plot_observation_position(output_dir, date, time, lat, lon, duration_hours):
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot the Earth
        earth = plt.Circle((0, 0), 1, color='lightblue', fill=False)
        ax.add_artist(earth)
        
        # Convert lat/lon to x/y coordinates on the unit circle
        x = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
        y = np.sin(np.radians(lat))
        
        # Plot the observation position
        sc = ax.scatter(x, y, c='red', s=200)
        
        ax.annotate(f'({lat:.2f}°, {lon:.2f}°)', (x, y), xytext=(5, 5),
                    textcoords='offset points', ha='left', va='bottom')
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Observation Position: {date} {time}\nDuration: {duration_hours:.2f} hours')
        
        output_path_position = os.path.join(output_dir, f'observation_position_{date}_{time}.png')
        plt.savefig(output_path_position)
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in plot_observation_position: {str(e)}")

def create_spectral_line_profile(signal_data, sampling_rate, center_frequency, bandwidth, output_dir, date, time):
    freq_range = np.fft.fftfreq(len(signal_data), d=1/sampling_rate)
    intensity = np.abs(signal_data)  # Assuming signal_data is already in FFT form

    plt.figure(figsize=(10, 6))
    plt.plot((freq_range + center_frequency) / 1e6, intensity, label='Spectral Line Profile')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Intensity')
    plt.title('Spectral Line Profile')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'spectral_line_profile_{date}_{time}.png.png'))
    
    plt.show()
