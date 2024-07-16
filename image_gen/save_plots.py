import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def save_plots_to_png(spectrogram, signal_strength, frequencies, times, png_location, date, time, lat, lon, duration_hours):
    with tqdm(total=1, desc='Generating Spectrogram & Signal_Strength:') as pbar:
        try:
            logging.debug(f"Starting plot generation for date: {date}, time: {time}")
            logging.debug(f"Spectrogram shape: {spectrogram.shape}, Signal strength shape: {signal_strength.shape}")
            logging.debug(f"Frequency range: {frequencies[0]} to {frequencies[-1]} Hz")
            logging.debug(f"Time range: {times[0]} to {times[-1]} seconds")

            # Spectrogram plot
            logging.debug(f"Signal strength min: {np.min(signal_strength)}, max: {np.max(signal_strength)}")
            plt.figure(figsize=(12, 8))
            spec_db = 10 * np.log10(spectrogram.T + 1e-10)
            vmin, vmax = np.percentile(spec_db, [1, 99])
            logging.debug(f"Spectrogram dB range: {vmin} to {vmax}")
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            plt.imshow(spec_db, aspect='auto', cmap='viridis', origin='lower',
                       extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
                       norm=norm)
            plt.title(f'Spectrogram\nLat: {lat:.2f}°, Lon: {lon:.2f}°\nDuration: {duration_hours:.2f} hours')
            plt.colorbar(label='Intensity (dB)')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            spectrogram_filename = f'{png_location}/spectrogram_{date}_{time}.png'
            plt.savefig(spectrogram_filename)
            plt.close()
            logging.info(f"Spectrogram saved to {spectrogram_filename}")

            # Signal Strength plot
            if signal_strength.ndim == 1:
                logging.debug("Generating 1D signal strength plot")
                plt.figure(figsize=(12, 6))
                plt.plot(signal_strength)
                plt.title(f'Signal Strength\nLat: {lat:.2f}°, Lon: {lon:.2f}°\nDuration: {duration_hours:.2f} hours')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                signal_strength_filename = f'{png_location}/signal_strength_{date}_{time}.png'
                plt.savefig(signal_strength_filename)
                plt.close()
                logging.info(f"1D Signal Strength plot saved to {signal_strength_filename}")
            elif signal_strength.ndim == 2:
                logging.debug("Generating 2D signal strength plot")
                plt.figure(figsize=(12, 8))
                plt.imshow(signal_strength.T, aspect='auto', cmap='viridis', origin='lower',
                           extent=[0, duration_hours, 0, signal_strength.shape[1]])
                plt.title(f'Signal Strength\nLat: {lat:.2f}°, Lon: {lon:.2f}°\nDuration: {duration_hours:.2f} hours')
                plt.colorbar(label='Amplitude')
                plt.xlabel('Time (hours)')
                plt.ylabel('Sample Index')
                signal_strength_filename = f'{png_location}/signal_strength_{date}_{time}.png'
                plt.savefig(signal_strength_filename)
                plt.close()
                logging.info(f"2D Signal Strength plot saved to {signal_strength_filename}")

            pbar.update(1)
            logging.debug("Plot generation completed successfully")
        except Exception as e:
            logging.error(f"An error occurred while saving plots: {str(e)}")

def analyze_signal_strength(signal_strength, output_dir, date, time):

    signal_strength_magnitude = np.abs(signal_strength)
    min_val = np.min(signal_strength_magnitude)
    max_val = np.max(signal_strength_magnitude)
    mean_val = np.mean(signal_strength_magnitude)
    std_val = np.std(signal_strength_magnitude)

    analysis_results = (
        f"Signal strength - min: {min_val}, max: {max_val}, mean: {mean_val}, std: {std_val}\n"
    )

    # Plot the distribution of the signal strength values
    plt.figure(figsize=(12, 6))
    plt.hist(signal_strength_magnitude, bins=50)
    plt.title('Signal Strength Distribution')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    distribution_plot_path = os.path.join(output_dir, f'signal_strength_distribution_{date}_{time}.png')
    plt.savefig(distribution_plot_path)
    plt.close()

    # Plot the signal strength values to check for spikes
    plt.figure(figsize=(12, 6))
    plt.plot(signal_strength_magnitude)
    plt.title('Signal Strength Over Sample Index')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    strength_plot_path = os.path.join(output_dir, f'signal_strength_{date}_{time}.png')
    plt.savefig(strength_plot_path)
    plt.close()
    
    print(f"Distribution plot saved to {distribution_plot_path}")
    print(f"Signal strength plot saved to {strength_plot_path}")