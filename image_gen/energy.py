import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def process_fft_and_plot(center_frequency, freq, fft_values, output_dir, date, time):
    try:
        logging.info(f"Starting FFT processing for center frequency {center_frequency / 1e6} MHz")

        # Compute brightness (specific intensity) from FFT values
        brightness = np.abs(fft_values)
        logging.debug("Computed brightness from FFT values")

        # Apply logarithmic scaling to brightness
        brightness_log = np.log10(brightness + 1e-10)  # Adding a small constant to avoid log(0)
        logging.debug("Applied logarithmic scaling to brightness")

        # Compute flux density by integrating brightness over frequency range
        flux_density = np.trapz(brightness, freq)
        logging.debug("Computed flux density by integrating brightness")

        # Plot Brightness (Specific Intensity) with logarithmic scale
        plt.figure(figsize=(10, 6))
        plt.plot(freq[:len(freq)//2], brightness_log[:len(brightness_log)//2])  # Plot only positive frequencies
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Log10 of Brightness (Specific Intensity)')
        plt.title(f'Brightness Spectrum for Center Frequency: {center_frequency / 1e6} MHz')
        plt.grid(True)
        plt.tight_layout()

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logging.debug(f"Created output directory: {output_dir}")
            
            output_file = os.path.join(output_dir, f'brightness_spectrum_{center_frequency / 1e6}MHz_{date}_{time}.png')
            plt.savefig(output_file)
            logging.info(f"Saved plot to {output_file}")

        plt.show()
        logging.info(f'Flux Density for Center Frequency {center_frequency / 1e6} MHz: {flux_density:.2f}')

    except Exception as e:
        logging.error(f"An error occurred while processing FFT: {e}")
        raise

def fft_processing_task(center_frequency, filtered_freq, filtered_fft_values, output_dir, date, time):
    logging.info("Starting FFT processing task")
    # Call the function to process FFT and plot
    process_fft_and_plot(center_frequency, filtered_freq, filtered_fft_values, output_dir, date, time)

def run_fft_processing(filtered_freq, filtered_fft_values, center_frequency, output_dir, date, time):
    logging.info("Running FFT processing")
    with ThreadPoolExecutor() as executor:
        # Submit the FFT processing task
        future = executor.submit(fft_processing_task, center_frequency, filtered_freq, filtered_fft_values, output_dir, date, time)
        try:
            future.result()  # Wait for the task to complete
            logging.info("FFT processing task completed successfully")
        except Exception as e:
            logging.error(f"An error occurred while running FFT processing: {e}")
            raise
