import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor

def process_fft_and_plot(center_frequency, freq, fft_values, output_dir, date, time):
    # Compute brightness (specific intensity) from FFT values
    brightness = np.abs(fft_values)
    
    # Apply logarithmic scaling to brightness
    brightness_log = np.log10(brightness + 1e-10)  # Adding a small constant to avoid log(0)
    
    # Compute flux density by integrating brightness over frequency range
    flux_density = np.trapz(brightness, freq)
    
    # Plot Brightness (Specific Intensity) with logarithmic scale
    plt.figure(figsize=(10, 6))
    plt.plot(freq[:len(freq)//2], brightness_log[:len(brightness_log)//2])  # Plot only positive frequencies
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Log10 of Brightness (Specific Intensity)')
    plt.title(f'Brightness Spectrum for Center Frequency: {center_frequency/1e6} MHz')
    plt.grid(True)
    plt.tight_layout()
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'brightness_spectrum_{center_frequency/1e6}MHz_{date}_{time}.png'))
    
    plt.show()
    
    # Print Flux Density
    print(f'Flux Density for Center Frequency {center_frequency/1e6} MHz: {flux_density:.2f}')

def fft_processing_task(center_frequency, filtered_freq, filtered_fft_values, output_dir, date, time):

    # Call the function to process FFT and plot
    process_fft_and_plot(center_frequency, filtered_freq, filtered_fft_values, output_dir, date, time)

def run_fft_processing(filtered_freq, filtered_fft_values, center_frequency, output_dir, date, time):

    with ThreadPoolExecutor() as executor:
        # Submit the FFT processing task
        future = executor.submit(fft_processing_task, center_frequency, filtered_freq, filtered_fft_values, output_dir, date, time)
        future.result()  # Wait for the task to complete
