import logging
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def annotate_peak(args):
    peak, value, harmonic = args
    plt.annotate(f'n={harmonic}', (peak/1e6, value), xytext=(5,5), textcoords='offset points')

def process_in_chunks(freq, fft_values, chunk_size=1000000):
    logging.debug(f"Processing data in chunks of size {chunk_size}")
    for i in range(0, len(freq), chunk_size):
        logging.debug(f"Yielding chunk {i//chunk_size + 1}")
        yield freq[i:i+chunk_size], fft_values[i:i+chunk_size]

def process_chunk(chunk_data, expected_frequencies):
    logging.debug("Processing chunk")
    freq_chunk, fft_chunk = chunk_data
    peak_indices, _ = find_peaks(fft_chunk, height=0)
    peak_freqs = freq_chunk[peak_indices]
    detected_peaks = list(filter(None, map(lambda peak: detect_peak(peak, expected_frequencies), peak_freqs)))
    logging.debug(f"Detected {len(detected_peaks)} peaks in chunk")
    return detected_peaks

def get_peak_value(peak, freq, fft_values):
    logging.debug(f"Getting peak value for frequency {peak}")
    with threading.Lock():
        return fft_values[np.where(freq == peak)[0][0]]

def detect_peak(peak_freq, expected_frequencies):
    for expected_freq in expected_frequencies:
        if np.isclose(peak_freq, expected_freq, atol=1e6):
            logging.debug(f"Detected peak at frequency {peak_freq}")
            return peak_freq
    return None

def calculate_harmonic_numbers(detected_peaks, gyrofrequency):
    logging.info("Calculating harmonic numbers")
    def process_batch(batch):
        return [round(peak / gyrofrequency) for peak in batch]

    batch_size = 10000
    logging.debug(f"Processing harmonic numbers in batches of size {batch_size}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        batches = (detected_peaks[i:i+batch_size] for i in range(0, len(detected_peaks), batch_size))
        results = executor.map(process_batch, batches)
   
    harmonics = [harmonic for batch in results for harmonic in batch]
    logging.debug(f"Calculated {len(harmonics)} harmonic numbers")
    return harmonics

def plot_chunk(chunk_peaks, chunk_values):
    plt.scatter(np.array(chunk_peaks) / 1e6, chunk_values, color='red')

def process_frequency_range(freq, fft_values, center_freq, output_dir, date, time):
    logging.info(f"Processing frequency range centered at {center_freq}")
    magnetic_field_strength = 100
    identify_gyrosynchrotron_emission(freq, fft_values, magnetic_field_strength, output_dir, date, time)

def identify_gyrosynchrotron_emission(freq, fft_values, magnetic_field_strength, output_dir, date_str, time_str):
    logging.info(f"Starting gyrosynchrotron emission identification for {date_str} {time_str}")
   
    # Constants
    electron_charge = 1.602e-19
    electron_mass = 9.109e-31
    speed_of_light = 3.0e8

    # Calculate gyrofrequency
    gyrofrequency = electron_charge * magnetic_field_strength / (2 * np.pi * electron_mass * speed_of_light)
    logging.debug(f"Calculated gyrofrequency: {gyrofrequency} Hz")

    # Expected gyroharmonic frequencies
    expected_frequencies = [n * gyrofrequency for n in range(1, 10)]
    logging.debug(f"Expected gyroharmonic frequencies: {expected_frequencies}")

    detected_peaks = []
    logging.info("Starting parallel processing of chunks")
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        chunk_results = executor.map(process_chunk, process_in_chunks(freq, fft_values),
                                     [expected_frequencies]*len(list(process_in_chunks(freq, fft_values))))
        for chunk_peaks in chunk_results:
            detected_peaks.extend(chunk_peaks)

    logging.info(f"Detected {len(detected_peaks)} gyrosynchrotron emission peaks")

    # Calculate harmonic numbers
    harmonic_numbers = calculate_harmonic_numbers(detected_peaks, gyrofrequency)

    # Plot the results
    logging.info("Generating plot")
    plt.figure(figsize=(14, 7))
    for freq_chunk, fft_chunk in process_in_chunks(freq, fft_values):
        plt.plot(freq_chunk / 1e6, fft_chunk, label='FFT Magnitude Spectrum', color='black')

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        valid_peaks = list(executor.map(lambda peak: peak if peak in freq else None, detected_peaks))
        valid_peaks = [peak for peak in valid_peaks if peak is not None]
        valid_peak_values = list(executor.map(lambda peak: get_peak_value(peak, freq, fft_values), valid_peaks))
   
    chunk_size = 1024  # Adjust based on your data size and system capabilities
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(0, len(valid_peaks), chunk_size):
            executor.submit(plot_chunk, 
                            valid_peaks[i:i+chunk_size], 
                            valid_peak_values[i:i+chunk_size])

    plt.legend(['Detected Peaks'])
   
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(annotate_peak, zip(valid_peaks, valid_peak_values, harmonic_numbers))
        
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude')
    plt.title(f'FFT Magnitude Spectrum with Identified Gyrosynchrotron Emission Peaks\n{date_str} {time_str}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(output_dir, f'gyrosynchrotron_emission_{date_str}_{time_str}.png')
    plt.savefig(output_filename, dpi=300)
    plt.close()
    logging.info(f"Saved gyrosynchrotron emission plot to {output_filename}")