import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
import concurrent.futures
import logging
import multiprocessing as mp
from cupyx import jit

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_MEMORY = 1 * 1024 * 1024 * 1024  # 1GB in bytes

def get_available_memory():
    meminfo = cp.cuda.runtime.memGetInfo()
    return meminfo[0]  # Returns available memory in bytes

def allocate_with_limit(shape, dtype):
    required_memory = np.prod(shape) * np.dtype(dtype).itemsize
    if required_memory > get_available_memory() or required_memory > MAX_MEMORY:
        return np.empty(shape, dtype)  # Allocate on CPU if exceeding limit
    return cp.empty(shape, dtype)

def annotate_peak(args):
    peak, value, harmonic = args
    plt.annotate(f'n={harmonic}', (peak / 1e6, value), xytext=(5, 5), textcoords='offset points')

def detect_peaks_gpu(fft_chunk_gpu):
    threshold = cp.percentile(cp.abs(fft_chunk_gpu), 95)
    peak_indices = cp.where(cp.abs(fft_chunk_gpu) > threshold)[0]
    return peak_indices

def process_chunk(chunk_data, expected_frequencies_gpu):
    logging.debug("Processing chunk")
    freq_chunk, fft_chunk = chunk_data
    freq_chunk_gpu = allocate_with_limit(freq_chunk.shape, freq_chunk.dtype)
    fft_chunk_gpu = allocate_with_limit(fft_chunk.shape, fft_chunk.dtype)
    
    if isinstance(freq_chunk_gpu, cp.ndarray):
        freq_chunk_gpu.set(freq_chunk)
        fft_chunk_gpu.set(fft_chunk)
    else:
        freq_chunk_gpu = freq_chunk
        fft_chunk_gpu = fft_chunk

    peak_indices = detect_peaks_gpu(fft_chunk_gpu)
    
    if peak_indices.size == 0:
        logging.debug("No peaks detected")
        return []

    peak_freqs = freq_chunk_gpu[peak_indices]
    
    detected_peaks = cp.array([peak for peak in peak_freqs if cp.isin(peak, expected_frequencies_gpu)])
    detected_peaks = detected_peaks.get()
    
    if detected_peaks.size == 0:
        logging.debug("No detected peaks match expected frequencies")
    
    logging.debug(f"Detected {len(detected_peaks)} peaks in chunk")

    if isinstance(freq_chunk_gpu, cp.ndarray):
        del freq_chunk_gpu
        del fft_chunk_gpu
        cp.get_default_memory_pool().free_all_blocks()

    return detected_peaks

def calculate_harmonic_numbers_gpu(detected_peaks, gyrofrequency):
    logging.info("Calculating harmonic numbers")

    if detected_peaks.size == 0:
        logging.warning("No detected peaks to process for harmonic numbers")
        return np.array([])

    def process_batch(batch):
        return np.round(batch / gyrofrequency)

    batch_size = 10000
    logging.debug(f"Processing harmonic numbers in batches of size {batch_size}")

    harmonics = np.concatenate([
        process_batch(detected_peaks[i:i + batch_size]) 
        for i in range(0, len(detected_peaks), batch_size)
        if detected_peaks[i:i + batch_size].size > 0
    ])
    
    logging.debug(f"Calculated {len(harmonics)} harmonic numbers")
    return harmonics

def plot_chunk(chunk_peaks, chunk_values):
    plt.scatter(np.array(chunk_peaks) / 1e6, chunk_values, color='red')

@jit.rawkernel()
def process_fft_kernel(fft_values, output):
    i = jit.grid(1)
    if i < output.size:
        output[i] = cp.abs(fft_values[i])

def identify_gyrosynchrotron_emission(freq, fft_values, magnetic_field_strength, output_dir, date_str, time_str):
    logging.info(f"Starting gyrosynchrotron emission identification for {date_str} {time_str}")
    time_str = time_str.replace(":","")
    electron_charge = 1.602e-19
    electron_mass = 9.109e-31
    speed_of_light = 3.0e8

    gyrofrequency = electron_charge * magnetic_field_strength / (2 * np.pi * electron_mass * speed_of_light)
    logging.debug(f"Calculated gyrofrequency: {gyrofrequency} Hz")

    expected_frequencies = np.array([n * gyrofrequency for n in range(1, 10)])
    expected_frequencies_gpu = cp.array(expected_frequencies)
    logging.debug(f"Expected gyroharmonic frequencies: {expected_frequencies}")

    detected_peaks = []
    logging.info("Starting parallel processing of chunks")

    chunk_size = 1000000  # Adjust based on available memory
    chunk_data = [(freq[i:i+chunk_size], fft_values[i:i+chunk_size]) for i in range(0, len(freq), chunk_size)]
   
    with mp.get_context("spawn").Pool(processes=os.cpu_count()) as pool:
        chunk_results = pool.starmap(process_chunk, [(chunk, expected_frequencies_gpu) for chunk in chunk_data])
        for chunk_peaks in chunk_results:
            detected_peaks.extend(chunk_peaks)

    logging.info(f"Detected {len(detected_peaks)} gyrosynchrotron emission peaks")

    if len(detected_peaks) == 0:
        logging.warning("No peaks detected, skipping harmonic number calculation.")
        logging.info("No peaks detected, skipping plotting.")
        return

    detected_peaks = np.array(detected_peaks)
    harmonic_numbers = calculate_harmonic_numbers_gpu(detected_peaks, gyrofrequency)

    plt.figure(figsize=(14, 7))
    for freq_chunk, fft_chunk in chunk_data:
        fft_magnitude = np.abs(fft_chunk)
        fft_magnitude_gpu = cp.array(fft_magnitude, dtype=cp.float32)  # Use float32 instead of float64
        output_gpu = cp.zeros_like(fft_magnitude_gpu)
        
        threads_per_block = 256
        blocks_per_grid = (fft_magnitude_gpu.size + threads_per_block - 1) // threads_per_block
        
        process_fft_kernel((blocks_per_grid,), (threads_per_block,), (fft_magnitude_gpu, output_gpu))
        plt.plot(freq_chunk / 1e6, output_gpu.get(), label='FFT Magnitude Spectrum', color='black')

    valid_peaks = [peak for peak in detected_peaks if peak in freq]
    valid_peak_values = [np.abs(fft_values[np.where(freq == peak)[0][0]]) for peak in valid_peaks]

    chunk_size = 1024
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(0, len(valid_peaks), chunk_size):
            executor.submit(plot_chunk, valid_peaks[i:i + chunk_size], valid_peak_values[i:i + chunk_size])

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
