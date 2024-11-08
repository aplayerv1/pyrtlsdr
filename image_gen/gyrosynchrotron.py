import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
import concurrent.futures
import logging
import multiprocessing as mp
from cupyx import jit
from scipy import signal

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

def detect_peaks_gpu(fft_chunk_gpu, peak_threshold=85):
    # For 2-4 MHz range, use more sensitive thresholds
    if peak_threshold > 80:
        peak_threshold = 70
    
    threshold = cp.percentile(cp.abs(fft_chunk_gpu), peak_threshold)
    # Add minimum prominence check for low frequencies
    peak_indices = cp.where(cp.abs(fft_chunk_gpu) > threshold * 0.7)[0]
    return peak_indices

def process_chunk(chunk_data, expected_frequencies_gpu, peak_threshold):
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

    peak_indices = detect_peaks_gpu(fft_chunk_gpu, peak_threshold)
    
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

@jit.rawkernel()
def process_fft_kernel(fft_values, output):
    i = jit.grid(1)
    if i < output.size:
        output[i] = cp.abs(fft_values[i])

def plot_gyrosynchrotron_spectrum(chunk_data, detected_peaks, harmonic_numbers, freq, fft_values, threads_per_block, max_freq):
    # Plot FFT magnitude spectrum with frequency-specific scaling
    for freq_chunk, fft_chunk in chunk_data:
        fft_magnitude = np.abs(fft_chunk)
        fft_magnitude_gpu = cp.array(fft_magnitude, dtype=cp.float32)
        output_gpu = cp.zeros_like(fft_magnitude_gpu)
        
        blocks_per_grid = (fft_magnitude_gpu.size + threads_per_block - 1) // threads_per_block
        process_fft_kernel((blocks_per_grid,), (threads_per_block,), (fft_magnitude_gpu, output_gpu))
        
        plt.plot(freq_chunk / 1e6, output_gpu.get(), color='black', alpha=0.5)

    # Plot detected peaks with frequency-specific markers
    valid_peaks = [peak for peak in detected_peaks if peak in freq]
    valid_peak_values = [np.abs(fft_values[np.where(freq == peak)[0][0]]) for peak in valid_peaks]
    
    plt.scatter(np.array(valid_peaks) / 1e6, valid_peak_values, color='red', marker='x' if max_freq > 1e9 else 'o')
    
    # Annotate harmonics
    for peak, value, harmonic in zip(valid_peaks, valid_peak_values, harmonic_numbers):
        plt.annotate(f'n={int(harmonic)}', (peak / 1e6, value), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8 if max_freq > 1e9 else 10)

def identify_gyrosynchrotron_emission(effective_freqs, filtered_fft_values, MAGNETIC_FIELD_STRENGTH, output_dir, date, time):
    folding_factor = int(5.55e9 / 20e6)  
    actual_freqs = effective_freqs + (folding_factor * 20e6)
    
    electron_cyclotron_freq = 2.8e6 * MAGNETIC_FIELD_STRENGTH
    harmonic_numbers = actual_freqs / electron_cyclotron_freq
    power_law_index = 3.0
    
    emission_intensity = np.abs(filtered_fft_values)
    spectral_index = np.gradient(np.log(emission_intensity)) / np.gradient(np.log(actual_freqs))
    peak_frequency = actual_freqs[np.argmax(emission_intensity)]
    
    # Use power law and harmonics for analysis
    theoretical_intensity = (harmonic_numbers ** power_law_index) * np.exp(-harmonic_numbers)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Emission spectrum
    ax1.plot(actual_freqs/1e9, emission_intensity, 'b-', label='Observed')
    ax1.plot(actual_freqs/1e9, theoretical_intensity, 'r--', label='Theoretical')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Emission Spectrum')
    ax1.legend()
    
    # Plot 2: Harmonic structure
    ax2.plot(harmonic_numbers, emission_intensity, 'g-')
    ax2.set_xlabel('Harmonic Number')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Harmonic Structure')
    
    # Plot 3: Power law fit
    ax3.loglog(actual_freqs/1e9, emission_intensity, 'b.')
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Log Intensity')
    ax3.set_title('Power Law Distribution')
    
    # Plot 4: Spectral index
    ax4.plot(actual_freqs/1e9, spectral_index, 'k-')
    ax4.set_xlabel('Frequency (GHz)')
    ax4.set_ylabel('Spectral Index')
    ax4.set_title('Spectral Index Variation')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gyro_analysis_{date}_{time}.png')
    plt.close()
    
    return peak_frequency, harmonic_numbers.mean(), power_law_index

