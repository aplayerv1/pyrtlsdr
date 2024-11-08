import numpy as np
import logging
from scipy import signal
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cupy as cp
import traceback
import gc

import cupy as cp
import logging
import numpy as np

def process_chunk(chunk, fs, full_data, chunk_start, gain, effective_freq=0, effective_low=0, effective_high=None, center_frequency=0):
    logging.debug(f"Processing chunk at {chunk_start} with effective freq {effective_freq} Hz")
    
    # Transfer chunk to GPU
    chunk_gpu = cp.asarray(chunk)
    
    # Initial signal validation on GPU
    chunk_power = cp.abs(chunk_gpu).mean()
    logging.debug(f"Chunk power before processing: {chunk_power}")
    
    # Apply gain if signal is weak
    if chunk_power < 1e-10:
        chunk_gpu *= gain  # Apply gain directly
        logging.debug(f"Applied gain. New chunk power: {cp.abs(chunk_gpu).mean()}")
    
    # Compute FFT with frequency scaling on GPU
    fft_chunk_gpu = cp.fft.fft(chunk_gpu)
    freqs_gpu = cp.fft.fftfreq(len(chunk_gpu), d=1/fs)
    
    # Log first few frequency bins for debugging
    logging.debug(f"Frequencies: {freqs_gpu[:10]}...")
    
    # Make sure effective_freq is valid (e.g., set to the center frequency)
    if effective_freq == 0:
        effective_freq = center_frequency  # Use center_frequency if effective_freq is zero
    
    # Calculate mask based on the effective frequency and low/high cutoffs
    if effective_high is not None and effective_low is not None:
        mask_gpu = cp.abs(freqs_gpu - effective_freq) <= (effective_high - effective_low) / 2
    else:
        # Use a default mask if high and low cutoffs are not set
        mask_gpu = cp.abs(freqs_gpu - effective_freq) <= 10e6  # Adjust as needed
    
    # Log the mask to debug
    logging.debug(f"Mask: {mask_gpu[:10]}...")  # Log first few values of the mask
    
    # Zero out frequencies outside of the mask
    fft_chunk_gpu[~mask_gpu] = 0
    
    # Log FFT statistics after applying the mask
    logging.debug(f"FFT mean magnitude after mask: {cp.abs(fft_chunk_gpu).mean()}")
    logging.debug(f"FFT max magnitude after mask: {cp.abs(fft_chunk_gpu).max()}")
    
    # Transfer processed FFT chunk back to the CPU for storage
    full_data[chunk_start:chunk_start + len(chunk_gpu)] = cp.asnumpy(fft_chunk_gpu)
    
    return fft_chunk_gpu




def adaptive_threshold(fft_values, n_std=3):
    fft_values_gpu = cp.asarray(fft_values)
    median = cp.median(fft_values_gpu)
    mad = cp.median(cp.abs(fft_values_gpu - median))
    threshold = median + n_std * mad / 0.6745
    return float(threshold.get())

def parallel_fft(data_chunk):
    return np.fft.fft(data_chunk)

def compute_gpu_fft(full_data):
    data_gpu = cp.asarray(full_data)
    fft_result = cp.fft.fft(data_gpu)
    return cp.asnumpy(fft_result)

def cyclostationary_feature_detection(data, fs, alpha, f):
    t = np.arange(len(data)) / fs
    x_alpha = data * np.exp(-1j * 2 * np.pi * alpha * t)
    f, Sxx = signal.welch(x_alpha, fs, nperseg=1024)
    return f, Sxx

def detect_cyclostationary_signals(freq, fft_values, fs, alpha_range):
    detected_signals = []
    for alpha in alpha_range:
        f, Sxx = cyclostationary_feature_detection(fft_values, fs, alpha, freq)
        peaks, _ = signal.find_peaks(Sxx, height=np.mean(Sxx) + 2*np.std(Sxx))
        if len(peaks) > 0:
            detected_signals.append((alpha, f[peaks[0]]))
    return detected_signals

def classify_signals(freq, fft_values, n_clusters=3):
    X = np.column_stack((freq, fft_values))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def analyze_clusters(freq, fft_values, labels):
    for i in range(max(labels) + 1):
        cluster_freq = freq[labels == i]
        cluster_fft = fft_values[labels == i]
        logging.info(f"Cluster {i}: Avg frequency = {np.mean(cluster_freq):.2f} Hz, Avg magnitude = {np.mean(cluster_fft):.2f}")

def robust_compute_bandwidth(data, center_frequency, low_cutoff, high_cutoff, fs):
    logging.debug(f"Starting robust_compute_bandwidth with parameters:")
    logging.debug(f"Center Frequency: {center_frequency} Hz, Low Cutoff: {low_cutoff} Hz, High Cutoff: {high_cutoff} Hz, fs: {fs} Hz")
    effective_low, effective_high = min(low_cutoff, high_cutoff), max(low_cutoff, high_cutoff)
    
    # Adjust the cutoffs to be within the Nyquist frequency limit
    nyquist_freq = fs / 2
    effective_low = min(effective_low, nyquist_freq)
    effective_high = min(effective_high, nyquist_freq)
    # Sort frequency range correctly for high frequencies
    logging.debug(f"Effective frequency range after sorting: {effective_low} Hz to {effective_high} Hz")
    
    # Specialized processing for different frequency ranges
    if center_frequency == 0:  # Baseband
        logging.debug("Using Baseband processing with Kaiser window")
        window = signal.kaiser(len(data), beta=14)
        gain_factor = 2e4
        freq_tolerance = 0.02 * fs
    elif center_frequency > 1e9:  # Above 1 GHz
        logging.debug("Using Above 1 GHz processing with Blackman window")
        window = signal.blackman(len(data))
        gain_factor = 1e6
        freq_tolerance = 0.005 * fs
    else:  # Standard processing
        logging.debug("Using Standard processing with Hamming window")
        window = signal.hamming(len(data))
        gain_factor = 1e4
        freq_tolerance = 0.01 * fs

    # Apply window and calculate FFT
    logging.debug(f"Applying window of length {len(data)} to the data")
    fft_values = np.fft.fft(data * window)
    freqs = np.fft.fftfreq(len(data), d=1/fs)
    
    # Log frequency bins
    logging.debug(f"FFT computation done. Frequency bins: {len(freqs)}")
    logging.debug(f"First few frequencies: {freqs[:10]}")
    logging.debug(f"Last few frequencies: {freqs[-10:]}")

    # Create frequency mask with tolerance
    mask = (freqs >= effective_low - freq_tolerance) & (freqs <= effective_high + freq_tolerance)
    logging.debug(f"Mask created with tolerance: {freq_tolerance} Hz")

    # Log the range of values in the mask
    logging.debug(f"Mask contains {np.sum(mask)} valid frequency bins.")
    logging.debug(f"First few values of mask: {mask[:10]}")
    logging.debug(f"Last few values of mask: {mask[-10:]}")
    
    # Check if mask is empty
    if np.sum(mask) == 0:
        logging.error(f"Mask is empty! No valid frequencies found in the range {effective_low} to {effective_high}.")
        return 0, np.array([]), mask, np.array([]), np.array([])

    # Apply gain for signal enhancement
    signal_strength = np.abs(fft_values).mean()
    logging.debug(f"Initial signal strength: {signal_strength}")
    
    if signal_strength < 1e-10:
        logging.debug(f"Signal strength below threshold, applying gain factor of {gain_factor}")
        fft_values *= gain_factor
        
    filtered_fft = fft_values[mask]
    filtered_freq = freqs[mask]
    logging.debug(f"Filtered FFT and frequency arrays created with {len(filtered_freq)} bins")

    # Ensure both arrays have the same length
    if len(filtered_freq) != len(filtered_fft):
        logging.error(f"Array size mismatch: filtered_freq has length {len(filtered_freq)}, "
                      f"filtered_fft has length {len(filtered_fft)}.")
        return 0, np.array([]), mask, np.array([]), np.array([])

    # Calculate signal power
    signal_power = np.abs(filtered_fft)**2
    logging.debug(f"Signal power calculated, shape: {signal_power.shape}")

    logging.debug(f"Frequency range: {freqs.min()} to {freqs.max()} Hz")
    logging.debug(f"Mask range: {effective_low} to {effective_high} Hz")
    logging.debug(f"Signal strength after processing: {np.abs(filtered_fft).mean()}")
    
    return effective_high - effective_low, signal_power, mask, filtered_freq, filtered_fft
