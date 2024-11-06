import numpy as np
import logging
from scipy import signal
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cupy as cp
import traceback
import gc

def process_chunk(chunk, fs, full_data, start_row, n_std=3, effective_freq=None, effective_low=None, effective_high=None, nyquist_zone=None):
    # Perform GPU-accelerated FFT
    chunk_gpu = cp.asarray(chunk)
    fft_result = cp.fft.fft(chunk_gpu)
    freq_bins = cp.fft.fftfreq(len(chunk_gpu), d=1/fs)
    
    # Adjust frequency bins for bandpass sampling if in higher Nyquist zones
    if nyquist_zone and nyquist_zone > 1:
        freq_bins = freq_bins + (nyquist_zone - 1) * fs/2
        
    # Apply bandpass filter for the effective frequency range
    if effective_low is not None and effective_high is not None:
        freq_mask = (cp.abs(freq_bins) >= effective_low) & (cp.abs(freq_bins) <= effective_high)
        fft_result = cp.where(freq_mask, fft_result, 0)
    
    # Apply adaptive thresholding to magnitude of FFT result
    magnitude = cp.abs(fft_result)
    threshold = adaptive_threshold(magnitude, n_std)
    
    # Filter the FFT result
    mask = magnitude > threshold
    filtered_fft = fft_result[mask]
    
    # Transfer results back to CPU and store in full_data
    filtered_fft = cp.asnumpy(filtered_fft)
    end_row = start_row + len(filtered_fft)
    full_data[start_row:end_row] = filtered_fft
    
    return len(filtered_fft)



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
    # Sort frequency range correctly for high frequencies
    effective_low, effective_high = min(low_cutoff, high_cutoff), max(low_cutoff, high_cutoff)
    
    # Calculate FFT with proper scaling for high frequencies
    fft_values = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1/fs)
    
    # Adjust frequency mask for high frequency ranges
    freq_tolerance = 0.01 * fs  # Add tolerance based on sampling rate
    mask = (freqs >= effective_low - freq_tolerance) & (freqs <= effective_high + freq_tolerance)
    
    # Apply gain for weak signals
    signal_strength = np.abs(fft_values).mean()
    if signal_strength < 1e-10:
        fft_values *= 1e6  # Increased gain for high frequencies
        
    filtered_fft = fft_values[mask]
    filtered_freq = freqs[mask]
    signal_power = np.abs(filtered_fft)**2
    
    logging.debug(f"Frequency range: {freqs.min()} to {freqs.max()} Hz")
    logging.debug(f"Mask range: {effective_low} to {effective_high} Hz")
    
    return effective_high - effective_low, signal_power, mask, filtered_freq, filtered_fft



def compute_bandwidth_and_cutoffs(full_data, center_frequency, low_cutoff, high_cutoff, min_bandwidth=1000):
    """Calculates bandwidth and applies cutoffs to frequency and FFT values."""
    
    # Compute FFT of time domain signal
    fft_values = np.fft.fft(full_data)
    freq = np.fft.fftfreq(len(full_data), d=1/20e6)

    # Convert to 2D array format expected by rest of function
    full_data = np.column_stack((freq, fft_values))
    
    # Continue with existing processing
    freq = full_data[:, 0].real
    fft_values = np.abs(full_data[:, 1])
    
    logging.debug(f"Frequency range: {freq.min()} to {freq.max()} Hz")
    logging.debug(f"FFT values range: {fft_values.min()} to {fft_values.max()}")

    # Calculate threshold for significant frequencies
    threshold = np.max(fft_values) * 0.1
    significant_freqs = freq[fft_values > threshold]
    
    logging.debug(f"Threshold: {threshold}")
    logging.debug(f"Number of significant frequencies: {len(significant_freqs)}")

    # Determine initial bandwidth if not provided
    initial_bandwidth = high_cutoff - low_cutoff

    logging.debug(f"Initial bandwidth: {initial_bandwidth} Hz")

    # Adjust cutoffs based on initial bandwidth and provided values
    low_cutoff = center_frequency - initial_bandwidth / 2
    high_cutoff = center_frequency + initial_bandwidth / 2
    
    # Ensure low_cutoff and high_cutoff are within the actual frequency range
    low_cutoff = max(low_cutoff, freq.min())
    high_cutoff = min(high_cutoff, freq.max())
    
    if low_cutoff >= high_cutoff:
        logging.warning("Low cutoff is not less than high cutoff. Reverting to full frequency range.")
        low_cutoff = freq.min()
        high_cutoff = freq.max()

    logging.debug(f"Low cutoff: {low_cutoff} Hz")
    logging.debug(f"High cutoff: {high_cutoff} Hz")

    # Filter frequencies and FFT values based on cutoffs
    mask = (freq >= low_cutoff) & (freq <= high_cutoff)
    filtered_freq = freq[mask]
    filtered_fft_values = fft_values[mask]

    if len(filtered_freq) == 0:
        logging.warning("No frequencies left after filtering. Reverting to full frequency range.")
        filtered_freq = freq
        filtered_fft_values = fft_values

    logging.debug(f"Number of frequencies after filtering: {len(filtered_freq)}")
    logging.debug(f"Frequency range After Bandwidth and Cutoff: {filtered_freq.min()} to {filtered_freq.max()} Hz")

    return initial_bandwidth, low_cutoff, high_cutoff, filtered_freq, filtered_fft_values
