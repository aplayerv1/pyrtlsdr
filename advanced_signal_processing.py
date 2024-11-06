import numpy as np
import cupy as cp
import multiprocessing as mp
from multiprocessing import Queue
import logging
from numba import cuda
import numba
from scipy import signal, ndimage
from scipy.signal import lfilter, medfilt
import pywt
from PyEMD import EMD
from sklearn.decomposition import FastICA
# from py_kernel.kernels import phase_locked_loop_gpu
from cupyx.scipy import signal as cp_signal
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import time
import math

# Constants
CHUNK_SIZE = 1024 * 1024  # Standard 1MB chunks
NUM_CPU_THREADS = multiprocessing.cpu_count()
THREADS_PER_BLOCK = 256
NUM_STREAMS = 4
SPEED_OF_LIGHT = 299792458
EARTH_ROTATION_RATE = 15
TOLERANCE = 1e6
NOTCH_FREQ = 9750
NOTCH_WIDTH = 30
MAGNETIC_FIELD_STRENGTH = 1
K_B = 1.38e-23
gpu_lock = mp.Lock()


# Optimized constants for P400
OPTIMAL_CHUNK_SIZE = 512 * 1024  # 512KB chunks
MAX_STREAMS = 2
PINNED_MEMORY_LIMIT = 1024 * 1024 * 1024  # 1GB pinned memory limit

@contextmanager
def optimized_gpu_memory():
    """Enhanced memory manager with pinned memory support"""
    pinned_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
    try:
        yield
    finally:
        pinned_pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()

# Utility functions and context managers
def get_mmap_path(function_name):
    """Generate unique memory map file path"""
    return f'temp_{function_name}_{time.time()}.mmap'

@contextmanager
def managed_mmap(path, dtype, shape):
    """Context manager for memory mapped files"""
    mmap_file = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    try:
        yield mmap_file
    finally:
        mmap_file._mmap.close()
        if os.path.exists(path):
            os.remove(path)

@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory cleanup"""
    try:
        yield
    finally:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def apply_rotation(data, rotation_angle):
    if len(data.shape) != 1:
        raise ValueError("Input data must be a 1D array.")
    data_2d = data[:, np.newaxis]
    rotated_data_2d = ndimage.rotate(data_2d, rotation_angle, reshape=False, mode='nearest')
    rotated_data = rotated_data_2d.flatten()
    return rotated_data

def remove_dc_offset(signal):
    if isinstance(signal, cp.ndarray):
        signal = cp.nan_to_num(signal)
        mean_val = cp.mean(signal)
        result = signal - mean_val
        logging.debug(f"DC Offset Mean: {mean_val}")
        return cp.asnumpy(result)
    else:
        signal = np.nan_to_num(signal)
        mean_val = np.mean(signal)
        logging.debug(f"DC Offset Mean: {mean_val}")
        return signal - mean_val


def denoise_signal(data, wavelet='db1', level=1):
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    
    if len(data.shape) != 1:
        raise ValueError("Input data must be a 1D array.")

    coeffs = pywt.wavedec(data, wavelet, level=level)

    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))

    new_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(new_coeffs, wavelet)

    return cp.asarray(denoised_signal) if isinstance(data, cp.ndarray) else denoised_signal

def remove_lnb_effect(signal_data, fs, notch_freq, notch_width):
    logging.debug("Starting remove_lnb_effect function")
    
    # Ensure fs, notch_freq, and notch_width are not zero
    if fs <= 0:
        raise ValueError("Sampling frequency (fs) must be greater than zero.")
    if notch_freq <= 0:
        raise ValueError("Notch frequency must be greater than zero.")
    if notch_width <= 0:
        raise ValueError("Notch width must be greater than zero.")
    
    signal_data = np.asarray(signal_data, dtype=np.float64)
    signal_data = np.nan_to_num(signal_data)  # Handle NaNs
    
    logging.debug(f"Original signal size: {signal_data.size}")
    logging.debug(f"Original signal mean: {np.mean(signal_data)}")
    
    # Convert notch frequency and width to normalized values
    notch_freq_normalized = notch_freq / (0.5 * fs)
    notch_width_normalized = notch_width / (0.5 * fs)
    
    # Ensure normalized values are within the valid range
    if notch_freq_normalized <= 0 or notch_freq_normalized >= 1:
        raise ValueError(f"Normalized notch frequency out of range: {notch_freq_normalized}.")
    if notch_width_normalized <= 0 or notch_width_normalized >= 1:
        raise ValueError(f"Normalized notch width out of range: {notch_width_normalized}.")
    
    # Compute notch filter coefficients using scipy.signal.iirnotch
    try:
        b, a = signal.iirnotch(notch_freq_normalized, notch_width_normalized)
    except Exception as e:
        logging.error(f"Error computing notch filter coefficients: {e}")
        raise
    
    logging.debug(f"Filter coefficients (b): {b}")
    logging.debug(f"Filter coefficients (a): {a}")
    
    # Apply the notch filter
    try:
        processed_signal = signal.filtfilt(b, a, signal_data)
    except Exception as e:
        logging.error(f"Error applying notch filter: {e}")
        raise
    
    logging.debug("Applied notch filter to the signal")
    
    logging.debug(f"Processed signal size: {processed_signal.size}")
    logging.debug(f"Processed signal mean: {np.mean(processed_signal)}")
    
    return processed_signal
    

@cuda.jit
def adaptive_filter_kernel(input_signal, desired_signal, output_signal, filter_coeffs, step_size):
    idx = cuda.grid(1)
    tx = cuda.threadIdx.x
    filter_length = filter_coeffs.size
    shared_input = cuda.shared.array(shape=(256,), dtype=numba.complex128)
    
    if idx < input_signal.size:
        # Load data into shared memory
        shared_input[tx] = input_signal[idx]
        cuda.syncthreads()
        
        # Compute filter output
        if idx < filter_length:
            segment = input_signal[max(0, idx-filter_length+1):idx+1][::-1]
            output_signal[idx] = 0
            for j in range(len(segment)):
                output_signal[idx] += filter_coeffs[j] * segment[j]
        else:
            output_signal[idx] = 0
            for j in range(filter_length):
                output_signal[idx] += filter_coeffs[j] * input_signal[idx-j]
                
        # Update filter coefficients
        error = desired_signal[idx] - output_signal[idx]
        update_length = min(filter_length, idx+1)
        for j in range(update_length):
            filter_coeffs[j] += step_size * error.conjugate() * shared_input[tx-j]

def adaptive_filter(input_signal, desired_signal, effective_freq, step_size=0.1, filter_length=10):
    """GPU-accelerated adaptive filter with memory mapping and CPU fallback"""
    logging.debug(f"Starting adaptive filter - step_size: {step_size}, filter_length: {filter_length}")
    logging.debug(f"Processing at effective frequency: {effective_freq} Hz")
    
    # Adjust step size based on frequency range
    if effective_freq > 1e9:
        step_size *= 0.5  # Smaller steps for higher frequencies
    elif effective_freq > 500e6:
        step_size *= 0.75
    
    mmap_path = get_mmap_path('adaptive_filtered')
    
    with managed_mmap(mmap_path, input_signal.dtype, input_signal.shape) as filtered_mmap:
        streams = [cp.cuda.Stream() for _ in range(NUM_STREAMS)]
        
        for i in range(0, len(input_signal), CHUNK_SIZE):
            chunk_end = min(i + CHUNK_SIZE, len(input_signal))
            input_chunk = input_signal[i:chunk_end]
            desired_chunk = desired_signal[i:chunk_end]
            stream_idx = (i // CHUNK_SIZE) % NUM_STREAMS
            
            try:
                with streams[stream_idx], gpu_memory_manager():
                    input_gpu = cp.asarray(input_chunk)
                    desired_gpu = cp.asarray(desired_chunk)
                    filter_coeffs = cp.zeros(filter_length, dtype=cp.complex128)
                    output_gpu = cp.zeros_like(input_gpu)
                    
                    blocks = (len(input_chunk) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
                    adaptive_filter_kernel[blocks, THREADS_PER_BLOCK](
                        input_gpu, desired_gpu, output_gpu, filter_coeffs, step_size
                    )
                    
                    filtered_mmap[i:chunk_end] = cp.asnumpy(output_gpu)
                    filtered_mmap.flush()
                    
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
                logging.warning(f"GPU processing failed: {str(e)}. Using CPU processing.")
                filtered_chunk = cpu_adaptive_filter(input_chunk, desired_chunk, step_size, filter_length)
                filtered_mmap[i:chunk_end] = filtered_chunk
                filtered_mmap.flush()

        return np.array(filtered_mmap[:])

def cpu_adaptive_filter(input_signal, desired_signal, step_size=0.1, filter_length=10):
    """CPU implementation of adaptive filter"""
    output_signal = np.zeros_like(input_signal)
    filter_coeffs = np.zeros(filter_length, dtype=np.complex128)
    
    def process_chunk(chunk_data):
        chunk, desired = chunk_data
        chunk_output = np.zeros_like(chunk)
        local_coeffs = filter_coeffs.copy()
        
        for i in range(len(chunk)):
            if i < filter_length:
                segment = chunk[max(0, i-filter_length+1):i+1][::-1]
            else:
                segment = chunk[i-filter_length:i][::-1]
            
            chunk_output[i] = np.sum(local_coeffs[:len(segment)] * segment)
            error = desired[i] - chunk_output[i]
            local_coeffs[:len(segment)] += step_size * error * np.conj(segment)
        
        return chunk_output
    
    chunk_size = len(input_signal) // (NUM_CPU_THREADS * 4)
    with ThreadPoolExecutor(max_workers=NUM_CPU_THREADS) as executor:
        chunks = [(input_signal[i:i+chunk_size], desired_signal[i:i+chunk_size]) 
                 for i in range(0, len(input_signal), chunk_size)]
        results = list(executor.map(process_chunk, chunks))
        
        for i, result in enumerate(results):
            start_idx = i * chunk_size
            output_signal[start_idx:start_idx+len(result)] = result
    
    return output_signal

def wavelet_denoise(signal, effective_freq, wavelet='db4', level=1):
    logging.debug(f"Starting wavelet denoising - wavelet: {wavelet}, level: {level}")
    logging.debug(f"Processing at effective frequency: {effective_freq} Hz")
    
    # Adjust wavelet parameters based on frequency
    if effective_freq > 1e9:
        level = 2  # More decomposition levels for high frequencies
        wavelet = 'db6'  # Different wavelet for better high frequency handling
    elif effective_freq > 500e6:
        level = 1
        wavelet = 'db4'
    
    mmap_path = get_mmap_path('wavelet_denoised')
    
    with managed_mmap(mmap_path, signal.dtype, signal.shape) as denoised_mmap:
        streams = [cp.cuda.Stream() for _ in range(NUM_STREAMS)]
        
        for i in range(0, len(signal), CHUNK_SIZE):
            chunk_end = min(i + CHUNK_SIZE, len(signal))
            chunk = signal[i:chunk_end]
            stream_idx = (i // CHUNK_SIZE) % NUM_STREAMS
            
            try:
                with streams[stream_idx], gpu_memory_manager():
                    # Convert chunk to NumPy for wavelet processing
                    np_chunk = cp.asnumpy(chunk) if isinstance(chunk, cp.ndarray) else chunk
                    coeffs = pywt.wavedec(np_chunk, wavelet, level=level)
                    threshold = float(cp.sqrt(2 * cp.log(len(chunk))))
                    coeffs = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]
                    denoised_chunk = pywt.waverec(coeffs, wavelet)
                    
                    if len(denoised_chunk) != len(chunk):
                        denoised_chunk = denoised_chunk[:len(chunk)]
                    
                    denoised_mmap[i:chunk_end] = denoised_chunk
                    denoised_mmap.flush()
                    
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
                logging.warning(f"GPU processing failed. Using CPU processing.")
                denoised_chunk = cpu_wavelet_denoise(chunk, wavelet, level)
                denoised_mmap[i:chunk_end] = denoised_chunk
                denoised_mmap.flush()

        return np.array(denoised_mmap[:])


def cpu_wavelet_denoise(signal, wavelet='db4', level=1):
    """CPU implementation of wavelet denoising using parallel processing"""
    logging.debug(f"Starting CPU wavelet denoising - wavelet: {wavelet}, level: {level}")
    
    def process_chunk(chunk):
        coeffs = pywt.wavedec(chunk, wavelet, level=level)
        threshold = np.sqrt(2 * np.log(len(chunk)))
        coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]
        return pywt.waverec(coeffs, wavelet)
    
    chunk_size = len(signal) // (NUM_CPU_THREADS * 4)
    chunks = [signal[i:i+chunk_size] for i in range(0, len(signal), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=NUM_CPU_THREADS) as executor:
        denoised_chunks = list(executor.map(process_chunk, chunks))
    
    return np.concatenate(denoised_chunks)


@cuda.jit
def kalman_kernel(x_hat, P, K, z, Q, R):
    idx = cuda.grid(1)
    tx = cuda.threadIdx.x
    
    # Match shared memory type with input array type
    shared_mem = cuda.shared.array(shape=(THREADS_PER_BLOCK,), dtype=numba.complex128)
    
    if idx < z.size and tx < THREADS_PER_BLOCK:
        shared_mem[tx] = z[idx]
        cuda.syncthreads()
        
        if idx == 0:
            x_hat[0] = shared_mem[tx]
            P[0] = 1.0
            K[0] = 0.0
        elif idx < x_hat.size:
            x_minus = x_hat[idx-1]
            P_minus = P[idx-1] + Q
            K[idx] = P_minus / (P_minus + R)
            x_hat[idx] = x_minus + K[idx] * (shared_mem[tx] - x_minus)
            P[idx] = (1 - abs(K[idx])) * P_minus


def kalman_filter_gpu(z, effective_freq, Q=1e-5, R=1e-2):
    """Optimized Kalman filter for high frequencies"""
    # Adjust Q and R based on frequency range
    if effective_freq > 1e9:
        Q = 1e-6  # More precise process noise for high frequencies
        R = 1e-3  # Lower measurement noise for high frequencies
    elif effective_freq > 500e6:
        Q = 1e-5
        R = 1e-2

    OPTIMAL_CHUNK_SIZE = 512 * 1024
    total_chunks = len(z) // OPTIMAL_CHUNK_SIZE + (1 if len(z) % OPTIMAL_CHUNK_SIZE else 0)
    result = np.zeros_like(z)
    
    for i in range(0, len(z), OPTIMAL_CHUNK_SIZE):
        chunk_end = min(i + OPTIMAL_CHUNK_SIZE, len(z))
        chunk = z[i:chunk_end]
        
        d_chunk = cp.array(chunk)
        x_hat = cp.zeros_like(d_chunk)
        P = cp.zeros(len(chunk), dtype=cp.float64)
        K = cp.zeros_like(d_chunk)
        
        blocks = (len(chunk) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        kalman_kernel[blocks, THREADS_PER_BLOCK](x_hat, P, K, d_chunk, Q, R)
        result[i:chunk_end] = cp.asnumpy(x_hat)
        
        cp.get_default_memory_pool().free_all_blocks()
        
        logging.info(f"Kalman Filter Progress: {(i//OPTIMAL_CHUNK_SIZE)+1}/{total_chunks} chunks ({((i//OPTIMAL_CHUNK_SIZE)+1)/total_chunks*100:.1f}%)")
    
    return result

def cpu_kalman_filter(z, Q=1e-5, R=1e-2):
    x_hat = np.zeros_like(z)
    P = np.zeros(len(z), dtype=np.float64)
    K = np.zeros_like(z)
    for i in range(1, len(z)):
        x_minus = x_hat[i-1]
        P_minus = P[i-1] + Q
        K[i] = P_minus / (P_minus + R)
        x_hat[i] = x_minus + K[i] * (z[i] - x_minus)
        P[i] = (1 - abs(K[i])) * P_minus

    return x_hat

def spectral_subtraction(signal, noise_estimate, effective_freq, alpha=2, beta=0.01):
    """
    Apply spectral subtraction with frequency-aware parameters
    """
    logging.debug(f"Applying spectral subtraction at effective frequency: {effective_freq} Hz")
    
    # Adjust parameters based on frequency range
    if effective_freq > 1e9:
        alpha = 3  # More aggressive noise reduction for high frequencies
        beta = 0.005  # Lower floor for high frequencies
    elif effective_freq > 500e6:
        alpha = 2.5
        beta = 0.008

    signal_fft = np.fft.fft(signal)
    signal_mag = np.abs(signal_fft)
    signal_phase = np.angle(signal_fft)

    subtracted_mag = np.maximum(signal_mag**2 - alpha * noise_estimate**2, beta * signal_mag**2)**0.5
    subtracted_signal = np.fft.ifft(subtracted_mag * np.exp(1j * signal_phase))

    logging.debug(f"Max magnitude before subtraction: {np.max(signal_mag)}")
    logging.debug(f"Max magnitude after subtraction: {np.max(subtracted_mag)}")

    return np.real(subtracted_signal)


def median_filter(signal, effective_freq, kernel_size=3):
    """
    Apply median filtering with frequency-optimized kernel size
    """
    # Adjust kernel size based on frequency range
    if effective_freq > 1e9:
        kernel_size = 7  # Larger kernel for high frequencies
    elif effective_freq > 500e6:
        kernel_size = 5  # Medium kernel for mid-range frequencies
        
    logging.debug(f"Applying median filter at effective frequency: {effective_freq} Hz")
    logging.debug(f"Using kernel size: {kernel_size}")
    
    return medfilt(signal, kernel_size=kernel_size)

@cuda.jit(device=True)
def process_shared_data(data):
    """Device function to process shared memory data"""
    # Apply filtering operations
    filtered_value = data * (1.0 - 1.0/(1.0 + data * data))
    return filtered_value

@cuda.jit
def optimized_comb_kernel(signal, freqs, notch_freqs, Q, output):
    idx = cuda.grid(1)
    if idx < signal.shape[0]:
        shared_freq = cuda.shared.array(shape=(256,), dtype=numba.float64)
        
        for i in range(0, len(notch_freqs), 256):
            if idx < min(256, len(notch_freqs) - i):
                shared_freq[idx] = notch_freqs[i + idx]
            cuda.syncthreads()
            
            for j in range(min(256, len(notch_freqs) - i)):
                notch = shared_freq[j]
                freq_diff = abs(freqs[idx] - notch)
                if freq_diff < Q:
                    signal[idx] *= (freq_diff / Q)
            cuda.syncthreads()

def comb_filter(input_signal, fs, effective_freq, notch_freq, Q=2):
    logging.debug(f"Starting comb filter at effective frequency: {effective_freq} Hz")
    
    # Adjust Q factor based on frequency range
    if effective_freq > 1e9:
        Q = 4  # Sharper notches for high frequencies
    elif effective_freq > 500e6:
        Q = 3
        
    CHUNK_SIZE = 128 * 1024
    total_chunks = len(input_signal) // CHUNK_SIZE + (1 if len(input_signal) % CHUNK_SIZE else 0)
    result = np.zeros_like(input_signal)
    
    for i in range(0, len(input_signal), CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE, len(input_signal))
        signal_chunk = input_signal[i:chunk_end]
        
        try:
            fft_chunk = np.fft.fft(signal_chunk)
            d_fft_chunk = cp.asarray(fft_chunk)
            d_freqs = cp.fft.fftfreq(len(signal_chunk), 1/fs)
            d_notch_freqs = cp.arange(notch_freq, 0.5 * fs, notch_freq)
            d_output = cp.zeros_like(d_fft_chunk)
            
            blocks = (len(signal_chunk) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            optimized_comb_kernel[blocks, THREADS_PER_BLOCK](
                d_fft_chunk, d_freqs, d_notch_freqs, Q, d_output
            )
            
            filtered_chunk = cp.asnumpy(d_output)
            result[i:chunk_end] = np.real(np.fft.ifft(filtered_chunk))
            
            del d_fft_chunk, d_freqs, d_notch_freqs, d_output
            cp.get_default_memory_pool().free_all_blocks()
            
        except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
            logging.warning(f"GPU processing failed: {str(e)}. Using CPU processing.")
            result[i:chunk_end] = cpu_comb_filter(signal_chunk, fs, notch_freq, Q)
        
        logging.info(f"Comb Filter Progress: {(i//CHUNK_SIZE)+1}/{total_chunks} chunks ({((i//CHUNK_SIZE)+1)/total_chunks*100:.1f}%)")
    
    return result

def cpu_comb_filter(signal_chunk, fs, notch_freq, Q):
    """CPU fallback for comb filter processing"""
    fft_chunk = np.fft.fft(signal_chunk)
    freqs = np.fft.fftfreq(len(signal_chunk), 1/fs)
    notch_freqs = np.arange(notch_freq, 0.5 * fs, notch_freq)
    
    with ThreadPoolExecutor(max_workers=NUM_CPU_THREADS) as executor:
        def process_notch(notch):
            mask = np.abs(freqs - notch) < Q
            fft_chunk[mask] *= 0.1
        
        list(executor.map(process_notch, notch_freqs))
    
    return fft_chunk

@cuda.jit
def emd_shared_memory_kernel(signal, imfs, shared_mem):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    idx = bx * block_size + tx
    
    if idx < signal.size:
        shared_mem[tx] = signal[idx]
    cuda.syncthreads()
    
    if idx < signal.size:
        for i in range(imfs.shape[0]):
            imfs[i, idx] = shared_mem[tx]

def empirical_mode_decomposition(signal, num_imfs, chunk_size=CHUNK_SIZE):
    """GPU-accelerated EMD with memory mapping and CPU fallback"""
    logging.debug(f"Starting EMD processing - chunk_size: {chunk_size}")
    
    if signal is None or len(signal) == 0:
        return cp.array([]), cp.array([])

    imfs_path = get_mmap_path('emd_imfs')
    residual_path = get_mmap_path('emd_residual')
    
    with managed_mmap(imfs_path, signal.dtype, (num_imfs, len(signal))) as imfs_mmap, \
         managed_mmap(residual_path, signal.dtype, signal.shape) as residual_mmap:
        
        streams = [cp.cuda.Stream() for _ in range(NUM_STREAMS)]
        
        for i in range(0, len(signal), chunk_size):
            chunk_end = min(i + chunk_size, len(signal))
            chunk = signal[i:chunk_end]
            stream_idx = (i // chunk_size) % NUM_STREAMS
            
            try:
                with streams[stream_idx], gpu_memory_manager():
                    chunk_gpu = cp.array(chunk)
                    blocks = (len(chunk) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
                    
                    emd_shared_memory_kernel[blocks, THREADS_PER_BLOCK](
                        chunk_gpu,
                        cp.zeros((num_imfs, len(chunk)), dtype=cp.float64),
                        cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=numba.float64)
                    )
                    
                    chunk_imfs = EMD().emd(cp.asnumpy(chunk_gpu), max_imf=num_imfs)
                    imfs_mmap[:, i:chunk_end] = chunk_imfs
                    residual_mmap[i:chunk_end] = chunk - np.sum(chunk_imfs, axis=0)
                    imfs_mmap.flush()
                    residual_mmap.flush()
                    
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
                logging.warning(f"GPU processing failed: {str(e)}. Using CPU processing.")
                chunk_imfs = cpu_emd_denoising(chunk, num_imfs, 0.1)
                imfs_mmap[:, i:chunk_end] = chunk_imfs
                residual_mmap[i:chunk_end] = chunk - np.sum(chunk_imfs, axis=0)
                imfs_mmap.flush()
                residual_mmap.flush()

        return cp.array(imfs_mmap[:]), cp.array(residual_mmap[:])

@cuda.jit
def emd_denoising_kernel(imfs, energy_ratio, noise_threshold, denoised_signal):
    idx = cuda.grid(1)
    tx = cuda.threadIdx.x
    
    # Match shared memory dimensions to thread block size
    shared_imfs = cuda.shared.array(shape=(THREADS_PER_BLOCK, 2), dtype=numba.float64)
    
    if idx < denoised_signal.size and tx < THREADS_PER_BLOCK:
        for i in range(imfs.shape[0]):
            if i < shared_imfs.shape[1]:
                shared_imfs[tx, i] = imfs[i, idx]
        
        cuda.syncthreads()
        
        denoised_signal[idx] = 0
        for i in range(imfs.shape[0]):
            if i < shared_imfs.shape[1] and energy_ratio[i] > noise_threshold:
                denoised_signal[idx] += shared_imfs[tx, i]

def parallel_emd_gpu(signal, max_imf=None):
    imfs = []
    residue = signal.copy()
    
    num_streams = 4
    streams = [cp.cuda.Stream() for _ in range(num_streams)]
    current_stream = 0
    
    while len(imfs) < max_imf if max_imf else True:
        with streams[current_stream]:
            window_length = 801
            upper_env = cp_signal.savgol_filter(residue, window_length=window_length, polyorder=3)
            lower_env = -cp_signal.savgol_filter(-residue, window_length=window_length, polyorder=3)
            mean_env = (upper_env + lower_env) / 2
            
            imf = residue - mean_env
            imfs.append(imf)
            residue = mean_env
            
            current_stream = (current_stream + 1) % num_streams
            
            if cp.std(residue) < 1e-8:
                break
    
    return cp.stack(imfs)

def emd_denoising(signal, effective_freq, num_imfs=None, noise_threshold=0.1):
    """GPU-accelerated EMD denoising optimized for frequency ranges"""
    logging.debug(f"Starting EMD denoising at effective frequency: {effective_freq} Hz")
    
    # Adjust parameters based on frequency
    if effective_freq > 1e9:
        noise_threshold = 0.05  # Lower threshold for high frequencies
        num_imfs = 4 if num_imfs is None else num_imfs
    elif effective_freq > 500e6:
        noise_threshold = 0.08
        num_imfs = 3 if num_imfs is None else num_imfs
    
    micro_buffer_size = CHUNK_SIZE
    signal_cpu = np.array(signal).reshape(-1)
    mmap_path = get_mmap_path('emd_denoised')
    
    with managed_mmap(mmap_path, signal_cpu.dtype, signal_cpu.shape) as denoised_mmap:
        streams = [cp.cuda.Stream() for _ in range(NUM_STREAMS)]
        buffer_size = micro_buffer_size // NUM_STREAMS
        
        for i in range(0, len(signal_cpu), buffer_size):
            chunk_end = min(i + buffer_size, len(signal_cpu))
            chunk = signal_cpu[i:chunk_end]
            stream_idx = (i // buffer_size) % NUM_STREAMS
            
            try:
                with streams[stream_idx], gpu_memory_manager():
                    signal_gpu = cp.array(chunk)
                    imfs_gpu = parallel_emd_gpu(signal_gpu, max_imf=num_imfs)
                    energy_ratio = cp.sum(cp.square(imfs_gpu), axis=1)
                    energy_ratio /= cp.sum(energy_ratio)
                    
                    denoised_chunk = cp.zeros_like(signal_gpu)
                    blocks = (len(chunk) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
                    
                    emd_denoising_kernel[blocks, THREADS_PER_BLOCK](
                        imfs_gpu, energy_ratio, noise_threshold, denoised_chunk
                    )
                    
                    denoised_mmap[i:chunk_end] = cp.asnumpy(denoised_chunk)
                    denoised_mmap.flush()
                    
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
                logging.warning(f"GPU processing failed: {str(e)}. Using CPU processing.")
                denoised_chunk = cpu_emd_denoising(chunk, num_imfs, noise_threshold)
                denoised_mmap[i:chunk_end] = denoised_chunk
                denoised_mmap.flush()

        return np.array(denoised_mmap[:])


def cpu_emd_denoising(chunk, num_imfs, noise_threshold):
    logging.debug(f"CPU EMD denoising - chunk size: {len(chunk)}, num_imfs: {num_imfs}, noise_threshold: {noise_threshold}")
    
    emd = EMD()
    imfs = emd.emd(chunk, max_imf=num_imfs)
    
    imf_energies = np.sum(np.square(imfs), axis=1)
    total_energy = np.sum(imf_energies)
    energy_ratio = imf_energies / total_energy
    
    denoised_chunk = np.zeros_like(chunk)
    for i, ratio in enumerate(energy_ratio):
        if ratio > noise_threshold:
            denoised_chunk += imfs[i]
    
    logging.debug(f"CPU EMD denoising complete - denoised chunk size: {len(denoised_chunk)}")
    return denoised_chunk


def independent_component_analysis(signals, n_components=None):
    logging.debug(f"Applying ICA - n_components: {n_components}")
    logging.debug(f"Input signals shape: {signals.shape}")
    
    ica = FastICA(n_components=n_components, random_state=0)
    sources = ica.fit_transform(signals.T).T
    
    logging.debug(f"Separated sources shape: {sources.shape}")
    return sources

def ica_denoise(signal, noise_reference, n_components=None):
    logging.debug(f"Applying ICA denoising - n_components: {n_components}")
    logging.debug(f"Signal shape: {signal.shape}, Noise reference shape: {noise_reference.shape}")
    
    mixed_signals = np.vstack((signal, noise_reference))
    sources = independent_component_analysis(mixed_signals, n_components)
    
    denoised_signal = sources[0]
    logging.debug(f"Denoised signal shape: {denoised_signal.shape}")
    return denoised_signal

def matched_filter(input_signal, template):
    logging.debug(f"Applying matched filter")
    logging.debug(f"Input signal shape: {input_signal.shape}, Template shape: {template.shape}")
    
    template = template / np.linalg.norm(template)
    filtered_signal = signal.correlate(input_signal, template, mode='same')
    
    logging.debug(f"Filtered signal shape: {filtered_signal.shape}")
    return filtered_signal

def detect_peaks(filtered_signal, threshold=None):
    logging.debug(f"Detecting peaks - threshold: {threshold}")
    logging.debug(f"Filtered signal shape: {filtered_signal.shape}")
    
    if threshold is None:
        threshold = 3 * np.std(filtered_signal)
    
    peaks, _ = signal.find_peaks(filtered_signal, height=threshold)
    logging.debug(f"Number of peaks detected: {len(peaks)}")
    return peaks

def amplify_signal(data, effective_freq, factor=4):
    xp = cp.get_array_module(data)
    
    logging.debug(f"Amplifying signal - Input data shape: {xp.shape(data)}, dtype: {data.dtype}")
    logging.debug(f"Effective frequency: {effective_freq} Hz")
    logging.debug(f"Initial amplification factor: {factor}")

    if effective_freq > 1e9:  # Above 1 GHz
        factor = factor * 2
    elif effective_freq > 500e6:  # Above 500 MHz
        factor = factor * 1.5

    factor = xp.asarray(factor)
    amplified_data = xp.multiply(data, factor)

    logging.debug(f"Final amplification factor: {factor}")
    logging.debug(f"Max amplitude after amplification: {xp.max(xp.abs(amplified_data))}")

    return amplified_data


def apply_notch_filter(fft_data, effective_freq, sampling_rate, quality_factor=30):
    logging.debug(f"Starting notch filter - effective_freq: {effective_freq}, sampling_rate: {sampling_rate}")
    
    if fft_data is None or len(fft_data) == 0:
        logging.error("Empty FFT data provided to notch filter")
        return fft_data
        
    mmap_path = get_mmap_path('notch_filtered')
    # Use effective frequency for normalized calculation
    effective_freq_normalized = (effective_freq % (sampling_rate / 2)) / (sampling_rate / 2)
    b, a = signal.iirnotch(effective_freq_normalized, quality_factor)
    
    with managed_mmap(mmap_path, fft_data.dtype, fft_data.shape) as filtered_mmap:
        streams = [cp.cuda.Stream() for _ in range(NUM_STREAMS)]
        
        for i in range(0, len(fft_data), CHUNK_SIZE):
            chunk_end = min(i + CHUNK_SIZE, len(fft_data))
            fft_chunk = fft_data[i:chunk_end]
            stream_idx = (i // CHUNK_SIZE) % NUM_STREAMS
            
            try:
                with streams[stream_idx], gpu_memory_manager():
                    fft_chunk_gpu = cp.asarray(fft_chunk)
                    b_gpu, a_gpu = cp.asarray(b), cp.asarray(a)
                    
                    filtered_real = cp_signal.lfilter(b_gpu, a_gpu, cp.real(fft_chunk_gpu))
                    filtered_imag = cp_signal.lfilter(b_gpu, a_gpu, cp.imag(fft_chunk_gpu))
                    filtered_chunk = filtered_real + 1j * filtered_imag
                    
                    filtered_mmap[i:chunk_end] = cp.asnumpy(filtered_chunk)
                    filtered_mmap.flush()
                    
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
                logging.warning(f"GPU processing failed: {str(e)}. Using CPU processing.")
                filtered_real = signal.lfilter(b, a, np.real(fft_chunk))
                filtered_imag = signal.lfilter(b, a, np.imag(fft_chunk))
                filtered_chunk = filtered_real + 1j * filtered_imag
                filtered_mmap[i:chunk_end] = filtered_chunk
                filtered_mmap.flush()

        return np.array(filtered_mmap[:])

def bandpass_filter(fft_data, freq, effective_freq, effective_bandwidth):
    """GPU-accelerated bandpass filter with memory mapping and CPU fallback"""
    logging.debug(f"Starting bandpass filter - effective_freq: {effective_freq}, bandwidth: {effective_bandwidth}")
    
    mmap_path = get_mmap_path('bandpass_filtered')
    lowcut = effective_freq - effective_bandwidth / 2
    highcut = effective_freq + effective_bandwidth / 2
    
    with managed_mmap(mmap_path, fft_data.dtype, fft_data.shape) as filtered_mmap:
        streams = [cp.cuda.Stream() for _ in range(NUM_STREAMS)]
        
        for i in range(0, len(fft_data), CHUNK_SIZE):
            chunk_end = min(i + CHUNK_SIZE, len(fft_data))
            fft_chunk = fft_data[i:chunk_end]
            freq_chunk = freq[i:chunk_end]
            stream_idx = (i // CHUNK_SIZE) % NUM_STREAMS
            
            try:
                with streams[stream_idx], gpu_memory_manager():
                    fft_chunk_gpu = cp.asarray(fft_chunk)
                    freq_gpu = cp.asarray(freq_chunk).real.astype(cp.float64)
                    
                    mask = (freq_gpu >= lowcut) & (freq_gpu <= highcut)
                    filtered_chunk = cp.where(mask, fft_chunk_gpu, 0)
                    
                    filtered_mmap[i:chunk_end] = cp.asnumpy(filtered_chunk)
                    filtered_mmap.flush()
                    
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
                logging.warning(f"GPU processing failed: {str(e)}. Using CPU processing.")
                mask = (freq_chunk >= lowcut) & (freq_chunk <= highcut)
                filtered_chunk = np.where(mask, fft_chunk, 0)
                filtered_mmap[i:chunk_end] = filtered_chunk
                filtered_mmap.flush()

        return np.array(filtered_mmap[:])



def process_signal(time_domain_signal):
    logging.debug("Time domain signal data:")
    logging.debug(f"Type: {type(time_domain_signal)}")
    logging.debug(f"Shape: {time_domain_signal.shape if hasattr(time_domain_signal, 'shape') else 'No shape (not a numpy array)'}")
    logging.debug(f"Data type: {time_domain_signal.dtype if hasattr(time_domain_signal, 'dtype') else 'No dtype (not a numpy array)'}")
    
    # Print some statistics
    if isinstance(time_domain_signal, np.ndarray):
        logging.debug(f"Min value: {np.min(time_domain_signal)}")
        logging.debug(f"Max value: {np.max(time_domain_signal)}")
        logging.debug(f"Mean value: {np.mean(time_domain_signal)}")
        logging.debug(f"Standard deviation: {np.std(time_domain_signal)}")
    
    # Print first few and last few elements
    if len(time_domain_signal) > 10:
        logging.debug("First 5 elements: " + str(time_domain_signal[:5]))
        logging.debug("Last 5 elements: " + str(time_domain_signal[-5:]))
    else:
        logging.debug("All elements: " + str(time_domain_signal))
    
    # Check for NaN or infinite values
    if isinstance(time_domain_signal, np.ndarray):
        nan_count = np.isnan(time_domain_signal).sum()
        inf_count = np.isinf(time_domain_signal).sum()
        logging.debug(f"Number of NaN values: {nan_count}")
        logging.debug(f"Number of infinite values: {inf_count}")
        if nan_count > 0 or inf_count > 0:
            logging.warning("NaN or infinite values detected in time domain signal.")

    # Check for negative values
    if isinstance(time_domain_signal, np.ndarray):
        negative_count = np.sum(time_domain_signal < 0)
        logging.debug(f"Number of negative values: {negative_count}")

        if negative_count > 0:
            logging.warning("Negative values detected in time domain signal.")

    # Check for zero values
    if isinstance(time_domain_signal, np.ndarray):
        zero_count = np.sum(time_domain_signal == 0)
        logging.debug(f"Number of zero values: {zero_count}")

        if zero_count > 0:
            logging.warning("Zero values detected in time domain signal.")

@cuda.jit
def pll_kernel(input_chunk, phi, freq, fs, fmin, fmax, Kp, Ki):
    i = cuda.grid(1)
    if i > 0 and i < input_chunk.shape[0]:
        phase_error = math.atan2(input_chunk[i].imag * math.cos(phi[i-1]) - input_chunk[i].real * math.sin(phi[i-1]),
                                 input_chunk[i].real * math.cos(phi[i-1]) + input_chunk[i].imag * math.sin(phi[i-1]))
        freq[i] = freq[i-1] + Kp * phase_error + Ki * phase_error
        freq[i] = max(min(freq[i], fmax), fmin)
        phi[i] = phi[i-1] + 2 * math.pi * freq[i] / fs

def phase_locked_loop_gpu(signal, fs, effective_freq, fmin=0, fmax=None, loop_bw=0.01):
    """Optimized PLL for effective frequency ranges"""
    logging.debug(f"Applying PLL at effective frequency: {effective_freq} Hz")
    
    # Adjust loop bandwidth based on frequency
    if effective_freq > 1e9:
        loop_bw = 0.005  # Tighter loop for high frequencies
    elif effective_freq > 500e6:
        loop_bw = 0.008
    
    OPTIMAL_CHUNK_SIZE = 512 * 1024
    total_chunks = len(signal) // OPTIMAL_CHUNK_SIZE + (1 if len(signal) % OPTIMAL_CHUNK_SIZE else 0)
    result = np.zeros_like(signal)
    
    for i in range(0, len(signal), OPTIMAL_CHUNK_SIZE):
        chunk_end = min(i + OPTIMAL_CHUNK_SIZE, len(signal))
        chunk = signal[i:chunk_end]
        
        d_chunk = cp.array(chunk)
        phase = cp.zeros_like(d_chunk)
        output = cp.zeros_like(d_chunk)
        
        blocks = (len(chunk) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        pll_kernel[blocks, THREADS_PER_BLOCK](d_chunk, phase, output, loop_bw, fs, fmin, fmax, THREADS_PER_BLOCK)
        result[i:chunk_end] = cp.asnumpy(output)
        
        cp.get_default_memory_pool().free_all_blocks()
        
        logging.info(f"PLL Progress: {(i//OPTIMAL_CHUNK_SIZE)+1}/{total_chunks} chunks ({((i//OPTIMAL_CHUNK_SIZE)+1)/total_chunks*100:.1f}%)")
    
    return result


def calculate_sampling_rate(center_freq, bandwidth):
    max_freq = center_freq + (bandwidth / 2)  # Maximum frequency to capture
    sampling_rate = 2 * max_freq  # Nyquist rate
    return sampling_rate

def enhance_fft_values(fft_values):
    return np.log10(np.abs(fft_values) + 1)

def preprocess_fft_values(fft_values, kernel_size=3):
    denoised_fft_values = signal.medfilt(np.abs(fft_values), kernel_size=kernel_size)
    return enhance_fft_values(denoised_fft_values)
def advanced_signal_processing_pipeline(filtered_freq, filtered_fft, fs, center_frequency, low_cutoff, high_cutoff, rotation_angle, **kwargs):
    # Calculate effective frequencies for bandpass sampling
    nyquist_zone = int(np.ceil(center_frequency / (fs/2)))
    effective_freq = center_frequency % (fs/2)
    effective_low = low_cutoff % (fs/2)
    effective_high = high_cutoff % (fs/2)
    
    # Calculate both bandwidths
    initial_bandwidth = high_cutoff - low_cutoff
    effective_bandwidth = effective_high - effective_low
    
    logging.debug(f"Initial Bandwidth: {initial_bandwidth}")
    logging.debug(f"Effective Bandwidth: {effective_bandwidth}")
    logging.debug(f"Nyquist zone: {nyquist_zone}")
    logging.debug(f"Effective frequency range: {effective_low} to {effective_high}")

    # Extract parameters with frequency-aware defaults based on effective frequency
    wavelet = kwargs.get('wavelet', 'db1')
    wavelet_level = kwargs.get('wavelet_level', 1)
    notch_center_freq = kwargs.get('notch_center_freq', 60)
    quality_factor = kwargs.get('quality_factor', 50 if effective_freq > 1e9 else 30)
    step_size = kwargs.get('step_size', 0.05 if effective_freq > 1e9 else 0.1)
    filter_length = kwargs.get('filter_length', 20 if effective_freq > 1e9 else 10)
    Q = kwargs.get('Q', 1e-6 if effective_freq > 1e9 else 1e-5)
    R = kwargs.get('R', 0.01)
    alpha = kwargs.get('alpha', 2)
    beta = kwargs.get('beta', 0.01)
    kernel_size = kwargs.get('kernel_size', 5 if effective_freq > 1e9 else 3)
    fmin = kwargs.get('fmin', 0)
    fmax = kwargs.get('fmax', fs / 2)
    loop_bw = kwargs.get('loop_bw', 0.01)
    comb_notch_freq = kwargs.get('comb_notch_freq', 50)
    num_imfs = kwargs.get('num_imfs', None)
    noise_threshold = kwargs.get('noise_threshold', 0.1)
    noise_reference = kwargs.get('noise_reference', None)
    n_components = kwargs.get('n_components', None)
    template = kwargs.get('template', None)

    # Apply filters using effective frequencies
    filtered_fft = apply_notch_filter(filtered_fft, effective_freq, fs, quality_factor)
    time_domain_signal = np.fft.ifft(filtered_fft)
    time_domain_signal = bandpass_filter(time_domain_signal, filtered_freq, effective_freq, effective_bandwidth)
    time_domain_signal = amplify_signal(time_domain_signal, effective_freq, 4 if effective_freq > 1e9 else 2)

    
    desired_signal = kwargs.get('desired_signal', time_domain_signal)
    time_domain_signal = adaptive_filter(time_domain_signal, desired_signal, effective_freq, step_size=step_size, filter_length=filter_length)
    time_domain_signal = wavelet_denoise(time_domain_signal, effective_freq, wavelet=wavelet, level=wavelet_level)
    time_domain_signal = kalman_filter_gpu(time_domain_signal,effective_freq, Q=Q, R=R)
    
    noise_estimate = kwargs.get('noise_estimate', np.mean(time_domain_signal))
    time_domain_signal = spectral_subtraction(time_domain_signal, noise_estimate, effective_freq, alpha=alpha, beta=beta)
    time_domain_signal = median_filter(time_domain_signal, effective_freq, kernel_size=kernel_size)
    time_domain_signal = phase_locked_loop_gpu(time_domain_signal, fs, effective_freq, fmin=effective_low, fmax=effective_high, loop_bw=loop_bw)
    time_domain_signal = comb_filter(time_domain_signal, fs, effective_freq, comb_notch_freq, quality_factor)
    time_domain_signal = emd_denoising(time_domain_signal, effective_freq, num_imfs=num_imfs, noise_threshold=noise_threshold)


    if noise_reference is not None:
        time_domain_signal = ica_denoise(time_domain_signal, noise_reference, n_components=n_components)
    
    if template is not None:
        time_domain_signal = matched_filter(time_domain_signal, template)

    return time_domain_signal, filtered_freq