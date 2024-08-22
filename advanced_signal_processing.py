import numpy as np
from scipy import signal
import pywt
from scipy.signal import lfilter, medfilt
from pyemd import emd
from sklearn.decomposition import FastICA
from scipy import ndimage
import logging
import cupy as cp
from cupyx.scipy import signal as cp_signal
import numpy as np
import cupy as cp
from numba import cuda

# Define the speed of light in meters per second
speed_of_light = 299792458  # meters per second
EARTH_ROTATION_RATE = 15  # degrees per hour
tolerance = 1e6
# Low band LO frequency in MHz
notch_freq = 9750
# Notch width in MHz
notch_width = 30
magnetic_field_strength=1
k_B = 1.38e-23 

def apply_rotation(data, rotation_angle):
    if len(data.shape) != 1:
        raise ValueError("Input data must be a 1D array.")
    data_2d = data[:, np.newaxis]
    rotated_data_2d = ndimage.rotate(data_2d, rotation_angle, reshape=False, mode='nearest')
    rotated_data = rotated_data_2d.flatten()
    return rotated_data

def remove_dc_offset(signal):
    signal = np.nan_to_num(signal)
    mean_val = np.mean(signal)
    logging.debug(f"DC Offset Mean: {mean_val}")
    return signal - mean_val

def denoise_signal(data, wavelet='db1', level=1):
    if len(data.shape) != 1:
        raise ValueError("Input data must be a 1D array.")
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(new_coeffs, wavelet)

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
    

def adaptive_filter(input_signal, desired_signal, step_size=0.1, filter_length=10):
    """
    Implement an adaptive filter using the Least Mean Squares (LMS) algorithm for complex signals.
    """
    logging.debug(f"Applying adaptive filter - step_size: {step_size}, filter_length: {filter_length}")
    logging.debug(f"Input signal shape: {input_signal.shape}, desired signal shape: {desired_signal.shape}")

    filter_coeffs = np.zeros(filter_length, dtype=np.complex128)
    output_signal = np.zeros_like(input_signal)

    for i in range(len(input_signal)):
        if i < filter_length:
            segment = input_signal[max(0, i-filter_length+1):i+1][::-1]
            output_signal[i] = np.dot(filter_coeffs[:len(segment)], segment)
        else:
            output_signal[i] = np.dot(filter_coeffs, input_signal[i-filter_length+1:i+1][::-1])
        
        error = desired_signal[i] - output_signal[i]
        update_length = min(filter_length, i+1)
        filter_coeffs[:update_length] += step_size * np.conj(error) * input_signal[max(0, i-update_length+1):i+1][::-1]
    
    logging.debug(f"Output signal shape: {output_signal.shape}")
    logging.debug(f"Final filter coefficients: {filter_coeffs}")

    return output_signal


def wavelet_denoise(signal, wavelet='db4', level=1):
    """
    Apply wavelet denoising to the input signal.
    
    Args:
    signal (array): Input signal to be denoised
    wavelet (str): Wavelet to use for decomposition
    level (int): Level of decomposition
    
    Returns:
    array: Denoised signal
    """
    # Decompose the signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Threshold the coefficients
    threshold = np.sqrt(2*np.log(len(signal)))
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
    
    # Reconstruct the signal
    denoised_signal = pywt.waverec(coeffs, wavelet)
    
    return denoised_signal

def kalman_filter(z, Q=1e-5, R=1e-2):
    """
    Apply Kalman filter to the complex-valued input signal.
    """
    logging.debug(f"Applying Kalman filter - Q: {Q}, R: {R}")
    logging.debug(f"Input signal shape: {z.shape}, dtype: {z.dtype}")

    n = len(z)
    x_hat = np.zeros(n, dtype=np.complex128)
    P = np.zeros(n, dtype=np.float64)
    K = np.zeros(n, dtype=np.complex128)

    x_hat[0] = z[0]
    P[0] = 1

    for k in range(1, n):
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        K[k] = P_minus / (P_minus + R)
        x_hat[k] = x_hat_minus + K[k] * (z[k] - x_hat_minus)
        P[k] = (1 - np.abs(K[k])) * P_minus

    logging.debug(f"Output signal shape: {x_hat.shape}, dtype: {x_hat.dtype}")
    logging.debug(f"Final Kalman gain: {K[-1]}")
    logging.debug(f"Final error covariance: {P[-1]}")

    return x_hat

def spectral_subtraction(signal, noise_estimate, alpha=2, beta=0.01):
    """
    Apply spectral subtraction to remove background noise.
    """
    logging.debug(f"Applying spectral subtraction - alpha: {alpha}, beta: {beta}")
    logging.debug(f"Input signal shape: {signal.shape}, noise estimate shape: {noise_estimate.shape}")

    signal_fft = np.fft.fft(signal)
    signal_mag = np.abs(signal_fft)
    signal_phase = np.angle(signal_fft)

    subtracted_mag = np.maximum(signal_mag**2 - alpha * noise_estimate**2, beta * signal_mag**2)**0.5

    subtracted_signal = np.fft.ifft(subtracted_mag * np.exp(1j * signal_phase))

    logging.debug(f"Output signal shape: {subtracted_signal.shape}")
    logging.debug(f"Max magnitude before subtraction: {np.max(signal_mag)}")
    logging.debug(f"Max magnitude after subtraction: {np.max(subtracted_mag)}")

    return np.real(subtracted_signal)


def median_filter(signal, kernel_size=3):
    """
    Apply median filtering to the input signal.
    
    Args:
    signal (array): Input signal
    kernel_size (int): Size of the median filter kernel
    
    Returns:
    array: Filtered signal
    """
    return medfilt(signal, kernel_size)

def phase_locked_loop(input_signal, fs, fmin, fmax, loop_bw=0.01):
    """
    Implement a phase-locked loop for frequency tracking.
    """
    logging.debug(f"Applying phase-locked loop - fs: {fs}, fmin: {fmin}, fmax: {fmax}, loop_bw: {loop_bw}")
    logging.debug(f"Input signal shape: {input_signal.shape}")

    N = len(input_signal)
    phi = np.zeros(N)
    freq = np.zeros(N)

    freq[0] = (fmin + fmax) / 2
    Kp = loop_bw * 2
    Ki = loop_bw**2 / 4

    for i in range(1, N):
        phase_error = np.angle(input_signal[i] * np.exp(-1j * phi[i-1]))
        freq[i] = freq[i-1] + Kp * phase_error + Ki * np.sum(phase_error)
        freq[i] = np.clip(freq[i], fmin, fmax)
        phi[i] = phi[i-1] + 2 * np.pi * freq[i] / fs

    logging.debug(f"Output phase shape: {phi.shape}, frequency shape: {freq.shape}")
    logging.debug(f"Final tracked frequency: {freq[-1]}")

    return phi, freq

@cuda.jit
def comb_filter_kernel(input_signal, output_signal, fs, notch_freqs, Q):
    idx = cuda.grid(1)
    if idx < input_signal.shape[0]:
        for freq in notch_freqs:
            w0 = freq / (0.5 * fs)
            alpha = np.sin(w0) / (2 * Q)
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
            b0 = (1 + np.cos(w0)) / 2
            b1 = -(1 + np.cos(w0))
            b2 = (1 + np.cos(w0)) / 2
            
            if idx >= 2:
                output_signal[idx] = (b0 * input_signal[idx] + b1 * input_signal[idx-1] + b2 * input_signal[idx-2] - 
                                      a1 * output_signal[idx-1] - a2 * output_signal[idx-2]) / a0

def comb_filter(input_signal, fs, notch_freq, Q=30):
    logging.debug(f"Applying CUDA-accelerated comb filter - fs: {fs}, notch_freq: {notch_freq}, Q: {Q}")
    
    nyq = 0.5 * fs
    notch_freqs = np.arange(notch_freq, nyq, notch_freq)
    
    d_input_signal = cuda.to_device(input_signal)
    d_output_signal = cuda.device_array_like(d_input_signal)
    d_notch_freqs = cuda.to_device(notch_freqs)
    
    threads_per_block = 256
    blocks_per_grid = (input_signal.shape[0] + threads_per_block - 1) // threads_per_block
    
    comb_filter_kernel[blocks_per_grid, threads_per_block](d_input_signal, d_output_signal, fs, d_notch_freqs, Q)
    
    filtered_signal = d_output_signal.copy_to_host()
    
    logging.debug(f"CUDA comb filter completed. Output shape: {filtered_signal.shape}")
    return filtered_signal


def empirical_mode_decomposition(signal, num_imfs=None):
    logging.debug(f"Applying EMD - num_imfs: {num_imfs}")
    logging.debug(f"Input signal shape: {signal.shape}")

    try:
        imfs = emd(signal, None, num_imfs)
        if imfs is None or len(imfs) == 0:
            # Fallback: treat the entire signal as a single IMF
            imfs = np.array([signal])
            logging.warning("EMD returned None or empty. Using original signal as single IMF.")
        
        logging.debug(f"Number of IMFs extracted: {imfs.shape[0]}")
        logging.debug(f"IMFs shape: {imfs.shape}")

        residual = signal - np.sum(imfs, axis=0)
        logging.debug(f"Residual shape: {residual.shape}")

        return imfs, residual
    except Exception as e:
        logging.error(f"Error in EMD: {str(e)}")
        # Fallback: return original signal as single IMF and zero residual
        return np.array([signal]), np.zeros_like(signal)

def emd_denoise(signal, num_imfs=None, noise_threshold=0.1):
    """
    Denoise signal using EMD by removing low-amplitude IMFs.
    """
    logging.debug(f"Applying EMD denoising - num_imfs: {num_imfs}, noise_threshold: {noise_threshold}")
    logging.debug(f"Input signal shape: {signal.shape}")

    imfs, residual = empirical_mode_decomposition(signal, num_imfs)

    significant_imfs = [imf for imf in imfs if np.max(np.abs(imf)) > noise_threshold * np.max(np.abs(signal))]
    logging.debug(f"Number of significant IMFs: {len(significant_imfs)}")

    denoised_signal = np.sum(significant_imfs, axis=0) + residual
    logging.debug(f"Denoised signal shape: {denoised_signal.shape}")
    logging.debug(f"Max amplitude before denoising: {np.max(np.abs(signal))}")
    logging.debug(f"Max amplitude after denoising: {np.max(np.abs(denoised_signal))}")

    return denoised_signal

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

def amplify_signal(data, factor=2):
    logging.debug(f"Amplifying signal - Input data shape: {np.shape(data)}, dtype: {data.dtype}")
    logging.debug(f"Amplification factor: {factor}")
    
    factor = np.asarray(factor)
    amplified_data = np.multiply(data, factor)
    
    logging.debug(f"Amplified data shape: {np.shape(amplified_data)}, dtype: {amplified_data.dtype}")
    logging.debug(f"Max amplitude after amplification: {np.max(np.abs(amplified_data))}")
    
    return amplified_data

def apply_notch_filter(fft_data, center_freq, sampling_rate, quality_factor=30):
    """
    Apply a notch filter to the FFT data to remove a specific frequency component.
    
    Args:
    fft_data (array): The FFT of the signal to be filtered.
    center_freq (float): The center frequency of the notch filter (in Hz).
    sampling_rate (float): The sampling rate of the original signal (in Hz).
    quality_factor (float): The quality factor of the notch filter.
    
    Returns:
    array: Filtered FFT data.
    """
    logging.debug(f"Applying notch filter - center_freq: {center_freq} Hz, sampling_rate: {sampling_rate} Hz, quality_factor: {quality_factor}")

    # Normalize the center frequency to the Nyquist frequency
    center_freq_normalized = center_freq / (0.5 * sampling_rate)
    
    # Check if the normalized center frequency is within the valid range [0, 1)
    if center_freq_normalized <= 0 or center_freq_normalized >= 1:
        raise ValueError(f"Normalized center frequency is out of range: {center_freq_normalized}. Must be in (0, 1).")
    
    # Design the notch filter
    b, a = signal.iirnotch(center_freq_normalized, quality_factor)
    
    # Check if filter coefficients are valid
    if np.any(np.isnan(b)) or np.any(np.isnan(a)):
        raise ValueError(f"Invalid filter coefficients: b={b}, a={a}")
    
    # Apply the notch filter to the FFT data
    filtered_fft = signal.lfilter(b, a, np.real(fft_data)) + 1j * signal.lfilter(b, a, np.imag(fft_data))
    
    logging.debug(f"Filter coefficients - b: {b}")
    logging.debug(f"Filter coefficients - a: {a}")
    logging.debug(f"Filtered FFT data shape: {filtered_fft.shape}, dtype: {filtered_fft.dtype}")
    
    return filtered_fft


def bandpass_filter(fft_data, freq, center_freq, bandwidth):
    logging.debug(f"Applying bandpass filter - center_freq: {center_freq} Hz, bandwidth: {bandwidth} Hz")
    logging.debug(f"Input FFT data shape: {fft_data.shape}, dtype: {fft_data.dtype}")
    
    mask = np.zeros_like(freq, dtype=bool)
    lowcut = center_freq - bandwidth / 2
    highcut = center_freq + bandwidth / 2
    mask[(freq >= lowcut) & (freq <= highcut)] = True
    
    logging.debug(f"Frequency range: {lowcut} Hz to {highcut} Hz")
    logging.debug(f"Number of frequencies in passband: {np.sum(mask)}")
    
    filtered_fft = np.where(mask, fft_data, 0)
    
    logging.debug(f"Filtered FFT data shape: {filtered_fft.shape}, dtype: {filtered_fft.dtype}")
    logging.debug(f"Max amplitude in passband: {np.max(np.abs(filtered_fft))}")
    
    return filtered_fft


def advanced_signal_processing_pipeline(filtered_freq, filtered_fft, fs, center_frequency, initial_bandwidth, rotation_angle, **kwargs):
    """
    Apply a comprehensive signal processing pipeline to the input signal.
    
    Args:
    filtered_freq (array): Frequencies corresponding to FFT data.
    filtered_fft (array): FFT magnitudes (or complex values).
    fs (float): Sampling frequency.
    center_frequency (float): Center frequency for bandpass filter.
    initial_bandwidth (float): Bandwidth for the bandpass filter.
    rotation_angle (float): Angle for signal rotation.
    **kwargs: Additional parameters for individual processing functions.
    
    Returns:
    array: Processed signal.
    """
    # Extract parameters from kwargs with defaults
    wavelet = kwargs.get('wavelet', 'db1')
    wavelet_level = kwargs.get('wavelet_level', 1)
    notch_center_freq = kwargs.get('notch_center_freq', 60)
    quality_factor = kwargs.get('quality_factor', 30)
    step_size = kwargs.get('step_size', 0.1)
    filter_length = kwargs.get('filter_length', 10)
    Q = kwargs.get('Q', 1e-5)
    R = kwargs.get('R', 1e-2)
    alpha = kwargs.get('alpha', 2)
    beta = kwargs.get('beta', 0.01)
    kernel_size = kwargs.get('kernel_size', 3)
    fmin = kwargs.get('fmin', 0)
    fmax = kwargs.get('fmax', fs / 2)
    loop_bw = kwargs.get('loop_bw', 0.01)
    comb_notch_freq = kwargs.get('comb_notch_freq', 50)
    num_imfs = kwargs.get('num_imfs', None)
    noise_threshold = kwargs.get('noise_threshold', 0.1)
    noise_reference = kwargs.get('noise_reference', None)
    n_components = kwargs.get('n_components', None)
    template = kwargs.get('template', None)
    
    # Apply the notch filter in the frequency domain
    filtered_fft = apply_notch_filter(filtered_fft,center_frequency,fs, quality_factor)
    
    # Convert back to time domain
    time_domain_signal = np.fft.ifft(filtered_fft)
    
    # Apply rotation
    time_domain_signal = apply_rotation(time_domain_signal, rotation_angle)
    
    # Remove DC offset
    time_domain_signal = remove_dc_offset(time_domain_signal)
    
    # Denoise
    time_domain_signal = denoise_signal(time_domain_signal, wavelet=wavelet, level=wavelet_level)
    
    # Apply bandpass filter
    time_domain_signal = bandpass_filter(time_domain_signal, fs, center_frequency, initial_bandwidth)
    
    # Amplify signal
    time_domain_signal = amplify_signal(time_domain_signal, 2)
    
    # Apply adaptive filtering
    desired_signal = kwargs.get('desired_signal', time_domain_signal)  # Use signal itself as default desired signal
    time_domain_signal = adaptive_filter(time_domain_signal, desired_signal, step_size=step_size, filter_length=filter_length)
    
    # Apply wavelet denoising
    time_domain_signal = wavelet_denoise(time_domain_signal, wavelet=wavelet, level=wavelet_level)
    
    # Apply Kalman filtering
    time_domain_signal = kalman_filter(time_domain_signal, Q=Q, R=R)
    
    # Apply spectral subtraction
    noise_estimate = kwargs.get('noise_estimate', np.mean(time_domain_signal))  # Use mean as a default noise estimate
    time_domain_signal = spectral_subtraction(time_domain_signal, noise_estimate, alpha=alpha, beta=beta)
    
    # Apply median filtering
    time_domain_signal = median_filter(time_domain_signal, kernel_size=kernel_size)
    
    # Apply phase-locked loop
    _, time_domain_signal = phase_locked_loop(time_domain_signal, fs, fmin=fmin, fmax=fmax, loop_bw=loop_bw)
    
    # Apply comb filtering
    time_domain_signal = comb_filter(time_domain_signal, fs, comb_notch_freq, quality_factor)
    
    # Apply empirical mode decomposition denoising
    time_domain_signal = emd_denoise(time_domain_signal, num_imfs=num_imfs, noise_threshold=noise_threshold)
    
    # Apply ICA-based denoising if noise reference is provided
    if noise_reference is not None:
        time_domain_signal = ica_denoise(time_domain_signal, noise_reference, n_components=n_components)
    
    # Apply matched filtering if template is provided
    if template is not None:
        time_domain_signal = matched_filter(time_domain_signal, template)
    
    return time_domain_signal