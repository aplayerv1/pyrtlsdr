import logging
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from numba import jit, prange
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import gaussian_filter
import concurrent.futures
from matplotlib.colors import LogNorm
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_fft_values(fft_values, kernel_size=3):
    logging.debug("Preprocessing FFT values on GPU")

    # Check if fft_values is a cupy array
    if not isinstance(fft_values, cp.ndarray):
        logging.error("fft_values is not a cupy array")
        raise TypeError("fft_values must be a cupy array")

    # Log the original shape of the fft_values
    logging.debug(f"Original fft_values shape: {fft_values.shape}")

    # Ensure fft_values is a 2D array before applying median
    if fft_values.ndim == 1:
        fft_values = fft_values.reshape((1, -1))  # Reshape to 2D if it's 1D
        logging.debug(f"Reshaped fft_values shape: {fft_values.shape}")

    # Compute the median along the specified axis
    denoised_fft_values = cp.median(fft_values, axis=0)
    logging.debug(f"Median computed, result shape: {denoised_fft_values.shape}")

    # Check if denoised_fft_values is a scalar and log it
    if denoised_fft_values.ndim == 0:
        logging.warning("Denoised FFT values is a scalar")
        denoised_fft_values = cp.expand_dims(denoised_fft_values, axis=0)
        logging.debug(f"Expanded denoised shape: {denoised_fft_values.shape}")

    # Apply log transformation and return the result
    result = cp.log10(denoised_fft_values + 1)
    logging.debug(f"Preprocessed result shape: {result.shape}")

    return result

@jit(nopython=True, parallel=True)
def compute_intensity(intensity_map):
    return np.sum(intensity_map, axis=0)

def process_chunk(data, start_idx, end_idx):
    chunk = data[start_idx:end_idx]
    
    if len(chunk) == 0:
        return None

    # Ensure chunk is a 1D array
    if chunk.ndim != 1:
        return None

    # Example operation
    outer_product = np.outer(chunk, np.sin(np.linspace(0, np.pi, len(chunk))))
    
    if outer_product.ndim != 2:
        return None

    intensity = np.sum(outer_product, axis=0)
    
    if intensity.ndim != 1:
        return None

    non_zero_intensity = intensity[intensity != 0]

    if len(non_zero_intensity) > 0:
        return non_zero_intensity, start_idx, end_idx
    else:
        return None

def process_data_parallel(data, chunk_size):
    num_chunks = len(data) // chunk_size
    results = []

    logging.debug(f'Starting parallel processing with {num_chunks} chunks')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_chunk, data[i*chunk_size:(i+1)*chunk_size], i*chunk_size, (i+1)*chunk_size): i for i in range(num_chunks)}
        for future in concurrent.futures.as_completed(futures):
            chunk_idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    logging.debug(f'Processed chunk {chunk_idx} with result shape: {result[0].shape}, start_idx: {result[1]}, end_idx: {result[2]}')
                else:
                    logging.debug(f'Chunk {chunk_idx} produced no non-zero results')
            except Exception as e:
                logging.error(f'Error processing future for chunk {chunk_idx}: {e}')
    return results

def create_intensity_map(filtered_freq, filtered_fft_values, sampling_rate, output_dir, date, time, temperature=2.7):
    logging.info("Starting create_intensity_map function")

    if not os.path.exists(output_dir):
        logging.debug(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir)

    try:
        if isinstance(filtered_fft_values, np.ndarray):
            logging.debug("Converting filtered_fft_values to cupy array")
            filtered_fft_values = cp.asarray(filtered_fft_values)
        else:
            logging.error("filtered_fft_values is not a numpy array")
            raise TypeError("filtered_fft_values must be a numpy array")

        if isinstance(filtered_freq, np.ndarray):
            logging.debug("Converting filtered_freq to cupy array")
            filtered_freq = cp.asarray(filtered_freq)
        else:
            logging.error("filtered_freq is not a numpy array")
            raise TypeError("filtered_freq must be a numpy array")

        logging.debug("Preprocessing FFT values on GPU")
        preprocessed_fft_values = preprocess_fft_values(filtered_fft_values)
        logging.debug("Preprocessing FFT values on GPU Done")
        chunk_size = 1000
        num_chunks = (len(filtered_freq) + chunk_size - 1) // chunk_size
        logging.debug(f"Total size: {len(filtered_freq)}, chunk_size: {chunk_size}, num_chunks: {num_chunks}")

        gpu_array = cp.asnumpy(preprocessed_fft_values)  # Convert back to numpy for processing
        results = process_data_parallel(gpu_array, chunk_size)

        if not results:
            logging.warning("No results returned from processing data")
            return None

        combined_intensity = np.concatenate([result[0] for result in results if result is not None and len(result[0]) > 0])

        if len(combined_intensity) == 0:
            logging.warning("No non-zero intensity data found")
            return None

        side_length = int(np.sqrt(len(combined_intensity)))
        intensity_map = combined_intensity[:side_length**2].reshape(side_length, side_length)

        logging.debug(f"Created intensity map with shape: {intensity_map.shape}")

        peaks = find_peaks(np.ravel(intensity_map))[0]
        logging.debug(f"Found {len(peaks)} peaks")

        logging.debug("Detecting events")
        threshold = np.mean(intensity_map) + 5 * np.std(intensity_map)
        logging.debug(f"Calculated event detection threshold: {threshold}")
        all_events = detect_events(intensity_map, threshold)

        logging.debug("Clustering events")
        clustered_events = cluster_events(all_events)

        features = np.array(all_events)
        logging.debug(f"Classifying bursts with {len(features)} features")
        real_bursts = classify_bursts(clustered_events, features)

        logging.debug("Applying Gaussian filter")
        intensity_map = gaussian_filter(intensity_map, sigma=1)

        logging.debug("Plotting burst detections")
        plot_burst_detections(real_bursts, intensity_map, sampling_rate, peaks, side_length, date, time, output_dir)

        logging.info("Intensity map creation completed")
        return intensity_map

    except Exception as e:
        logging.error(f"An error occurred while processing: {e}")
        raise

def detect_events(filtered_data, threshold):
    try:
        logging.debug("Entering detect_events")
        if isinstance(filtered_data, np.ndarray):
            events = np.where(filtered_data > threshold)
            logging.debug(f"Detected {len(events[0])} events")
            return events
        else:
            logging.error("filtered_data is not a numpy array")
            raise TypeError("filtered_data is not a numpy array")
    except Exception as e:
        logging.error(f"Error in detect_events: {e}")
        raise

def cluster_events(events):
    try:
        logging.debug("Entering cluster_events")
        if len(events[0]) > 0:
            clustering = DBSCAN(eps=3, min_samples=2).fit(np.array(events).T)
            logging.debug(f"Clustering resulted in {len(set(clustering.labels_))} clusters")
            return clustering.labels_
        return np.array([])
    except Exception as e:
        logging.error(f"Error in cluster_events: {e}")
        raise

def classify_bursts(clustered_events, features):
    try:
        logging.debug("Entering classify_bursts")
        if len(features) > 0 and len(clustered_events) > 0:
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(features, clustered_events)
            logging.debug(f"Classifier trained with {len(features)} features")
            return clf.predict(features)
        return np.array([])
    except Exception as e:
        logging.error(f"Error in classify_bursts: {e}")
        raise

def plot_burst_detections(real_bursts, intensity_map, sampling_rate, peaks, side_length, date, time, output_dir):
    
    plt.figure(figsize=(12, 10))
    plt.imshow(intensity_map, cmap='viridis', norm=LogNorm(), extent=(0, side_length, side_length, 0), aspect='auto')
    plt.colorbar(label='Intensity (log scale)')

    if len(peaks) > 0:
        plt.scatter(peaks % side_length, peaks // side_length, color='red', s=10, label='Peaks')

    if len(real_bursts) > 0 and real_bursts[0].size > 0 and real_bursts[1].size > 0:
        plt.scatter(real_bursts[0], real_bursts[1], color='white', s=20, label='Detected Bursts', edgecolors='black')

    plt.title(f"Intensity Map with Burst Detections: {date} {time}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time')
    plt.legend()

    output_path = os.path.join(output_dir, f"intensity_map_{date}_{time}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Intensity map saved to {output_path}")