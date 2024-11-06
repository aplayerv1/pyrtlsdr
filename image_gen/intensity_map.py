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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@jit(nopython=True, parallel=True)
def compute_intensity(intensity_map):
    return np.sum(intensity_map, axis=0)

def process_chunk(data, start_idx, end_idx):
    chunk = data[start_idx:end_idx]
    
    if len(chunk) == 0:
        return None
    
    if chunk.ndim != 1:
        return None
    
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
        batch_size = 1000000  # Process 1 million samples at a time
        all_results = []
        
        for i in range(0, len(filtered_fft_values), batch_size):
            batch = filtered_fft_values[i:i+batch_size]
            
            if isinstance(batch, np.ndarray):
                logging.debug("Converting batch to cupy array")
                batch = cp.asarray(batch)
            
            logging.debug("Preprocessing FFT values on GPU")
            preprocessed_batch = batch
            
            chunk_size = 500  # Reduced from 1000
            num_chunks = (len(preprocessed_batch) + chunk_size - 1) // chunk_size
            logging.debug(f"Total size: {len(preprocessed_batch)}, chunk_size: {chunk_size}, num_chunks: {num_chunks}")
            
            gpu_array = cp.asnumpy(preprocessed_batch)
            results = process_data_parallel(gpu_array, chunk_size)
            all_results.extend(results)
            
            # Clean up GPU memory
            cp.get_default_memory_pool().free_all_blocks()
        
        if not all_results:
            logging.warning("No results returned from processing data")
            return None
        
        combined_intensity = np.concatenate([result[0] for result in all_results if result is not None and len(result[0]) > 0])
        
        if len(combined_intensity) == 0:
            logging.warning("No non-zero intensity data found")
            return None
        
        side_length = int(np.sqrt(len(combined_intensity)))
        intensity_map = combined_intensity[:side_length**2].reshape(side_length, side_length)
        
        intensity_map = np.abs(intensity_map)  # Convert complex values to magnitude

        logging.debug(f"Created intensity map with shape: {intensity_map.shape}")
        
        peaks, _ = find_peaks(np.ravel(intensity_map), height=0.02, distance=5)
        scaled_peaks = np.array([(p % side_length, p // side_length) for p in peaks])
        logging.debug(f"Found {len(peaks)} peaks")
        amplification_factor = 2  # Change as needed
        for x, y in scaled_peaks:
            intensity_map[y, x] *= amplification_factor
        
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
        plot_burst_detections(real_bursts, intensity_map, sampling_rate, scaled_peaks, side_length, date, time, output_dir)
        
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
            # Ensure features and clustered_events have the same number of samples
            min_samples = min(len(features), len(clustered_events))
            features = features[:min_samples]
            clustered_events = clustered_events[:min_samples]
            
            # Reshape features if necessary
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            
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

    # Check for valid intensity map
    if np.all(intensity_map == 0) or intensity_map.size == 0:
        logging.warning("Empty or all-zero intensity map. Skipping plot.")
        plt.close()
        return

    # Calculate intensity map bounds
    vmin, vmax = np.percentile(intensity_map[intensity_map > 0], [1, 99]) if np.any(intensity_map > 0) else (1e-10, 1)
    norm = LogNorm(vmin=max(vmin, 1e-10), vmax=max(vmax, vmin * 1.1)) if vmax / vmin > 10 else plt.Normalize(vmin, vmax)

    # Plot intensity map
    plt.imshow(intensity_map, cmap='viridis', norm=norm,
               extent=(0, side_length, side_length, 0), aspect='auto')
    plt.colorbar(label='Intensity')

    # Plot peaks if present
    if peaks is not None and len(peaks) > 0:
        plt.scatter(peaks % side_length, peaks // side_length, color='red', s=10, label='Peaks')

    # Plot bursts with dimension checking
    if isinstance(real_bursts, (list, tuple)):
        if len(real_bursts) >= 2 and isinstance(real_bursts[0], np.ndarray) and isinstance(real_bursts[1], np.ndarray):
            if real_bursts[0].size > 0 and real_bursts[1].size > 0:
                plt.scatter(real_bursts[0], real_bursts[1], color='white', s=20, label='Detected Bursts', edgecolors='black')

    plt.title(f"Intensity Map with Burst Detections: {date} {time}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time')
    plt.legend()

    output_path = os.path.join(output_dir, f"intensity_map_{date}_{time}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Intensity map saved to {output_path}")