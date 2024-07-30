from asyncio import futures
import logging
import os
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def enhance_fft_values(fft_values):
    return np.log10(np.abs(fft_values) + 1)

def preprocess_fft_values(fft_values, kernel_size=3):
    denoised_fft_values = signal.medfilt(np.abs(fft_values), kernel_size=kernel_size)
    return enhance_fft_values(denoised_fft_values)


def compute_intensity(intensity_map):
    try:
        logging.debug(f'Computing intensity for intensity_map with shape: {intensity_map.shape}')
        intensity = np.sum(intensity_map, axis=0)
        logging.debug(f'Computed intensity with shape: {intensity.shape}')
        return intensity
    except Exception as e:
        logging.error(f'Error in compute_intensity: {e}')
        raise

def process_chunk(data, start_idx, end_idx):
    try:
        logging.debug(f'Processing chunk from {start_idx} to {end_idx}')
        chunk = data[start_idx:end_idx]
        logging.debug(f'Chunk shape: {chunk.shape}')
        intensity_map = np.outer(chunk, np.sin(np.linspace(0, np.pi, len(chunk))))
        logging.debug(f'Computed intensity_map with shape: {intensity_map.shape}')
        intensity = compute_intensity(intensity_map)
        
        # Filter out zero values
        non_zero_intensity = intensity[intensity != 0]
        
        if len(non_zero_intensity) > 0:
            return non_zero_intensity, start_idx, end_idx
        else:
            return None  # Return None for empty chunks
    except Exception as e:
        logging.error(f'Error in process_chunk: {e}')
        raise

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

def filter_frequencies(freq_data, threshold=0.1):
    try:
        logging.debug(f'Filtering frequencies with threshold: {threshold}')
        filtered_data = freq_data[freq_data > threshold]
        logging.debug(f'Filtered frequencies shape: {filtered_data.shape}')
        return filtered_data
    except Exception as e:
        logging.error(f'Error in filter_frequencies: {e}')
        raise

def create_intensity_map(filtered_freq, filtered_fft_values, sampling_rate, output_dir, date, time, temperature=2.7):
    logging.info("Starting create_intensity_map function")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Preprocess FFT values
        preprocessed_fft_values = preprocess_fft_values(filtered_fft_values)

        chunk_size = 1000
        num_chunks = (len(filtered_freq) + chunk_size - 1) // chunk_size
        logging.debug(f"Total size: {len(filtered_freq)}, chunk_size: {chunk_size}, num_chunks: {num_chunks}")

        results = process_data_parallel(preprocessed_fft_values, chunk_size)

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
        events = np.where(filtered_data > threshold)
        logging.debug(f"Detected {len(events[0])} events")
        return events
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
    try:
        logging.debug("Entering plot_burst_detections")
        freq_bins = np.fft.fftfreq(intensity_map.shape[0], d=1/sampling_rate)
        freq_bins = np.fft.fftshift(freq_bins)

        plt.figure(figsize=(12, 10))
        plt.imshow(np.log1p(intensity_map), aspect='auto', cmap='viridis',
                   extent=[freq_bins.min(), freq_bins.max(), freq_bins.min(), freq_bins.max()])
        plt.title(f"Rayleigh-Jeans Intensity Map: {date} {time}")
        plt.colorbar(label='Log Intensity (W/m²/Hz/sr)')
        
        if len(peaks) > 0:
            plt.scatter(peaks % side_length, peaks // side_length, color='red', s=10, label='Peaks')
        
        if len(real_bursts) > 0 and real_bursts[0].size > 0 and real_bursts[1].size > 0:
            plt.scatter(real_bursts[0], real_bursts[1], color='green', s=20, label='Detected Bursts')
            plt.title(f"Rayleigh-Jeans Intensity Map with Burst Detections: {date} {time}")
        else:
            logging.info("No bursts detected in the data")
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Frequency (Hz)')
        plt.legend()

        output_path = os.path.join(output_dir, f"intensity_map_{date}_{time}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        logging.info(f"Rayleigh-Jeans intensity map saved to {output_path}")
    except Exception as e:
        logging.error(f"Error in plot_burst_detections: {e}")
        raise