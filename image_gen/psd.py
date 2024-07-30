import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import concurrent.futures

def enhance_fft_values(fft_values):
    return np.log10(np.abs(fft_values) + 1)

def preprocess_fft_values(fft_values, kernel_size=3):
    denoised_fft_values = signal.medfilt(np.abs(fft_values), kernel_size=kernel_size)
    return enhance_fft_values(denoised_fft_values)

def calculate_and_save_psd(freq, fft_values, sampling_rate, output_dir, date, time, center_frequency, bandwidth, low_cutoff, high_cutoff):
    logging.info(f"Starting PSD calculation for {date} {time}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        logging.debug(f"Frequency range: {freq.min()} to {freq.max()} Hz")
        logging.debug(f"FFT values shape: {fft_values.shape}")
        
        # Apply preprocessing to FFT values
        preprocessed_fft_values = preprocess_fft_values(fft_values)
        
        def process_segment(segment):
            try:
                f, psd_segment = signal.welch(segment, fs=sampling_rate, nperseg=1024, scaling='density')
                return f, psd_segment
            except Exception as e:
                logging.error(f"Error processing segment: {e}")
                return None, None

        # Determine chunk size for parallel processing
        chunk_size = min(len(preprocessed_fft_values) // os.cpu_count(), 1024)
        chunks = [preprocessed_fft_values[i:i + chunk_size] for i in range(0, len(preprocessed_fft_values), chunk_size)]
        
        logging.info(f"Processing {len(chunks)} chunks in parallel")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_segment, chunks))
        
        # Filter out None results due to errors
        results = [result for result in results if result[0] is not None]

        if not results:
            logging.error("No valid PSD segments were computed.")
            return
        
        f = results[0][0]  # Frequency array is the same for all segments
        psd_segments = [result[1] for result in results]

        # Find the minimum length among all segments
        min_length = min(len(segment) for segment in psd_segments)

        # Truncate all segments to the minimum length
        psd_segments_truncated = [segment[:min_length] for segment in psd_segments]

        # Calculate the mean of the truncated segments
        psd = np.mean(psd_segments_truncated, axis=0)
        
        logging.debug(f"PSD shape after parallel processing: {psd.shape}")
        logging.debug(f"PSD min: {np.min(psd)}, max: {np.max(psd)}, mean: {np.mean(psd)}")
        
        # Convert PSD to dB scale
        psd_db = 10 * np.log10(np.maximum(psd, 1e-10))
        
        logging.info("Generating PSD plot")
        with tqdm(total=1, desc='Generating PSD Plot:') as pbar:
            plt.figure(figsize=(12, 6))
            plt.semilogx(f[:min_length], psd_db)
            plt.grid(True)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power/Frequency (dB/Hz)')
            plt.title(f'Power Spectral Density (Welch Method)\n{date} {time}')

            # Adjust the x-axis limits based on low_cutoff and high_cutoff
            plt.xlim(low_cutoff, high_cutoff)

            # Add vertical lines for cutoff frequencies
            plt.axvline(x=low_cutoff, color='g', linestyle='--', label='Low Cutoff')
            plt.axvline(x=high_cutoff, color='b', linestyle='--', label='High Cutoff')

            # Add vertical lines for bandwidth
            lower_bound = max(f[0], center_frequency - bandwidth / 2)
            upper_bound = min(f[min_length-1], center_frequency + bandwidth / 2)
            plt.axvline(x=lower_bound, color='r', linestyle='--', label='Bandwidth')
            plt.axvline(x=upper_bound, color='r', linestyle='--')

            plt.legend()
            psd_filename = os.path.join(output_dir, f'psd_welch_{date}_{time}.png')
            plt.savefig(psd_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            pbar.update(1)
        
        logging.info(f"PSD plot (Welch method) saved: {psd_filename}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.exception("Detailed traceback:")

    logging.info(f"PSD calculation completed for {date} {time}")
