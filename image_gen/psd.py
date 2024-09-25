import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import concurrent.futures

def calculate_and_save_psd(freq, fft_values, sampling_rate, output_dir, date, time, center_frequency, bandwidth, low_cutoff, high_cutoff):
    logging.info(f"Starting PSD calculation for {date} {time}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        logging.debug(f"Frequency range: {freq.min()} to {freq.max()} Hz")
        logging.debug(f"FFT values shape: {fft_values.shape}")
        
        # Ensure freq and fft_values have the same length
        min_length = min(len(freq), len(fft_values))
        freq = freq[:min_length]
        fft_values = fft_values[:min_length]
        
        logging.debug(f"Adjusted lengths - freq: {len(freq)}, fft_values: {len(fft_values)}")
        
        def process_segment(segment):
            try:
                f, psd_segment = signal.welch(segment, fs=sampling_rate, nperseg=1024, scaling='density')
                logging.debug(f"Processed segment. Shape: {psd_segment.shape}")
                return f, psd_segment
            except Exception as e:
                logging.error(f"Error processing segment: {e}")
                return None, None

        # Determine chunk size for parallel processing
        chunk_size = min(len(fft_values) // os.cpu_count(), 1024)
        chunks = [fft_values[i:i + chunk_size] for i in range(0, len(fft_values), chunk_size)]
        
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

            # Handling cutoff frequencies
            if low_cutoff <= 0:
                logging.warning(f"Low cutoff {low_cutoff} is non-positive. Setting to minimum positive value (1 Hz).")
                low_cutoff = 1  # Set to a minimum positive value

            if high_cutoff <= 0:
                logging.warning(f"High cutoff {high_cutoff} is non-positive. Setting to minimum positive value (1 Hz).")
                high_cutoff = max(low_cutoff + 1, 1)  # Ensure high_cutoff is greater than low_cutoff

            # Ensure high_cutoff is greater than low_cutoff
            if high_cutoff <= low_cutoff:
                logging.warning(f"High cutoff {high_cutoff} is less than or equal to low cutoff {low_cutoff}. Adjusting high_cutoff.")
                high_cutoff = low_cutoff + 1  # Ensure there's a valid range

            # Now you can safely apply xlim to the plot
            plt.xlim(max(0, low_cutoff), high_cutoff)

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
