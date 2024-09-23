import logging
from scipy.signal import medfilt
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import tempfile

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Adjust rcParams to handle large plots
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['path.simplify_threshold'] = 1.0

def clip_indices(indices, max_value):
    return np.clip(indices, 0, max_value - 1)

def enhance_fft_values(fft_values):
    return np.log10(np.abs(fft_values) + 1)

def preprocess_fft_values(fft_values, kernel_size=3):
    denoised_fft_values = medfilt(np.abs(fft_values), kernel_size=kernel_size)
    return enhance_fft_values(denoised_fft_values)

def process_chunk(chunk):
    return preprocess_fft_values(chunk)

def save_spectra(freq, fft_values, peaks, troughs, output_dir, date, time, center_freq=None, bandwidth=None):
    with tqdm(total=1, desc='Generating Spectra:') as pbar:
        try:
            os.makedirs(output_dir, exist_ok=True)
            min_length = min(len(freq), len(fft_values))
            freq = freq[:min_length]
            fft_values = fft_values[:min_length]
            # Create memory-mapped array for FFT values
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            mmap_fft = np.memmap(temp_file.name, dtype='float32', mode='w+', shape=fft_values.shape)
            mmap_fft[:] = fft_values[:]

            # Process FFT values in parallel
            chunk_size = len(mmap_fft) // os.cpu_count()
            chunks = [mmap_fft[i:i+chunk_size] for i in range(0, len(mmap_fft), chunk_size)]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                magnitude_chunks = list(executor.map(np.abs, chunks))

            magnitude = np.concatenate(magnitude_chunks)

            plt.figure(figsize=(14, 7))
            plt.semilogy(freq / 1e6, magnitude, label='Spectrum', color='black')
            
            valid_peaks = clip_indices(peaks.astype(int), len(freq))
            valid_troughs = clip_indices(troughs.astype(int), len(freq))
            
            if valid_peaks.size > 0:
                plt.plot(freq[valid_peaks] / 1e6, magnitude[valid_peaks], 'ro', label='Emission Lines')
            if valid_troughs.size > 0:
                plt.plot(freq[valid_troughs] / 1e6, magnitude[valid_troughs], 'bo', label='Absorption Lines')
            
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Magnitude (log scale)')
            plt.title(f'Emission and Absorption Spectra\n{date} {time}')
            if center_freq and bandwidth:
                plt.title(f'Emission and Absorption Spectra\n{date} {time}\nCenter: {center_freq/1e6:.2f} MHz, BW: {bandwidth/1e6:.2f} MHz')
            
            plt.legend(loc="upper right")
            plt.xlim([min(freq) / 1e6, max(freq) / 1e6])
            
            output_path = os.path.join(output_dir, f'spectra_{date}_{time}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Clean up temporary file
            os.unlink(temp_file.name)

            pbar.update(1)
            logging.info(f"Spectra saved: {output_path}")
        except Exception as e:
            logging.error(f"Error saving spectra: {str(e)}")
            pbar.update(1)

def save_enhanced_spectra(freq, fft_values, peaks, troughs, output_dir, date, time):
    logging.info(f"Starting enhanced spectra generation for {date} {time}")
    with tqdm(total=1, desc='Generating Enhanced Spectra:') as pbar:
        try:
            os.makedirs(output_dir, exist_ok=True)
            min_length = min(len(freq), len(fft_values))
            freq = freq[:min_length]
            fft_values = fft_values[:min_length]
            # Create memory-mapped array for FFT values
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            mmap_fft = np.memmap(temp_file.name, dtype='float32', mode='w+', shape=fft_values.shape)
            mmap_fft[:] = fft_values[:]

            # Process FFT values in parallel
            chunk_size = len(mmap_fft) // os.cpu_count()
            chunks = [mmap_fft[i:i+chunk_size] for i in range(0, len(mmap_fft), chunk_size)]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                enhanced_chunks = list(executor.map(process_chunk, chunks))

            enhanced_fft_values = np.concatenate(enhanced_chunks)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(freq / 1e6, enhanced_fft_values, label='Spectrum', color='black')
            
            valid_peaks = clip_indices(np.round(peaks).astype(int), len(freq))
            valid_troughs = clip_indices(np.round(troughs).astype(int), len(freq))
            
            def plot_peaks():
                if valid_peaks.size > 0:
                    ax.plot(freq[valid_peaks] / 1e6, enhanced_fft_values[valid_peaks], 'ro', label='Emission Lines')
                logging.debug(f"Plotted {valid_peaks.size} emission lines")

            def plot_troughs():
                if valid_troughs.size > 0:
                    ax.plot(freq[valid_troughs] / 1e6, enhanced_fft_values[valid_troughs], 'bo', label='Absorption Lines')
                logging.debug(f"Plotted {valid_troughs.size} absorption lines")

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(plot_peaks)
                executor.submit(plot_troughs)

            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title('Enhanced Emission and Absorption Spectra')
            ax.legend(loc="upper right")
            ax.set_xlim([min(freq) / 1e6, max(freq) / 1e6])
            ax.set_ylim([np.min(enhanced_fft_values), np.max(enhanced_fft_values)])
            
            output_path = os.path.join(output_dir, f'enhanced_spectra_Emission_{date}_{time}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Clean up temporary file
            os.unlink(temp_file.name)
            
            pbar.update(1)
            logging.info(f"Enhanced spectra saved: {output_path}")
        except Exception as e:
            logging.error(f"Error saving enhanced spectra: {str(e)}")
            pbar.update(1)
