import logging
from scipy import signal
from scipy.signal import medfilt
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_and_save_psd(signal_data, sampling_rate, output_dir, date, time, center_frequency, bandwidth):
    try:
        from tqdm import tqdm
        with tqdm(total=1, desc='Generating PSD Plot:') as pbar:
            os.makedirs(output_dir, exist_ok=True)
            
            # Ensure signal_data is a 1D array
            signal_data = np.asarray(signal_data).flatten()
            logging.debug(f"Signal data shape after flattening: {signal_data.shape}")
            
            # Calculate nperseg
            nperseg = min(8192, len(signal_data))
            logging.debug(f"Calculated nperseg: {nperseg}")
            
            # Calculate PSD using Welch's method
            frequencies, psd = signal.welch(signal_data, fs=sampling_rate, nperseg=nperseg, scaling='density', return_onesided=False)
            logging.debug(f"Frequencies shape: {frequencies.shape}, PSD shape: {psd.shape}")
            
            # Convert PSD to dB scale
            psd_db = 10 * np.log10(np.abs(psd) + 1e-10)
            
            # Plotting the PSD
            plt.figure(figsize=(12, 6))
            plt.semilogx(np.fft.fftshift(frequencies), np.fft.fftshift(psd_db))
            plt.grid(True)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power/Frequency (dB/Hz)')
            plt.title(f'Power Spectral Density\n{date} {time}')
            
            # Mark the bandwidth on the plot
            lower_bound = max(frequencies[0], center_frequency - bandwidth / 2)
            upper_bound = min(frequencies[-1], center_frequency + bandwidth / 2)
            plt.axvline(x=lower_bound, color='r', linestyle='--', label='Bandwidth')
            plt.axvline(x=upper_bound, color='r', linestyle='--')
            
            plt.legend()
            
            # Save the plot
            psd_filename = os.path.join(output_dir, f'psd_{date}_{time}.png')
            plt.savefig(psd_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            pbar.update(1)
        
        logging.info(f"PSD plot saved: {psd_filename}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
