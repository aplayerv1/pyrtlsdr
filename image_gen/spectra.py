from scipy.signal import medfilt
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

def enhance_fft_values(fft_values):
    # Apply logarithmic scaling to enhance FFT values
    enhanced_fft_values = np.log10(np.abs(fft_values) + 1)
    return enhanced_fft_values

def preprocess_fft_values(fft_values, kernel_size=3):
    # Denoise FFT values using median filter
    denoised_fft_values = medfilt(np.abs(fft_values), kernel_size=kernel_size)
    
    # Apply enhancement to denoised FFT values
    enhanced_fft_values = enhance_fft_values(denoised_fft_values)
    
    return enhanced_fft_values

def save_spectra(freq, fft_values, peaks, troughs, output_dir, date, time, center_freq=None, bandwidth=None):
    with tqdm(total=1, desc='Generating Spectra:') as pbar:
        try:
            os.makedirs(output_dir, exist_ok=True)
            magnitude = np.abs(fft_values)
            
            plt.figure(figsize=(14, 7))
            plt.semilogy(freq / 1e6, magnitude, label='Spectrum', color='black')  # Convert Hz to MHz
            if peaks.size > 0:
                plt.plot(freq[peaks] / 1e6, magnitude[peaks], 'ro', label='Emission Lines')
            if troughs.size > 0:
                plt.plot(freq[troughs] / 1e6, magnitude[troughs], 'bo', label='Absorption Lines')
            
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
            pbar.update(1)
            print(f"Spectra saved: {output_path}")
        except Exception as e:
            print(f"Error saving spectra: {str(e)}")
            pbar.update(1)

def save_spectra2(freq, fft_values, peaks, troughs, output_dir, date, time):
    with tqdm(total=1, desc='Generating Spectra:') as pbar:
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply logarithmic scaling and denoising
        enhanced_fft_values = preprocess_fft_values(fft_values)
        
        plt.figure(figsize=(14, 7))
        plt.plot(freq, enhanced_fft_values, label='Spectrum', color='black')
        
        if peaks.size > 0:
            plt.plot(freq[peaks], enhanced_fft_values[peaks], 'ro', label='Emission Lines')
        if troughs.size > 0:
            plt.plot(freq[troughs], enhanced_fft_values[troughs], 'bo', label='Absorption Lines')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Emission and Absorption Spectra')
        plt.legend(loc="upper right")
        plt.xlim([min(freq), max(freq)])
        plt.ylim([np.min(enhanced_fft_values), np.max(enhanced_fft_values)])
        output_path = os.path.join(output_dir, f'spectra_Emission_{date}_{time}.png')
        plt.savefig(output_path)
        plt.close()
        pbar.update(1)