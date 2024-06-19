import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
from scipy import signal
from tqdm import tqdm
import pywt
import time as tm
from scipy.signal import butter, sosfiltfilt
from astropy.io import fits
import cv2
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta
from scipy import ndimage
from skimage import exposure
import gc
import warnings
gc.enable()


# Suppress the specific warning
# Define the speed of light in meters per second

def extract_date_from_filename(filename):
    # Extract date and time from filename using regular expression
    pattern = r'(\d{8})_(\d{6})'  # Assuming the format is YYYYMMDD_HHMMSS
    match = re.search(pattern, filename)
    if match:
        date = match.group(1)
        time = match.group(2)
        return date, time
    else:
        return None, None

def extract_observation_start_time(fits_filename):
    with fits.open(fits_filename, ignore_missing_simple=True) as hdul:
        header = hdul[0].header
        # Extract observation start time from the header (adjust the keyword as per your FITS file)
        start_time_str = header['DATE']
        # Convert the string to a datetime object
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d')
    return start_time

def reconstruct_image(raw_data, date, time, output_dir):
    # Preprocess the raw data
    
    val = args.input
    val2 = val.split('/')[0]+'/'

    fft_filename0 = os.path.join(val2, raw_data)
    # Apply Fourier Transform
    fft_data = read_fft_file(fft_filename0)

    # Perform any processing in the frequency domain if needed

    # Apply Inverse Fourier Transform for image reconstruction
    reconstructed_image = np.fft.ifft2(fft_data).real  # Take the real part to get the image

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with the extracted date, time, and shift type
    image_filename = os.path.join(output_dir, f'reconstructed_image_{date}_{time}.png')

    # Visualize and save the reconstructed image
    plt.figure(figsize=(10, 8))
    plt.imshow(reconstructed_image, cmap='gray')  # Adjust cmap as per your image data
    plt.colorbar(label='Intensity')
    plt.title(f'Reconstructed Image')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(image_filename, bbox_inches='tight')
    plt.close()

    print(f"Reconstructed image saved as: {image_filename}")

def image_reconstruction(signal_data, date, time, output_dir):
    # Apply shift based on the shift_type
    shifted_signal = signal_data  # no shift

    # Calculate the total number of elements in the signal data
    total_elements = len(shifted_signal)

    # Find the nearest square root of the total elements
    sqrt_elements = int(np.sqrt(total_elements))

    # Find the closest divisor to the square root
    for i in range(sqrt_elements, 0, -1):
        if total_elements % i == 0:
            width = i
            height = total_elements // i
            break

    # Reshape the signal data to the determined dimensions
    reconstructed_image = shifted_signal.reshape((height, width))

    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with the extracted date, time, and shift type
    image_filename = os.path.join(output_dir, f'reconstructed_image_{date}_{time}.png')

    # Visualize and save the reconstructed image
    plt.figure(figsize=(10, 8))
    plt.imshow(reconstructed_image, cmap='gray')  # Adjust cmap as per your image data
    plt.colorbar(label='Intensity')
    plt.title(f'Reconstructed Image')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(image_filename, bbox_inches='tight')
    plt.close()

    print(f"Reconstructed image saved as: {image_filename}")

def compute_spectrogram(samples):
    spectrogram, frequencies, times, _ = plt.specgram(samples, Fs=2.4e6)
    return spectrogram

def preprocess_data(data):
    # Replace negative values with zeros
    data[data < 0] = 0
    return data

def apply_filter_and_enhancement(data):
    # Apply Gaussian smoothing to reduce noise
    smoothed_data = gaussian_filter(data, sigma=1)

    # Apply contrast stretching for enhancement
    p_low, p_high = np.percentile(smoothed_data, (5, 95))  # Adjust percentiles as needed
    stretched_data = exposure.rescale_intensity(smoothed_data, in_range=(p_low, p_high))

    return stretched_data

# Inside the save_plots_to_png function
def save_plots_to_png(spectrogram, signal_strength, png_location, date, time):
    with tqdm(total=1, desc='saving plots to png') as pbar:
        try:
            # Plot and save spectrogram
            plt.figure()
            # Check for zeros or negative values in the spectrogram
            if np.min(spectrogram) <= 0:
                # Add a small constant to avoid zeros or negative values
                spectrogram = np.abs(spectrogram) + 1e-10
            plt.imshow(spectrogram.T, aspect='auto', cmap='viridis', origin='lower')
            plt.title('Spectrogram')
            plt.colorbar()
            plt.savefig(f'{png_location}/spectrogram_{date}_{time}.png')
            plt.close()

            # Plot and save signal strength
            plt.figure()
            plt.plot(signal_strength)
            plt.title('Signal Strength')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.savefig(f'{png_location}/signal_strength_{date}_{time}.png')
            plt.close()

        except Exception as e:
            print(f"Error occurred while saving plots: {e}")
    pbar.update(1)

def generate_heatmap(data, output_dir, date, time):
    with tqdm(total=1, desc='Generating Heatmap with Filtering and Enhancement') as pbar:

        pepro = preprocess_data(data)
                # Normalize the data using logarithmic scaling
        normalized_data = np.log1p(pepro)
        # Apply filtering and enhancement
        filtered_data = apply_filter_and_enhancement(normalized_data)
        # Determine the heatmap dimensions based on the size of the data
        num_samples = len(normalized_data)
        num_columns = int(np.sqrt(num_samples))  # Adjust to sqrt for more square-like heatmap
        num_rows = int(np.ceil(num_samples / num_columns))  # Adjust to ceil for any remaining rows

        # Determine the size of the heatmap array
        heatmap_size = num_rows * num_columns

        # Pad the data with zeros if necessary to ensure the reshaping works
        padded_data = np.pad(filtered_data, (0, heatmap_size - num_samples), mode='constant')

        # Reshape the padded data into a 2D array for the heatmap
        heatmap_data = padded_data.reshape((num_rows, -1))

        # Create and save the heatmap image
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, cmap='coolwarm', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Intensity (log scale)')
        plt.title(f'Heatmap of RTL-SDR Data with Filtering and Enhancement')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Generate filename with the extracted date
        heatmap_filename = os.path.join(output_dir, f'heatmap_{date}_{time}.png')
        plt.savefig(heatmap_filename, bbox_inches='tight')
        plt.close()

        pbar.update(1)  # Update progress bar

def read_fft_file(fft_filename):
    frequencies = []
    fft_values_real = []
    fft_values_imag = []
    print(fft_filename)
    # Open the FFT file and read its contents
    with open(fft_filename, 'r') as file:
        # Skip the header line
        next(file)

        # Read each line and extract frequency and FFT values
        for line in file:
            # Split the line into frequency and FFT value components
            freq, fft_real, fft_imag = map(float, line.strip().split(','))

            # Append the values to their respective lists
            frequencies.append(freq)
            fft_values_real.append(fft_real)
            fft_values_imag.append(fft_imag)

    return frequencies, fft_values_real, fft_values_imag

def generate_frequency_spectrum_from_file(fft_filename, output_dir, date, time):
    with tqdm(total=1, desc='Generating Frequency Spectrum') as pbar:
        # Read FFT data from the file
        print("gfs1",fft_filename)
        val = args.input
        val2 = val.split('/')[0]+'/'

        fft_filename0 = os.path.join(val2, fft_filename)
        print("gfs2",fft_filename0)

        frequencies, fft_values_real, fft_values_imag = read_fft_file(fft_filename0)

        # Plot the frequency spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, np.abs([a + 1j * b for a, b in zip(fft_values_real, fft_values_imag)]))
        plt.title(f'Frequency Spectrum of RTL-SDR Data')
        plt.xlabel('Frequency (Hz)')  # Corrected x-axis label
        plt.ylabel('Magnitude')
        plt.grid(True)

        # Save the frequency spectrum plot
        spectrum_filename = os.path.join(output_dir, f'frequency_spectrum_{date}_{time}.png')
        plt.savefig(spectrum_filename, bbox_inches='tight')
        plt.close()
        pbar.update(1)  # Update progress bar



def analyze_signal_strength(data, output_dir, date, time):
    with tqdm(total=1, desc='Analyzing Signal Strength') as pbar:
        
        shifted_data = apply_filter_and_enhancement(data)  # no shift

        # Calculate signal strength from the data
        signal_strength = np.abs(shifted_data)

        # Perform statistical analysis
        mean_strength = np.mean(signal_strength)
        std_dev_strength = np.std(signal_strength)

        # Plot histogram of signal strength
        plt.figure(figsize=(8, 6))
        plt.hist(signal_strength, bins=50, color='blue', alpha=0.7)
        plt.axvline(mean_strength, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_strength:.2f}')
        plt.axvline(mean_strength + std_dev_strength, color='green', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_dev_strength:.2f}')
        plt.axvline(mean_strength - std_dev_strength, color='green', linestyle='dashed', linewidth=1)
        plt.xlabel('Signal Strength')
        plt.ylabel('Frequency')
        plt.title('Histogram of Signal Strength')
        plt.legend()
        plt.grid(True)

        # Save the histogram plot
        histogram_filename = os.path.join(output_dir, f'signal_strength_histogram_{date}_{time}.png')
        plt.savefig(histogram_filename, bbox_inches='tight')
        plt.close()

        # # Save the statistical analysis results
        # analysis_results = f"Mean Signal Strength: {mean_strength:.2f}\n"
        # analysis_results += f"Standard Deviation of Signal Strength: {std_dev_strength:.2f}\n"
        # analysis_results_filename = os.path.join(output_dir, f'signal_strength_analysis_{date}_{time}.txt')
        # with open(analysis_results_filename, 'w') as f:
        #     f.write("Signal Strength Analysis Results:\n")
        #     f.write(analysis_results)

        pbar.update(1)  # Update progress bar


def save_visualized_image(reconstructed_image, output_dir, date, time):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with the extracted date, time, and shift type
    image_filename = os.path.join(output_dir, f'reconstructed_image_{date}_{time}.png')

    # Visualize and save the reconstructed image
    plt.figure(figsize=(10, 8))
    plt.imshow(reconstructed_image, cmap='gray')  # Adjust cmap as per your image data
    plt.colorbar(label='Intensity')
    plt.title(f'Reconstructed Image')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(image_filename, bbox_inches='tight')
    plt.close()

    print(f"Reconstructed image saved as: {image_filename}")



def main(args):
    # Extract date from the input filename
    filename = os.path.basename(args.input)
    date, time = extract_date_from_filename(filename)
    start_time_begin = tm.time()
    spectrogram_data, signal_strength_data = [],[]
    # Extract observation start time from FITS header
    # start_time = extract_observation_start_time(args.input)
    # Calculate Earth's rotation angle
    # Extract the filename without extension
    filename_without_extension = os.path.splitext(os.path.basename(args.input))[0]

    # Create the output directory path
    output_directory = filename_without_extension

    args.output = args.output + output_directory

    print("1",output_directory)
    print("2",args.output)

    leta = args.output
    if date:
        # Read the data from the FITS file
        hdul = fits.open(args.input, ignore_missing_simple=True)
        data = hdul[0].data
        hdul.close()
        
        # Create the output directory if it does not exist
        os.makedirs(args.output, exist_ok=True)
        
       
        # Preprocess the data
        binary_data_no_lnb = data


       
       # Generate heatmap
        generate_heatmap(binary_data_no_lnb, args.output, date, time)

        # # Analyze signal strength
        analyze_signal_strength(binary_data_no_lnb, args.output, date, time)

        reconstruct_image(output_directory+'_fft.txt', date, time, args.output)
        # save_visualized_image(reconstructed_image, args.output, date, time)

        apfe = data

        spectrogram = compute_spectrogram(apfe)
        signal_strength = np.abs(apfe)

        spectrogram_data.append(spectrogram)
        signal_strength_data.append(signal_strength)

        spectrogram_data = np.array(spectrogram_data)
        signal_strength_data = np.array(signal_strength_data)

    
        save_plots_to_png(spectrogram_data[-1], signal_strength_data[-1],args.output, date, time)
        #save_plots_to_png(spectrogram_data[-1],args.output, date, time)
        print("3",output_directory)

        generate_frequency_spectrum_from_file(output_directory+'_fft.txt',args.output, date, time)



        end_time = tm.time()
        total_time = (end_time - start_time_begin)/60
        print(f"Total time taken: {total_time} minutes")
        
    else:
        print("Unable to extract date from the filename.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTL-SDR binary data and generate heatmap and signal strength plots.')
    parser.add_argument('-i', '--input', type=str, help='Path to RTL-SDR binary file')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory for PNG files (default: output)')
    parser.add_argument('-s', '--sampling_rate', type=float, default=2.4e6, help='Sampling rate in Hz (default: 2.4e6)')
    parser.add_argument('-c', '--center_frequency', type=float, default=1420.40e6, help='Center frequency in Hz (default: 1420.30e6)')
    parser.add_argument('-l', '--lnb-offset', type=float, default=9750e6, help='LNB offset frequency in Hz')
    parser.add_argument('-g', '--gain-factor', type=float, default=1.0, help='Digital gain factor')
    args = parser.parse_args()

    main(args)
    
gc.collect()