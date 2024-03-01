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

gc.enable()
# Define the speed of light in meters per second
speed_of_light = 299792458  # meters per second
delta_lambda = 1e-2  # Change in wavelength in 
lambda_0 = 0.211      # Rest wavelength in meters (500 nanometers)
EARTH_ROTATION_RATE = 15  # degrees per hour

def extract_date_from_filename(filename):
    # Extract date and time from filename using regular expression
    pattern = r'(\d{8})_(\d{6})'  # Assuming the format is YYYYMMDD_HHMMSS
    match = re.search(pattern, filename)
    header = fits.Header()
    if match:
        date = match.group(1)
        time = match.group(2)
        with fits.open(filename, ignore_missing_simple=True) as hdul:
            header['DATE'] = date
            hdu = fits.PrimaryHDU(header=header)
            hdul.append(hdu)
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

# Function to calculate Earth's rotation angle
def calculate_rotation_angle(start_time):
    # Determine elapsed time since the observation start time
    elapsed_time = datetime.now() - start_time
    # Calculate rotation angle (assume 15 degrees per hour)
    rotation_angle = int((elapsed_time.total_seconds() / 3600) * EARTH_ROTATION_RATE) % 4
    return rotation_angle


def apply_rotation(data, rotation_angle):
    # Reshape the 1D data array to 2D (a column vector)
    data_2d = data[:, np.newaxis]
    # Rotate the data array by the specified angle
    rotated_data_2d = ndimage.rotate(data_2d, rotation_angle, reshape=False, mode='nearest')   
    # Reshape the rotated 2D array back to 1D
    rotated_data = rotated_data_2d.flatten()
    
    return rotated_data

def calculate_velocity(delta_lambda, lambda_0, is_redshift=True):
    # Determine the sign based on redshift or blueshift
    sign = 1 if is_redshift else -1
    velocity = sign * (delta_lambda / lambda_0) * speed_of_light
    return velocity

# Function to apply redshift
def apply_redshift(data, delta_lambda, lambda_0):
    # Calculate velocity
    velocity = calculate_velocity(delta_lambda, lambda_0, is_redshift=True)

    # Check if velocity calculation was successful
    if velocity is not None:
        # Perform redshift operation
        if isinstance(data, np.ndarray):
            shifted_data = data * np.sqrt((1 - velocity / speed_of_light) / (1 + velocity / speed_of_light))
            return shifted_data
        else:
            print("Error: data is not a NumPy array")
            return None
    else:
        # Return original data if velocity calculation failed
        return data


# Function to apply blueshift
def apply_blueshift(data):
    # No Doppler shift for radio signals, so return the original data
    shifted_data = data
    return shifted_data

def image_reconstruction(signal_data, shift_type='none', delta_lambda=None, lambda_0=None, apply_fft=False, inverse=False):
    # Apply shift based on the shift_type
    if shift_type == 'redshift':
        shifted_signal = apply_redshift(signal_data, delta_lambda, lambda_0)
    elif shift_type == 'blueshift':
        shifted_signal = apply_blueshift(signal_data)
    elif shift_type == 'both':
        redshifted_signal = apply_redshift(signal_data, delta_lambda, lambda_0)
        shifted_signal = apply_blueshift(redshifted_signal)
    else:
        shifted_signal = signal_data  # no shift

    # Apply FFT if specified
    if apply_fft:
        # Ensure the signal is 2D before FFT
        if len(shifted_signal.shape) == 1:
            shifted_signal_2d = np.reshape(shifted_signal, (1, len(shifted_signal)))
        else:
            shifted_signal_2d = shifted_signal

        # Compute the FFT of the signal data
        fft_result = np.fft.fft2(shifted_signal_2d)

        # Reconstruct the image from the complex FFT result
        reconstructed_image = np.fft.ifft2(fft_result).real

        # Apply inverse FFT if specified
        if inverse:
            # Compute the inverse FFT to reconstruct the original signal
            reconstructed_signal_2d = np.fft.ifft2(fft_result).real

            # Flatten the 2D reconstructed signal into a 1D array
            reconstructed_signal = np.ravel(reconstructed_signal_2d)

            return reconstructed_signal
        else:
            return reconstructed_image
    else:
        return shifted_signal

def save_visualized_image2(reconstructed_image, output_dir, date, time, shift_type='none'):
  with tqdm(total=1, desc='Generating Reconstructed Image:') as pbar:

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with the extracted date, time, and shift type
    image_filename = os.path.join(output_dir, f'reconstructed_image_{date}_{time}_{shift_type}.png')

    # Ensure that the reconstructed image has a 2D shape
    if len(reconstructed_image.shape) == 1:
        # Dynamically determine the dimensions based on factors
        total_elements = len(reconstructed_image)
        sqrt_elements = int(np.sqrt(total_elements))

        # Find the width and height for reshaping
        for i in range(sqrt_elements, 0, -1):
            if total_elements % i == 0:
                width = i
                height = total_elements // i
                break

        # Reshape the 1D array to a 2D array
        reconstructed_image = np.reshape(reconstructed_image, (height, width))

    # Visualize and save the reconstructed image
    plt.figure(figsize=(10, 8))
    plt.imshow(reconstructed_image, cmap='gray')  # Adjust cmap as per your image data
    plt.colorbar(label='Intensity')
    plt.title(f'Reconstructed Image ({shift_type.capitalize()} Shift)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(image_filename, bbox_inches='tight')
    plt.close()

    pbar.update(1)


def compute_spectrogram(samples):
    spectrogram, frequencies, times, _ = plt.specgram(samples, Fs=4.0e6)
    return spectrogram, frequencies, times

def save_plots_to_png(spectrogram, signal_strength, frequencies, times, png_location, date, time):
    with tqdm(total=1, desc='Generating Spectrogram & Signal_Strength: ') as pbar:
        try:
            # Plot and save spectrogram
            plt.figure()
            plt.imshow(spectrogram.T, aspect='auto', cmap='viridis', origin='lower',
                       extent=[times[0], times[-1], frequencies[0], frequencies[-1]])
            plt.title('Spectrogram')
            plt.colorbar(label='Intensity')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.savefig(f'{png_location}/spectrogram_{date}_{time}.png')
            plt.close()

            frequency_bin_index = 0
            signal_strength_frequency_bin = signal_strength[frequency_bin_index, :]

            plt.figure()
            plt.plot(signal_strength_frequency_bin)
            plt.title(f'Signal Strength (Frequency Bin {frequency_bin_index})')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.savefig(f'{png_location}/signal_strength_frequency_{frequency_bin_index}_{date}_{time}.png')
            plt.close()

            pbar.update(1)  # Update progress bar

        except Exception as e:
            print(f"Error occurred while saving plots: {e}")

def preprocess_data(data):
    # Replace negative values with zeros
    data[data < 0] = 0
    return data

def bandpass_filter(signal_data, sampling_rate, center_frequency, bandwidth=400e6, filter_order=5):
    preprocessed_signal = preprocess_data(signal_data)
    # Define the Nyquist frequency
    nyquist_rate = sampling_rate / 2
    # Define the normalized cutoff frequencies for the bandpass filter
    low_cutoff = ((center_frequency - 10e6) - bandwidth / 2) / nyquist_rate  # Adjusted for LNB frequency offset
    high_cutoff = ((center_frequency - 10e6) + bandwidth / 2) / nyquist_rate  # Adjusted for LNB frequency offset

    # Normalize the cutoff frequencies
    low_cutoff_normalized = low_cutoff / nyquist_rate
    high_cutoff_normalized = high_cutoff / nyquist_rate

    # Apply bandpass filter
    b, a = signal.butter(filter_order, [low_cutoff_normalized, high_cutoff_normalized], 'bandpass')
    sos = butter(filter_order, [low_cutoff, high_cutoff], btype='bandpass', output='sos', fs=sampling_rate)
    signal_data_filtered = sosfiltfilt(sos, preprocessed_signal)
    return signal_data_filtered

def apply_filter_and_enhancement(data):
    # Apply Gaussian smoothing to reduce noise
    smoothed_data = gaussian_filter(data, sigma=1)

    # Apply contrast stretching for enhancement
    p_low, p_high = np.percentile(smoothed_data, (5, 95))  # Adjust percentiles as needed
    stretched_data = exposure.rescale_intensity(smoothed_data, in_range=(p_low, p_high))

    return stretched_data



def generate_heatmap(data, output_dir, date, time, shift_type=None):
    with tqdm(total=1, desc='Generating Heatmap with Filtering and Enhancement') as pbar:
        # Apply redshift or blueshift if specified
        if shift_type == 'redshift':
            redshifted_data = apply_redshift(data, delta_lambda, lambda_0)
            data = redshifted_data
        elif shift_type == 'blueshift':
            blueshifted_data = apply_blueshift(data)
            data = blueshifted_data
        elif shift_type == 'both':
            redshifted_data = apply_redshift(data, delta_lambda, lambda_0)
            blueshifted_data = apply_blueshift(redshifted_data)
            data = blueshifted_data
        elif shift_type == 'none':
            pass  # No shift applied

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
        plt.title(f'Heatmap of RTL-SDR Data with Filtering and Enhancement ({shift_type.capitalize()} Shift)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Generate filename with the extracted date
        heatmap_filename = os.path.join(output_dir, f'heatmap_{date}_{time}_{shift_type}.png')
        plt.savefig(heatmap_filename, bbox_inches='tight')
        plt.close()

        pbar.update(1)  # Update progress bar


def generate_frequency_spectrum(data, output_dir, date, time, sampling_rate, shift_type='none', delta_lambda=None, lambda_0=None):
    with tqdm(total=1, desc='Generating Frequency Spectrum') as pbar:
        # Apply shift based on the shift_type
        if shift_type == 'redshift':
            shifted_data = apply_redshift(data, delta_lambda, lambda_0)
        elif shift_type == 'blueshift':
            shifted_data = apply_blueshift(data)
        elif shift_type == 'both':
            redshifted_data = apply_redshift(data, delta_lambda, lambda_0)
            shifted_data = apply_blueshift(redshifted_data)
        else:
            shifted_data = data  # no shift

        # Perform Fast Fourier Transform (FFT) to obtain the frequency spectrum
        spectrum = np.fft.fft(shifted_data)

        # Calculate the frequency range
        freq_range = np.fft.fftfreq(len(shifted_data), d=1/sampling_rate)

        # Plot the frequency spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(freq_range, np.abs(spectrum))
        plt.title(f'Frequency Spectrum of RTL-SDR Data ({shift_type.capitalize()} Shift)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)

        # Save the frequency spectrum plot
        spectrum_filename = os.path.join(output_dir, f'frequency_spectrum_{date}_{time}_{shift_type}.png')
        plt.savefig(spectrum_filename, bbox_inches='tight')
        plt.close()
        pbar.update(1)  # Update progress bar


def analyze_signal_strength(data, output_dir, date, time, shift_type='none'):
    with tqdm(total=1, desc='Analyzing Signal Strength') as pbar:
        # Apply shift based on the shift_type
        if shift_type == 'redshift':
            shifted_data = apply_redshift(data, delta_lambda, lambda_0)
        elif shift_type == 'blueshift':
            shifted_data = apply_blueshift(data)
        elif shift_type == 'both':
            redshifted_data = apply_redshift(data, delta_lambda, lambda_0)
            shifted_data = apply_blueshift(redshifted_data)
        else:
            shifted_data = data  # no shift

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
        plt.title(f'Histogram of Signal Strength ({shift_type.capitalize()} Shift)')
        plt.legend()
        plt.grid(True)

        # Save the histogram plot
        histogram_filename = os.path.join(output_dir, f'signal_strength_histogram_{date}_{time}_{shift_type}.png')
        plt.savefig(histogram_filename, bbox_inches='tight')
        plt.close()

        # # Save the statistical analysis results
        # analysis_results = f"Mean Signal Strength: {mean_strength:.2f}\n"
        # analysis_results += f"Standard Deviation of Signal Strength: {std_dev_strength:.2f}\n"
        # analysis_results_filename = os.path.join(output_dir, f'signal_strength_analysis_{date}_{time}_{shift_type}.txt')
        # with open(analysis_results_filename, 'w') as f:
        #     f.write("Signal Strength Analysis Results:\n")
        #     f.write(analysis_results)

        pbar.update(1)  # Update progress bar


def generate_preprocessed_heatmap(preprocessed_data, output_dir, date, time, shift_type='none'):
    with tqdm(total=1, desc='Generating Preprocessed Heatmap') as pbar:
        # Apply shift based on the shift_type
        if shift_type == 'redshift':
            shifted_data = apply_redshift(preprocessed_data, delta_lambda, lambda_0)
        elif shift_type == 'blueshift':
            shifted_data = apply_blueshift(preprocessed_data)
        elif shift_type == 'both':
            redshifted_data = apply_redshift(preprocessed_data, delta_lambda, lambda_0)
            shifted_data = apply_blueshift(redshifted_data)
        else:
            shifted_data = preprocessed_data  # no shift

        # Determine the heatmap dimensions based on the size of the shifted data
        num_samples = len(shifted_data)
        num_columns = int(np.sqrt(num_samples))  # Adjust to sqrt for more square-like heatmap
        num_rows = int(np.ceil(num_samples / num_columns))  # Adjust to ceil for any remaining rows

        # Determine the size of the heatmap array
        heatmap_size = num_rows * num_columns

        # Pad the data with zeros if necessary to ensure the reshaping works
        padded_data = np.pad(shifted_data, (0, heatmap_size - num_samples), mode='constant')

        # Reshape the padded data into a 2D array for the heatmap
        heatmap_data = padded_data.reshape((num_rows, -1))

        # Create and save the heatmap image
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(label='Intensity')
        plt.title(f'Preprocessed Heatmap of RTL-SDR Data ({shift_type.capitalize()} Shift)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Generate filename with the extracted date
        heatmap_filename = os.path.join(output_dir, f'preprocessed_heatmap_{date}_{time}_{shift_type}.png')
        plt.savefig(heatmap_filename, bbox_inches='tight')
        plt.close()

        pbar.update(1)  # Update progress bar


def remove_lnb_offset(signal_data, sampling_frequency, lnb_offset_frequency):
    nyquist = sampling_frequency / 2
    lnb_normalized_frequency = lnb_offset_frequency / nyquist
    if lnb_normalized_frequency >= 1:
        lnb_normalized_frequency = 0.99
    elif lnb_normalized_frequency <= 0:
        lnb_normalized_frequency = 0.01
    b, a = signal.butter(5, lnb_normalized_frequency, btype='high')
    filtered_data = signal.filtfilt(b, a, signal_data)
    return filtered_data


def save_visualized_image(reconstructed_image, output_dir, date, time, shift_type='none'):
  with tqdm(total=1, desc='Generating Reconstructed Image:') as pbar:  
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with the extracted date, time, and shift type
    image_filename = os.path.join(output_dir, f'reconstructed_image_{date}_{time}_{shift_type}.png')
    # Determine the dimensions based on the length of the 1D array
    total_elements = len(reconstructed_image)
    sqrt_elements = int(np.sqrt(total_elements))

    # Find the width and height for reshaping
    for i in range(sqrt_elements, 0, -1):
        if total_elements % i == 0:
            width = i
            height = total_elements // i
            break
    reconstructed_image = reconstructed_image.reshape((height, width))
    # Visualize and save the reconstructed image
    plt.figure(figsize=(10, 8))
    plt.imshow(reconstructed_image, cmap='gray')  # Adjust cmap as per your image data
    plt.colorbar(label='Intensity')
    plt.title(f'Reconstructed Image ({shift_type.capitalize()} Shift)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(image_filename, bbox_inches='tight')
    plt.close()

    pbar.update(1)


def main(args):
    # Extract date from the input filename
    filename = os.path.basename(args.input)
    date, time = extract_date_from_filename(filename)
    start_time_begin = tm.time()
    spectrogram_data, signal_strength_data = [],[]
    # Extract observation start time from FITS header
    start_time = extract_observation_start_time(args.input)
    # Calculate Earth's rotation angle
    rotation_angle = calculate_rotation_angle(start_time)

    if date:
        # Read the data from the FITS file
        hdul = fits.open(args.input, ignore_missing_simple=True)
        data = hdul[0].data
        hdul.close()
        rotated_data = apply_rotation(data, rotation_angle)
        
        # Create the output directory if it does not exist
        os.makedirs(args.output, exist_ok=True)
        
        # Preprocess the data
        binary_data_no_lnb = remove_lnb_offset(rotated_data,args.sampling_rate,args.lnb_offset)
       
        #Generate heatmap
        generate_heatmap(binary_data_no_lnb, args.output, date, time, shift_type='redshift')
        generate_heatmap(binary_data_no_lnb, args.output, date, time, shift_type='blueshift')
        generate_heatmap(binary_data_no_lnb, args.output, date, time, shift_type='both')
        generate_heatmap(binary_data_no_lnb, args.output, date, time, shift_type='none')

        preprocessed_data = preprocess_data(binary_data_no_lnb)

        generate_preprocessed_heatmap(preprocessed_data, args.output, date, time, shift_type='redshift')
        generate_preprocessed_heatmap(preprocessed_data, args.output, date, time, shift_type='blueshift')
        generate_preprocessed_heatmap(preprocessed_data, args.output, date, time, shift_type='both')
        generate_preprocessed_heatmap(preprocessed_data, args.output, date, time, shift_type='none')

        # Analyze signal strength
        analyze_signal_strength(binary_data_no_lnb, args.output, date, time, shift_type='redshift')
        analyze_signal_strength(binary_data_no_lnb, args.output, date, time, shift_type='blueshift')
        analyze_signal_strength(binary_data_no_lnb, args.output, date, time, shift_type='both')
        analyze_signal_strength(binary_data_no_lnb, args.output, date, time, shift_type='none')

        # Generate frequency spectrum
        generate_frequency_spectrum(binary_data_no_lnb, args.output, date, time, args.sampling_rate, shift_type='redshift', delta_lambda=delta_lambda, lambda_0=lambda_0)
        generate_frequency_spectrum(binary_data_no_lnb, args.output, date, time, args.sampling_rate, shift_type='blueshift', delta_lambda=delta_lambda, lambda_0=lambda_0)
        generate_frequency_spectrum(binary_data_no_lnb, args.output, date, time, args.sampling_rate, shift_type='both', delta_lambda=delta_lambda, lambda_0=lambda_0)
        generate_frequency_spectrum(binary_data_no_lnb, args.output, date, time, args.sampling_rate, shift_type='none', delta_lambda=delta_lambda, lambda_0=lambda_0)


        reconstructed_image = image_reconstruction(preprocessed_data, shift_type='redshift', delta_lambda=delta_lambda, lambda_0=lambda_0)
        save_visualized_image(reconstructed_image, args.output, date, time, shift_type='redshift')
        reconstructed_image = image_reconstruction(preprocessed_data, shift_type='blueshift', delta_lambda=delta_lambda, lambda_0=lambda_0)
        save_visualized_image(reconstructed_image, args.output, date, time, shift_type='blueshift')
        reconstructed_image = image_reconstruction(preprocessed_data, shift_type='both', delta_lambda=delta_lambda, lambda_0=lambda_0)
        save_visualized_image(reconstructed_image, args.output, date, time, shift_type='both')
        reconstructed_image = image_reconstruction(preprocessed_data, shift_type='none', delta_lambda=delta_lambda, lambda_0=lambda_0)
        save_visualized_image(reconstructed_image, args.output, date, time, shift_type='none')
        reconstructed_image_fft = image_reconstruction(preprocessed_data, shift_type='none', delta_lambda=delta_lambda, lambda_0=lambda_0, apply_fft=True, inverse=True)
        save_visualized_image2(reconstructed_image_fft, args.output, date, time, shift_type='FFT')
        reconstructed_image = image_reconstruction(preprocessed_data, shift_type='none', delta_lambda=delta_lambda, lambda_0=lambda_0, apply_fft=False, inverse=True)
        save_visualized_image2(reconstructed_image, args.output, date, time, shift_type='I')

        apfe = apply_filter_and_enhancement(binary_data_no_lnb)

        spectrogram, frequencies, times = compute_spectrogram(apfe)      
        signal_strength = np.abs(spectrogram)


        spectrogram_data.append(spectrogram)
        signal_strength_data.append(signal_strength)

        spectrogram_data = np.array(spectrogram_data)
        signal_strength_data = np.array(signal_strength_data)

        # Now, you can call the function save_plots_to_png to save the plots
        save_plots_to_png(spectrogram, signal_strength, frequencies, times, args.output, date, time)

        end_time = tm.time()
        total_time = (end_time - start_time_begin)/60
        print(f"Total time taken: {total_time} minutes")
        
    else:
        print("Unable to extract date from the filename.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process RTL-SDR binary data and generate heatmap and signal strength plots.')
    parser.add_argument('-i', '--input', type=str, help='Path to RTL-SDR binary file')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory for PNG files (default: output)')
    parser.add_argument('-s', '--sampling_rate', type=float, default=4.0e6, help='Sampling rate in Hz (default: 2.4e6)')
    parser.add_argument('-c', '--center_frequency', type=float, default=1420.40e6, help='Center frequency in Hz (default: 1420.30e6)')
    parser.add_argument('-l', '--lnb-offset', type=float, default=9750e6, help='LNB offset frequency in Hz')
    parser.add_argument('-g', '--gain-factor', type=float, default=1.0, help='Digital gain factor')
    args = parser.parse_args()

    main(args)
    
gc.collect()