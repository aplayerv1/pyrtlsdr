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

# Define the speed of light in meters per second
speed_of_light = 299792458  # meters per second
delta_lambda = 500e-9  # Change in wavelength in meters (500 nanometers)
lambda_0 = 500e-9      # Rest wavelength in meters (500 nanometers)

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
def apply_blueshift(data, delta_lambda, lambda_0):
    # Constants
    speed_of_light = 299792458  # Speed of light in meters per second

    # Calculate velocity using delta_lambda and lambda_0
    velocity = (delta_lambda / lambda_0) * speed_of_light

    # Apply blueshift formula
    shifted_data = data * np.sqrt((1 - velocity / speed_of_light) / (1 + velocity / speed_of_light))

    return shifted_data


def image_reconstruction(signal_data, shift_type='none', delta_lambda=None, lambda_0=None):
    # Apply shift based on the shift_type
    if shift_type == 'redshift':
        shifted_signal = apply_redshift(signal_data, delta_lambda, lambda_0)
    elif shift_type == 'blueshift':
        shifted_signal = apply_blueshift(signal_data, delta_lambda, lambda_0)
    elif shift_type == 'both':
        redshifted_signal = apply_redshift(signal_data, delta_lambda, lambda_0)
        shifted_signal = apply_blueshift(redshifted_signal, delta_lambda, lambda_0)
    else:
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

    return reconstructed_image




def plot_wavelet_features(features, output_dir, date, time, shift_type, delta_lambda, lambda_0):
    # Example: Plot wavelet features as a bar chart
    num_features = len(features)
    feature_labels = [f'Feature {i+1}' for i in range(num_features)]

    # Apply redshift or blueshift if specified
    if shift_type == 'redshift':
        shifted_data = apply_redshift(features, delta_lambda, lambda_0)
    elif shift_type == 'blueshift':
        shifted_data = apply_blueshift(features, delta_lambda, lambda_0)
    else:
        shifted_data = features  # No shift applied

    if shifted_data is not None:
        plt.figure(figsize=(36, 24))
        plt.bar(feature_labels, shifted_data)
        plt.xlabel('Feature')
        plt.ylabel('Value')
        plt.title(f'Wavelet Features ({shift_type.capitalize()} Shift)')
        plt.grid(True)

        # Save the plot
        wavelet_plot_filename = os.path.join(output_dir, f'wavelet_features_{date}_{time}_{shift_type}.png')
        plt.savefig(wavelet_plot_filename, bbox_inches='tight')
        plt.close()
    else:
        print(f"Error: Shifted data is None for {shift_type} shift")

def extract_wavelet_features_with_bandpass(signal_data, sampling_rate, center_frequency):
    # Apply bandpass filter to the signal data
    filtered_signal = bandpass_filter(signal_data, sampling_rate, center_frequency)

    # Decompose the filtered signal using wavelet transform
    coeffs = pywt.wavedec(filtered_signal, 'db1', level=5)  # Adjust wavelet and decomposition level as needed

    # Extract features from the wavelet coefficients
    features = []
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
        features.append(np.max(coeff))
        features.append(np.min(coeff))

    return features

def preprocess_data(signal_data):
    # Normalize the signal data to have zero mean and unit variance
    normalized_data = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    return normalized_data

def bandpass_filter(signal_data, sampling_rate, center_frequency, bandwidth=100e6, filter_order=5):
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

def generate_heatmap(data, output_dir, date, time, shift_type=None):
    with tqdm(total=1, desc='Generating Heatmap') as pbar:
        # Apply redshift or blueshift if specified
        if shift_type == 'redshift':
            redshifted_data = apply_redshift(data, delta_lambda, lambda_0)
            data = redshifted_data
        elif shift_type == 'blueshift':
            blueshifted_data = apply_blueshift(data, delta_lambda, lambda_0)
            data = blueshifted_data
        elif shift_type == 'both':
            redshifted_data = apply_redshift(data, delta_lambda, lambda_0)
            blueshifted_data = apply_blueshift(redshifted_data, delta_lambda, lambda_0)
            data = blueshifted_data
        elif shift_type == 'none':
            data = data

        # Determine the heatmap dimensions based on the size of the data
        num_samples = len(data)
        num_columns = int(np.sqrt(num_samples))  # Adjust to sqrt for more square-like heatmap
        num_rows = int(np.ceil(num_samples / num_columns))  # Adjust to ceil for any remaining rows

        # Determine the size of the heatmap array
        heatmap_size = num_rows * num_columns

        # Pad the data with zeros if necessary to ensure the reshaping works
        padded_data = np.pad(data, (0, heatmap_size - num_samples), mode='constant')

        # Reshape the padded data into a 2D array for the heatmap
        heatmap_data = padded_data.reshape((num_rows, -1))

        # Create and save the heatmap image
        plt.figure(figsize=(102, 57))
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(label='Intensity')
        plt.title(f'Heatmap of RTL-SDR Data ({shift_type.capitalize()} Shift)')
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
            shifted_data = apply_blueshift(data, delta_lambda, lambda_0)
        elif shift_type == 'both':
            redshifted_data = apply_redshift(data, delta_lambda, lambda_0)
            shifted_data = apply_blueshift(redshifted_data, delta_lambda, lambda_0)
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
            shifted_data = apply_blueshift(data, delta_lambda, lambda_0)
        elif shift_type == 'both':
            redshifted_data = apply_redshift(data, delta_lambda, lambda_0)
            shifted_data = apply_blueshift(redshifted_data, delta_lambda, lambda_0)
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
        plt.xlabel('Signal Strength')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Signal Strength ({shift_type.capitalize()} Shift)')
        plt.legend()
        plt.grid(True)

        # Save the histogram plot
        histogram_filename = os.path.join(output_dir, f'signal_strength_histogram_{date}_{time}_{shift_type}.png')
        plt.savefig(histogram_filename, bbox_inches='tight')
        plt.close()

        # Save the statistical analysis results
        analysis_results = f"Mean Signal Strength: {mean_strength:.2f}\n"
        analysis_results += f"Standard Deviation of Signal Strength: {std_dev_strength:.2f}\n"
        analysis_results_filename = os.path.join(output_dir, f'signal_strength_analysis_{date}_{time}_{shift_type}.txt')
        with open(analysis_results_filename, 'w') as f:
            f.write("Signal Strength Analysis Results:\n")
            f.write(analysis_results)

        pbar.update(1)  # Update progress bar


def generate_preprocessed_heatmap(preprocessed_data, output_dir, date, time, shift_type='none'):
    with tqdm(total=1, desc='Generating Preprocessed Heatmap') as pbar:
        # Apply shift based on the shift_type
        if shift_type == 'redshift':
            shifted_data = apply_redshift(preprocessed_data, delta_lambda, lambda_0)
        elif shift_type == 'blueshift':
            shifted_data = apply_blueshift(preprocessed_data, delta_lambda, lambda_0)
        elif shift_type == 'both':
            redshifted_data = apply_redshift(preprocessed_data, delta_lambda, lambda_0)
            shifted_data = apply_blueshift(redshifted_data, delta_lambda, lambda_0)
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
        plt.figure(figsize=(102, 57))
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
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with the extracted date, time, and shift type
    image_filename = os.path.join(output_dir, f'reconstructed_image_{date}_{time}_{shift_type}.png')

    # Visualize and save the reconstructed image
    plt.figure(figsize=(10, 8))
    plt.imshow(reconstructed_image, cmap='gray')  # Adjust cmap as per your image data
    plt.colorbar(label='Intensity')
    plt.title(f'Reconstructed Image ({shift_type.capitalize()} Shift)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(image_filename, bbox_inches='tight')
    plt.close()

    print(f"Reconstructed image saved as: {image_filename}")



def main(args):
    # Extract date from the input filename
    filename = os.path.basename(args.input)
    date, time = extract_date_from_filename(filename)
    start_time = tm.time()

    if date:
        # Read the RTL-SDR binary file
        with open(args.input, 'rb') as f:
            # Read the binary data and store it in a numpy array
            binary_data = np.fromfile(f, dtype=np.uint8)
            print("Binary data shape:", binary_data.shape)
        # Create the output directory if it does not exist
        os.makedirs(args.output, exist_ok=True)
        binary_data_no_lnb = remove_lnb_offset(binary_data, args.sampling_rate, args.lnb_offset)

        # Generate heatmap
       # generate_heatmap(binary_data_no_lnb, args.output, date, time)
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
       
       
       
        # # Extract wavelet features with bandpass filtering
        # wavelet_features = extract_wavelet_features_with_bandpass(binary_data_no_lnb, args.sampling_rate, args.center_frequency)

        # # Generate wavelet
        # plot_wavelet_features(wavelet_features, args.output, date, time, shift_type='redshift', delta_lambda=delta_lambda, lambda_0=lambda_0)
        # plot_wavelet_features(wavelet_features, args.output, date, time, shift_type='blueshift', delta_lambda=delta_lambda, lambda_0=lambda_0)
        # plot_wavelet_features(wavelet_features, args.output, date, time, shift_type='both', delta_lambda=delta_lambda, lambda_0=lambda_0)
        # plot_wavelet_features(wavelet_features, args.output, date, time, shift_type='none', delta_lambda=delta_lambda, lambda_0=lambda_0)

        reconstructed_image = image_reconstruction(preprocessed_data, shift_type='redshift', delta_lambda=delta_lambda, lambda_0=lambda_0)
        save_visualized_image(reconstructed_image, args.output, date, time, shift_type='redshift')
        reconstructed_image = image_reconstruction(preprocessed_data, shift_type='blueshift', delta_lambda=delta_lambda, lambda_0=lambda_0)
        save_visualized_image(reconstructed_image, args.output, date, time, shift_type='blueshift')
        reconstructed_image = image_reconstruction(preprocessed_data, shift_type='both', delta_lambda=delta_lambda, lambda_0=lambda_0)
        save_visualized_image(reconstructed_image, args.output, date, time, shift_type='both')
        reconstructed_image = image_reconstruction(preprocessed_data, shift_type='none', delta_lambda=delta_lambda, lambda_0=lambda_0)
        save_visualized_image(reconstructed_image, args.output, date, time, shift_type='none')


        end_time = tm.time()
        total_time = (end_time - start_time)/60
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
