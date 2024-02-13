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

# Add logging configuration if necessary

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
    
def plot_wavelet_features(features, output_dir, date, time):
    # Example: Plot wavelet features as a bar chart
    num_features = len(features)
    feature_labels = [f'Feature {i+1}' for i in range(num_features)]

    plt.figure(figsize=(36, 24))
    plt.bar(feature_labels, features)
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.title('Wavelet Features')
    plt.grid(True)

    # Save the plot
    wavelet_plot_filename = os.path.join(output_dir, f'wavelet_features_{date}_{time}.png')
    plt.savefig(wavelet_plot_filename, bbox_inches='tight')
    plt.close()


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

def preprocess_signal(signal_data):
    # Normalize the signal to have zero mean and unit variance
    normalized_signal = (signal_data - np.mean(signal_data)) / np.std(signal_data)

    # Clip the normalized signal to a reasonable range to avoid numerical instability
    clipped_signal = np.clip(normalized_signal, -1.0, 1.0)

    return clipped_signal

def bandpass_filter(signal_data, sampling_rate, center_frequency, bandwidth=100e6, filter_order=5):
    preprocessed_signal = preprocess_signal(signal_data)
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

def generate_heatmap(data, output_dir, date, time):
    with tqdm(total=1, desc='Generating Heatmap') as pbar:
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
        plt.title('Heatmap of RTL-SDR Data')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Generate filename with the extracted date
        heatmap_filename = os.path.join(output_dir, f'heatmap_{date}_{time}.png')
        plt.savefig(heatmap_filename, bbox_inches='tight')
        plt.close()

        pbar.update(1)  # Update progress bar

def generate_frequency_spectrum(data, output_dir, date, time, sampling_rate):
    with tqdm(total=1, desc='Generating Frequency Spectrum') as pbar:
        # Perform Fast Fourier Transform (FFT) to obtain the frequency spectrum
        spectrum = np.fft.fft(data)

        # Calculate the frequency range
        freq_range = np.fft.fftfreq(len(data), d=1/sampling_rate)

        # Plot the frequency spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(freq_range, np.abs(spectrum))
        plt.title('Frequency Spectrum of RTL-SDR Data')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)

        # Save the frequency spectrum plot
        spectrum_filename = os.path.join(output_dir, f'frequency_spectrum_{date}_{time}.png')
        plt.savefig(spectrum_filename, bbox_inches='tight')
        plt.close()
        pbar.update(1)  # Update progress bar

def analyze_signal_strength(data, output_dir, date, time):
    with tqdm(total=1, desc='Analyzing Signal Strength') as pbar:
        # Calculate signal strength from the data
        signal_strength = np.abs(data)

        # Perform statistical analysis
        mean_strength = np.mean(signal_strength)
        std_dev_strength = np.std(signal_strength)

        # Plot histogram of signal strength
        plt.figure(figsize=(8, 6))
        plt.hist(signal_strength, bins=50, color='blue', alpha=0.7)
        plt.axvline(mean_strength, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_strength:.2f}')
        plt.xlabel('Signal Strength')
        plt.ylabel('Frequency')
        plt.title('Histogram of Signal Strength')
        plt.legend()
        plt.grid(True)

        # Save the histogram plot
        histogram_filename = os.path.join(output_dir, f'signal_strength_histogram_{date}_{time}.png')
        plt.savefig(histogram_filename, bbox_inches='tight')
        plt.close()

        # Save the statistical analysis results
        analysis_results = f"Mean Signal Strength: {mean_strength:.2f}\n"
        analysis_results += f"Standard Deviation of Signal Strength: {std_dev_strength:.2f}\n"
        analysis_results_filename = os.path.join(output_dir, f'signal_strength_analysis_{date}_{time}.txt')
        with open(analysis_results_filename, 'w') as f:
            f.write("Signal Strength Analysis Results:\n")
            f.write(analysis_results)
        pbar.update(1)  # Update progress bar

def generate_preprocessed_heatmap(data, output_dir, date, time):
    with tqdm(total=1, desc='Generating Preprocessed Heatmap') as pbar:
        # Apply preprocessing steps to the data (e.g., normalization)
        preprocessed_data = preprocess_data(data)

        # Determine the heatmap dimensions based on the size of the preprocessed data
        num_samples = len(preprocessed_data)
        num_columns = int(np.sqrt(num_samples))  # Adjust to sqrt for more square-like heatmap
        num_rows = int(np.ceil(num_samples / num_columns))  # Adjust to ceil for any remaining rows

        # Determine the size of the heatmap array
        heatmap_size = num_rows * num_columns

        # Pad the data with zeros if necessary to ensure the reshaping works
        padded_data = np.pad(preprocessed_data, (0, heatmap_size - num_samples), mode='constant')

        # Reshape the padded data into a 2D array for the heatmap
        heatmap_data = padded_data.reshape((num_rows, -1))

        # Create and save the heatmap image
        plt.figure(figsize=(102, 57))
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(label='Intensity')
        plt.title('Preprocessed Heatmap of RTL-SDR Data')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Generate filename with the extracted date
        heatmap_filename = os.path.join(output_dir, f'preprocessed_heatmap_{date}_{time}.png')
        plt.savefig(heatmap_filename, bbox_inches='tight')
        plt.close()

        pbar.update(1)  # Update progress bar

def preprocess_data(data):
    # Apply preprocessing steps to the data (e.g., normalization)
    # Example: Normalize the data to have zero mean and unit variance
    normalized_data = (data - np.mean(data)) / np.std(data)
    return normalized_data

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

        # Generate heatmap
        generate_heatmap(binary_data, args.output, date, time)
        preprocessed_data = preprocess_data(binary_data)
        generate_preprocessed_heatmap(preprocessed_data, args.output, date, time)
        # Analyze signal strength
        analyze_signal_strength(binary_data, args.output, date, time)

        # Generate frequency spectrum
        generate_frequency_spectrum(binary_data, args.output, date, time, args.sampling_rate)
        # Extract wavelet features with bandpass filtering
        wavelet_features = extract_wavelet_features_with_bandpass(binary_data, args.sampling_rate, args.center_frequency)

        #Generate wavelet
        plot_wavelet_features(wavelet_features, args.output,date,time)
        
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
    args = parser.parse_args()

    main(args)
