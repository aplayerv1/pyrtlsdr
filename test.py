import numpy as np
import matplotlib.pyplot as plt
import logging
import os

def pad_to_square(signal_data):
    # Pad the signal data to the nearest perfect square length
    length = len(signal_data)
    next_square = int(np.ceil(np.sqrt(length)) ** 2)
    padded_data = np.pad(signal_data, (0, next_square - length), mode='constant')
    return padded_data

def create_intensity_map(signal_data, sampling_rate, output_dir, date, time):
    logging.debug("Starting create_intensity_map function")
    logging.debug(f"Original signal_data shape: {signal_data.shape}")

    # Plot original signal data
    plt.figure()
    plt.plot(np.abs(signal_data))
    plt.title("Original Signal Data")
    plt.show()

    # Ensure signal_data is 2D for plotting
    if len(signal_data.shape) == 1:
        logging.debug("Signal data is 1D, padding and reshaping to 2D")
        padded_data = pad_to_square(signal_data)
        side_length = int(np.sqrt(len(padded_data)))
        signal_data = np.reshape(padded_data, (side_length, side_length))
        logging.debug(f"Reshaped signal_data shape: {signal_data.shape}")

    # Plot reshaped signal data
    plt.figure()
    plt.imshow(np.abs(signal_data), aspect='auto', cmap='viridis')
    plt.title("Reshaped Signal Data")
    plt.colorbar()
    plt.show()

    # Calculate the frequency bins
    freq_bins = np.fft.fftfreq(signal_data.shape[0], d=1/sampling_rate)
    logging.debug(f"Frequency bins: {freq_bins}")

    # Calculate the intensity map
    intensity_map = np.abs(np.fft.fft2(signal_data))
    logging.debug(f"Intensity map shape: {intensity_map.shape}")

    # Plot the intensity map
    plt.figure()
    plt.imshow(np.log1p(intensity_map), aspect='auto', cmap='viridis', extent=[freq_bins.min(), freq_bins.max(), freq_bins.min(), freq_bins.max()])
    plt.title("Intensity Map")
    plt.colorbar()
    plt.show()

    # Save the intensity map
    output_path = os.path.join(output_dir, f"intensity_map_{date}_{time}.png")
    plt.imsave(output_path, np.log1p(intensity_map), cmap='viridis')
    logging.debug(f"Intensity map saved to {output_path}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    signal_data = np.random.rand(485376)  # Example signal data
    sampling_rate = 44100  # Example sampling rate
    output_dir = "./"  # Example output directory
    date = "2024-07-15"
    time = "09-37-25"
    create_intensity_map(signal_data, sampling_rate, output_dir, date, time)