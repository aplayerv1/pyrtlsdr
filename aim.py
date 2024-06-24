import logging
import numpy as np
from scipy.signal import lfilter, butter
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import os
import psutil
from astropy.io import fits
import scipy.signal
import datetime

# Constants
fs = 2.4e6  # Sampling frequency in Hz
notch_freq = 9750  # Low band LO frequency in MHz
notch_width = 30  # Notch width in MHz
lnb_offset = 9750e6

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_fits_directory(directory):
    """Read FITS files from a directory."""
    fits_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".fits")]
    return fits_files

def read_fits_chunk(filename, chunk_size):
    """Read FITS file in chunks."""
    with fits.open(filename, memmap=True) as hdulist:
        data = hdulist[0].data.astype(np.float64)
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]

def remove_lnb_effect(signal, fs, notch_freq, notch_width):
    """Remove LNB effect from the signal."""
    signal = np.asarray(signal, dtype=np.float64)
    t = np.tan(np.pi * notch_width / fs)
    beta = (1 - t) / (1 + t)
    gamma = -np.cos(2 * np.pi * notch_freq / fs)
    b = [1, gamma * (1 + beta), beta]
    a = [1, gamma * (1 - beta), -beta]
    return lfilter(b, a, signal)

def create_model():
    """Create a machine learning model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Reshape((1024, 1)),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def process_samples(samples, fs, lnb_offset, low_cutoff, high_cutoff):
    """Process the received samples"""
    # Remove LNB effect
    processed_samples = remove_lnb_effect(samples, fs, lnb_offset, notch_width)

    # Apply bandpass filter
    nyq_rate = fs / 2.0
    low_cutoff = low_cutoff / nyq_rate
    high_cutoff = high_cutoff / nyq_rate
    b, a = butter(4, [low_cutoff, high_cutoff], btype='bandpass')
    processed_samples = lfilter(b, a, processed_samples)

    return processed_samples

def generate_wow_signal(n_samples=1024, freq=1420.40e6, drift_rate=10):
    """Generate a synthetic 'Wow!' signal."""
    t = np.linspace(0, n_samples / fs, n_samples)
    f = freq + drift_rate * t
    x = np.sin(2 * np.pi * f * t)
    x += np.random.randn(n_samples) * 0.1
    return x

def generate_noise_sample(n_samples=1024):
    """Generate a noise sample."""
    return np.random.randn(n_samples)

def train_model():
    """Train the machine learning model."""
    wow_signals = []
    noise_samples = []
    for _ in range(1000):
        amplitude = np.random.uniform(0.1, 0.5)
        freq = np.random.uniform(1420.2e6, 1420.6e6)
        noise_level = np.random.uniform(0.05, 0.2)
        wow_signal = amplitude * generate_wow_signal(n_samples=1024, freq=freq, drift_rate=10)
        noise = noise_level * generate_noise_sample(n_samples=1024)
        wow_signal += noise
        wow_signals.append(wow_signal)

        noise_level = np.random.uniform(0.05, 0.2)
        noise_sample = noise_level * generate_noise_sample(n_samples=1024)
        noise_samples.append(noise_sample)

    X_train = np.concatenate([wow_signals, noise_samples])
    y_train = np.concatenate([np.ones(1000), np.zeros(1000)])
    X_train = np.reshape(X_train, (-1, 1024, 1))

    model = create_model()
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    model.save_weights('signal_classifier_weights.h5')

def predict_signal(model, samples, threshold):
    """Predict if a signal is present."""
    reshaped_samples = samples[np.newaxis, :]
    confidence = model.predict(reshaped_samples)
    return confidence[0][0] >= threshold

def plot_signal_strength(signal_strength, filename):
    """Plot signal strength."""
    plt.plot(signal_strength)
    plt.title('Signal Strength')
    plt.xlabel('Samples')
    plt.ylabel('Strength')
    plt.savefig(filename)
    plt.close()

def plot_spectrogram(signal, sample_rate, nperseg, title='Spectrogram'):
    """Plot the spectrogram."""
    datime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'spectrograph_{datime}.png'
    noverlap = nperseg // 2
    f, t, Sxx = scipy.signal.spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    fig = plt.figure()
    plt.imshow(Sxx, aspect='auto', extent=[f.min(), f.max(), 0, 100])
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    fig.savefig(filename, dpi=300)

def set_cpu_affinity(cores):
    """Set CPU affinity for the process."""
    p = psutil.Process(os.getpid())
    p.cpu_affinity(cores)

def main():
    parser = argparse.ArgumentParser(description='Process FITS files from a directory.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing FITS files')
    parser.add_argument('-c', '--chunk-size', type=int, default=1024, help='Chunk size for reading FITS files')
    parser.add_argument('-n', '--num-cores', type=int, default=None, help='Number of CPU cores to use')
    parser.add_argument('--nperseg', type=int, default=1024, help='Number of samples per segment for spectrogram')
    args = parser.parse_args()

    if args.num_cores:
        cores = list(range(args.num_cores))
        set_cpu_affinity(cores)
        logging.info(f'Setting CPU affinity to cores: {cores}')

    fits_files = read_fits_directory(args.directory)
    model_weights_file = 'signal_classifier_weights.h5'

    if os.path.exists(model_weights_file):
        model = create_model()
        model.load_weights(model_weights_file)
        logging.info('Loaded pre-trained model weights.')
    else:
        model = create_model()
        train_model()

    for fits_file in fits_files:
        logging.info(f'Reading FITS file: {fits_file}')
        total_chunks = len(list(read_fits_chunk(fits_file, args.chunk_size)))
        logging.info(f'Total chunks in file: {total_chunks}')
        
        for idx, chunk in enumerate(read_fits_chunk(fits_file, args.chunk_size)):
            low_cutoff = (1420.2e6 - notch_freq) / (fs / 2.0)
            high_cutoff = (1420.6e6 - notch_freq) / (fs / 2.0)
            processed_samples = process_samples(chunk, fs, lnb_offset, low_cutoff, high_cutoff)

            prediction = predict_signal(model, processed_samples, threshold=0.9)
            if prediction:
                logging.info("Signal detected! Creating signal strength plot...")
                plot_signal_strength(processed_samples, filename='signal_strength_plot.png')
                plot_spectrogram(processed_samples, fs, args.nperseg, title='Detected Signal Spectrogram')
            else:
                logging.info("No signal detected.")

            del processed_samples
            gc.collect()

            remaining_chunks = total_chunks - (idx + 1)
            logging.info(f'Remaining chunks of {fits_file}: {remaining_chunks}')

if __name__ == "__main__":
    main()
