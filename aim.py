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

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU Configuration
logger.info("Configuring GPU")
physical_devices = tf.config.list_physical_devices('GPU')
logger.info(f"Number of GPUs available: {len(physical_devices)}")
if len(physical_devices) > 0:
    logger.info(f"Enabling memory growth on GPU: {physical_devices[0]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Constants
fs = 2.4e6  # Sampling frequency in Hz
notch_freq = 9750  # Low band LO frequency in MHz
notch_width = 30  # Notch width in MHz
lnb_offset = 9750e6

logger.info(f"Constants set - fs: {fs}, notch_freq: {notch_freq}, notch_width: {notch_width}, lnb_offset: {lnb_offset}")

def read_fits_chunk(filename, chunk_size):
    logger.debug(f"Reading FITS file in chunks: {filename}")
    with fits.open(filename, memmap=True) as hdulist:
        data = hdulist[0].data.astype(np.float64)
        for i in range(0, len(data), chunk_size):
            logger.debug(f"Yielding chunk {i//chunk_size + 1}")
            yield data[i:i+chunk_size]

@tf.function
def tf_remove_lnb_effect(signal, fs, notch_freq, notch_width):
    logger.debug("Removing LNB effect")
    t = tf.tan(np.pi * notch_width / fs)
    beta = (1 - t) / (1 + t)
    gamma = -tf.cos(2 * np.pi * notch_freq / fs)
    b = tf.stack([1.0, gamma * (1 + beta), beta])
    a = tf.stack([1.0, gamma * (1 - beta), -beta])
    
    # Pad the signal and filter
    signal_pad = tf.pad(signal, [[0, tf.shape(b)[0] - 1]])
    b_pad = tf.pad(b, [[0, tf.shape(signal)[0] - 1]])
    
    # Perform convolution using FFT
    signal_fft = tf.signal.fft(tf.cast(signal_pad, tf.complex64))
    b_fft = tf.signal.fft(tf.cast(b_pad, tf.complex64))
    filtered_signal = tf.signal.ifft(signal_fft * b_fft)
    
    # Remove padding and return real part
    return tf.math.real(filtered_signal[:tf.shape(signal)[0]])

@tf.function
def tf_bandpass_filter(signal, low_cutoff, high_cutoff):
    logger.debug(f"Applying bandpass filter: {low_cutoff} - {high_cutoff}")
    b, a = tf.signal.butter(4, [low_cutoff, high_cutoff], btype='bandpass')
    return tf.signal.filter(signal, b, a)

def create_model():
    logger.info("Creating machine learning model")
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

@tf.function
def process_samples(samples, fs, lnb_offset, low_cutoff, high_cutoff):
    logger.debug(f"Processing samples with shape: {samples.shape}")
    samples = tf.cast(samples, dtype=tf.float32)
    fs = tf.cast(fs, dtype=tf.float32)
    lnb_offset = tf.cast(lnb_offset, dtype=tf.float32)

    # Remove LNB effect
    notch_freq_normalized = lnb_offset / (0.5 * fs)
    notch_width = tf.constant(30e6, dtype=tf.float32)  # 30 MHz width
    notch_width_normalized = notch_width / (0.5 * fs)

    b, a = tf.py_function(lambda x, y: signal.iirnotch(x, y), [notch_freq_normalized, notch_width_normalized], [tf.float32, tf.float32])
    
    # Apply filter using convolution
    padded_samples = tf.pad(samples, [[tf.shape(b)[0] - 1, 0]])
    processed_samples = tf.nn.conv1d(tf.expand_dims(padded_samples, axis=1), tf.expand_dims(b, axis=1), stride=1, padding='VALID')
    processed_samples = tf.squeeze(processed_samples)

    # Apply bandpass filter
    nyq_rate = fs / 2.0
    low_cutoff = low_cutoff / nyq_rate
    high_cutoff = high_cutoff / nyq_rate
    processed_samples = tf_bandpass_filter(processed_samples, low_cutoff, high_cutoff)

    fft_result = tf.signal.fft(tf.cast(processed_samples, tf.complex64))
    freqs = tf.signal.fftfreq(tf.shape(processed_samples)[0], 1/fs)
    peak_freq = freqs[tf.argmax(tf.abs(fft_result))]

    logger.debug(f"Peak frequency detected: {peak_freq}")
    return processed_samples, peak_freq

def generate_wow_signal(n_samples=1024, freq=1420.40e6, drift_rate=10):
    logger.debug(f"Generating 'Wow!' signal: freq={freq}, drift_rate={drift_rate}")
    t = np.linspace(0, n_samples / fs, n_samples)
    f = freq + drift_rate * t
    x = np.sin(2 * np.pi * f * t)
    x += np.random.randn(n_samples) * 0.1
    return x

def generate_noise_sample(n_samples=1024):
    logger.debug(f"Generating noise sample: n_samples={n_samples}")
    return np.random.randn(n_samples)

def train_model():
    logger.info("Starting model training")
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
    history = model.fit(X_train, y_train, epochs=20, batch_size=32)
    model.save_weights('signal_classifier_weights.h5')
    logger.info(f"Model training completed. Final accuracy: {history.history['accuracy'][-1]:.2f}")

def predict_signal(model, samples, threshold):
    logger.debug(f"Predicting signal with threshold: {threshold}")
    reshaped_samples = samples[np.newaxis, :]
    confidence = model.predict(reshaped_samples)
    logger.debug(f"Prediction confidence: {confidence[0][0]}")
    return confidence[0][0] >= threshold

def plot_signal_strength(signal_strength, output_dir):
    logger.info("Plotting signal strength")
    filename = os.path.join(output_dir, f'signal_strength_plot_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.plot(signal_strength)
    plt.title('Signal Strength')
    plt.xlabel('Samples')
    plt.ylabel('Strength')
    plt.savefig(filename)
    plt.close()
    logger.info(f"Signal strength plot saved to {filename}")

def plot_spectrogram(signal, sample_rate, nperseg, output_dir, title='Spectrogram'):
    logger.info("Plotting spectrogram")
    filename = os.path.join(output_dir, f'spectrogram_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
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
    logger.info(f"Spectrogram saved to {filename}")

def set_cpu_affinity(cores):
    logger.info(f"Setting CPU affinity to cores: {cores}")
    p = psutil.Process(os.getpid())
    p.cpu_affinity(cores)

def main():
    parser = argparse.ArgumentParser(description='Process a FITS file.')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the FITS file')
    parser.add_argument('-c', '--chunk-size', type=int, default=1024, help='Chunk size for reading FITS file')
    parser.add_argument('-n', '--num-cores', type=int, default=None, help='Number of CPU cores to use')
    parser.add_argument('--nperseg', type=int, default=1024, help='Number of samples per segment for spectrogram')
    parser.add_argument('-o', '--output-dir', type=str, default='output', help='Directory to save output files')
    args = parser.parse_args()

    logger.info(f"Arguments: {args}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_cores:
        cores = list(range(args.num_cores))
        set_cpu_affinity(cores)

    fits_file = args.file
    model_weights_file = 'signal_classifier_weights.h5'

    if os.path.exists(model_weights_file):
        logger.info('Loading pre-trained model weights')
        model = create_model()
        model.load_weights(model_weights_file)
    else:
        logger.info('Training new model')
        model = create_model()
        train_model()

    logger.info(f'Reading FITS file: {fits_file}')
    total_chunks = len(list(read_fits_chunk(fits_file, args.chunk_size)))
    logger.info(f'Total chunks in file: {total_chunks}')
    
    for idx, chunk in enumerate(read_fits_chunk(fits_file, args.chunk_size)):
        logger.debug(f"Processing chunk {idx+1}/{total_chunks}")
        low_cutoff = (1420.2e6 - notch_freq) / (fs / 2.0)
        high_cutoff = (1420.6e6 - notch_freq) / (fs / 2.0)
        processed_samples, peak_freq = process_samples(chunk, fs, lnb_offset, low_cutoff, high_cutoff)

        if abs(peak_freq - 1420e6) > 1e6:  # Allow 1 MHz tolerance
            logger.warning(f"Peak frequency {peak_freq/1e6:.2f} MHz is not close to 1420 MHz")
        else:
            logger.info(f"Verified: Peak frequency {peak_freq/1e6:.2f} MHz is close to 1420 MHz")

        prediction = predict_signal(model, processed_samples, threshold=0.9)
        if prediction:
            logger.info(f"Signal detected in chunk {idx+1}")
            plot_signal_strength(processed_samples, args.output_dir)
            plot_spectrogram(processed_samples, fs, args.nperseg, args.output_dir, title='Detected Signal Spectrogram')
        else:
            logger.debug(f"No signal detected in chunk {idx+1}")

        del processed_samples
        gc.collect()

        remaining_chunks = total_chunks - (idx + 1)
        logger.info(f'Remaining chunks of {fits_file}: {remaining_chunks}')

    logger.info("Finished processing FITS file")

if __name__ == "__main__":
    main()
