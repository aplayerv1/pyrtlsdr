import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from scipy import signal
import socket
import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
import matplotlib.pyplot as plt
import json
import matplotlib
matplotlib.use('TkAgg')  # Choose an appropriate backend

signal_strength_threshold = 6
NUM_SYNTHETIC_SAMPLES = 1024
NUM_EVENTS = 1024
log_file = 'aiml.output.log'
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(log_file, maxBytes=1*1024*1024, backupCount=2)
file_handler.setFormatter(log_formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_signal(signal_data):
    plt.figure()
    plt.plot(signal_data)
    plt.xlabel('Sample Index')
    plt.ylabel('Signal Value')
    plt.title('Signal Line Graph')
    plt.grid(True)
    plt.show()
    print("Ok!")



def generate_synthetic_wow_signal(num_samples, num_events):
    time = np.linspace(0, 1, num_samples)
    frequency = 1420.3
    amplitude = 1.0
    synthetic_signals = [amplitude * np.sin(2 * np.pi * frequency * time) for _ in range(num_events)]
    return synthetic_signals

def combine_data(real_data, synthetic_wow_data):
    num_samples = min(len(real_data), len(synthetic_wow_data))
    combined_data = np.vstack((real_data[:num_samples], synthetic_wow_data[:num_samples]))
    return combined_data

def calculate_signal_strength(data):
    return np.max(data)

def save_data_to_file(data, filename):
    with open(filename, 'wb') as f:
        np.save(f, data)

data_save_path = 'data_on_signal_detection.npy'

def receive_samples_from_server(client_socket):
    samples = client_socket.recv(1024)
    samples_array = np.frombuffer(samples, dtype=np.uint8)
    return samples_array

def remove_noise(signal_data, sampling_frequency, cutoff_frequency):
    nyquist = 0.5 * sampling_frequency
    cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(5, cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, signal_data)
    print(filtered_data)
    return filtered_data

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

def label_data(data, signal_threshold):
    data = np.array(data)
    labels = np.where(data > signal_threshold, 1, 0)
    return labels

def split_data(features, labels, test_size=0.2, random_state=42):
    assert features.shape[0] == labels.shape[0], "Number of samples in features and labels do not match."
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0.0)
    return accuracy, report

def save_model(model, model_file):
    joblib.dump(model, model_file)
    logging.info(f"Model saved to {model_file}")

def combine_data(real_data, synthetic_wow_data):
    combined_data = np.concatenate((real_data, synthetic_wow_data))
    return combined_data

def main(args):
    logging.info("Starting main function")
    logging.info(f"Received server address: {args.server_address}")
    logging.info(f"Received server port: {args.server_port}")
    logging.info(f"Received LNB offset: {args.lnb_offset}")
    logging.info(f"Received center frequency: {args.frequency}")
    logging.info(f"Received gain setting: {args.gain}")
    logging.info(f"Received sampling frequency: {args.sampling_frequency}")
    logging.info(f"Received cutoff frequency: {args.cutoff_frequency}")
    logging.info(f"Received notch bandwidth: {args.notch_bandwidth}")
    try:
        model = joblib.load('signal_detection_model.pkl')
        logging.info("Model loaded successfully")
    except FileNotFoundError:
        logging.warning("Model file not found. Creating a new model.")
        model = train_model()  # Train a new model or initialize it as per your requirements
        save_model(model, 'signal_detection_model.pkl')
    except Exception as e:
        logging.error(f"Error loading the model: {e}")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_address, args.server_port))
    logging.info(f"Connected to {args.server_address}:{args.server_port}")

    try:
        params = {
                'start_freq': args.frequency,
                'end_freq': args.frequency,
                'single_freq': True,
                'sample_rate': 2.4e6,
                'duration_seconds': 100
        }
        params_str = json.dumps(params)

            # Encode the JSON string and send it over the socket
        client_socket.sendall(params_str.encode())

        while True:
            os.system('sleep 1')
            try:
               model = joblib.load('signal_detection_model.pkl')
               logging.info("Model loaded successfully")
            except FileNotFoundError:
                 logging.warning("Model file not found. Creating a new model.")
                 model = train_model()  # Train a new model or initialize it as per your requirements
                 save_model(model, 'signal_detection_model.pkl')
            except Exception as e:
                 logging.error(f"Error loading the model: {e}")

            samples = receive_samples_from_server(client_socket)
            samples_array = np.frombuffer(samples, dtype=np.uint8)
            lnb_removed_data = remove_lnb_offset(samples_array, args.sampling_frequency, args.lnb_offset)
            logging.info("LNB removed from the signal")
            plt.ion()
            plot_signal(samples)

            noise_removed_data = remove_noise(lnb_removed_data, args.sampling_frequency, args.cutoff_frequency)
            logging.info("Noise removed from the signal")

            signal_threshold = 0.5
            real_data = noise_removed_data
            synthetic_wow_data = generate_synthetic_wow_signal(len(noise_removed_data), NUM_EVENTS)
            synthetic_wow_data = np.array(synthetic_wow_data)  # Convert to NumPy array
            logging.info("Synthetic Wow signals generated")

            # Ensure real data and synthetic data have the same number of samples
            num_samples = min(len(real_data), len(synthetic_wow_data))
            logging.info(f"Number of samples: {num_samples}")
            # Label data based on signal strength threshold
            real_labels = label_data(noise_removed_data, signal_threshold)
            synthetic_labels = np.ones(num_samples)  # Ensure synthetic labels match the number of samples
            logging.info(f"Real labels shape: {real_labels.shape}, Synthetic labels shape: {synthetic_labels.shape}")
            # Combine real and synthetic data
            num_samples = min(len(noise_removed_data), len(synthetic_wow_data))
            
            noise_removed_data_reshaped = noise_removed_data.reshape(-1, 1)

            # Ensure that synthetic_wow_data is resized to match the length of noise_removed_data
            synthetic_wow_data_resized = synthetic_wow_data[:len(noise_removed_data_reshaped)]
            # Take only the first 1024 rows of synthetic_wow_data_resized
            synthetic_wow_data_resized = synthetic_wow_data_resized[:, :1]


            # Concatenate noise_removed_data_reshaped and synthetic_wow_data_resized
            combined_data = np.concatenate((noise_removed_data_reshaped, synthetic_wow_data_resized), axis=1)
            assert noise_removed_data_reshaped.shape == (1024, 1), "Shape of noise_removed_data_reshaped is incorrect"
            assert synthetic_wow_data_resized.shape == (1024, 1), "Shape of synthetic_wow_data_resized is incorrect"
            # Concatenate noise_removed_data and synthetic_wow_data horizontally
            
            logging.info(f"Combined data shape: {combined_data.shape}")
        
        
            # Reshape labels to match combined data shape
            real_labels = real_labels.reshape(-1, 1)
            #Reshape real_labels to have two dimensions
            real_labels_reshaped = real_labels.reshape(-1, 1)

            # Reshape synthetic_labels to have two dimensions
            synthetic_labels_reshaped = synthetic_labels.reshape(-1, 1)

            # Concatenate real and synthetic labels
            labels = np.concatenate((real_labels_reshaped, synthetic_labels_reshaped), axis=1)

            logging.info("Labels done")

            X_train, X_test, y_train, y_test = split_data(combined_data, labels, test_size=0.2, random_state=42)
            logging.info("Data split into training and testing sets")

            model = train_model(X_train, y_train)
            logging.info("Model trained")

            accuracy, report = evaluate_model(model, X_test, y_test)
            logging.info(f"Accuracy: {accuracy}")
            logging.info("Classification Report:")
            logging.info(report)

            save_model(model, 'signal_detection_model.pkl')

            if calculate_signal_strength(noise_removed_data) > signal_strength_threshold:
                logging.info("Signal detected! Saving data to file...")
                save_data_to_file(noise_removed_data, data_save_path)
                logging.info("Data saved successfully.")

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Closing connection.")
    finally:
        client_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process stream data from RTL-SDR server.')
    parser.add_argument('-a', '--server-address', type=str, default='localhost', help='Server IP address')
    parser.add_argument('-p', '--server-port', type=int, default=8888, help='Server port')
    parser.add_argument('-o', '--lnb-offset', type=float, default=9750e6, help='LNB offset frequency in Hz')
    parser.add_argument('-f', '--frequency', type=float, default=100e6, help='Center frequency in Hz')
    parser.add_argument('-g', '--gain', type=float, default='auto', help='Gain setting')
    parser.add_argument('-s', '--sampling-frequency', type=float, default=2.4e6, help='Sampling frequency in Hz')
    parser.add_argument('-c', '--cutoff-frequency', type=float, default=1000, help='Cutoff frequency for noise removal')
    parser.add_argument('-n', '--notch-bandwidth', type=float, default=100, help='Bandwidth for notch filter')

    args = parser.parse_args()

    main(args)
