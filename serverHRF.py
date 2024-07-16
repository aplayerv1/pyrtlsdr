import argparse
import socket
import time
import json
import logging
import numpy as np
from pyhackrf2 import HackRF
from logging.handlers import RotatingFileHandler

# Set up logging to both console and file
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler with 10MB limit and 5 backup files
file_handler = RotatingFileHandler("serverHRF.log", maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Root logger configuration
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

def configure_sdr(sdr, sample_rate, gain):
    try:
        sdr.sample_rate = sample_rate
        sdr.lna_gain = gain
        sdr.vga_gain = gain
        sdr.amp_enable = gain > 0
        sdr.amplifier_on = True
        logging.info(f"Configured SDR with sample_rate={sample_rate}, gain={gain}")
    except Exception as e:
        logging.error(f"Error configuring HackRF: {e}")

def handle_client(client_socket, sdr, tuning_parameters):
    start_freq = int(tuning_parameters['start_freq'])
    end_freq = int(tuning_parameters.get('end_freq', start_freq))
    single_freq = tuning_parameters.get('single_freq', False)
    sample_rate = int(tuning_parameters['sample_rate'])
    gain = int(tuning_parameters.get('gain', 20))
    duration_seconds = int(tuning_parameters.get('duration_seconds', 10))
    buffer_size = int(tuning_parameters.get('buffer_size', 1024))

    logging.info(f"Start frequency: {start_freq} Hz, End frequency: {end_freq} Hz, Single frequency mode: {single_freq}, Sample rate: {sample_rate}, Gain: {gain}, Duration: {duration_seconds} seconds, Buffer size: {buffer_size}")

    configure_sdr(sdr, sample_rate, gain)

    start_time = time.time()

    try:
        if single_freq:
            sdr.center_freq = start_freq
            logging.info(f"Receiving at frequency {start_freq} Hz")
            _stream_samples(sdr, client_socket, buffer_size, duration_seconds, start_time)
        else:
            # Sweeping from start_freq to end_freq over duration_seconds
            step_size = (end_freq - start_freq) / (duration_seconds * 1e6)  # Increment per second in MHz
            current_freq = start_freq

            while current_freq <= end_freq and (time.time() - start_time) <= duration_seconds:
                sdr.center_freq = current_freq
                logging.info(f"Current frequency: {current_freq} Hz, Elapsed time: {time.time() - start_time:.2f} seconds")
                _stream_samples(sdr, client_socket, buffer_size, 1, time.time())  # Stream for 1 second at each frequency
                current_freq += step_size * 1e6  # Increment frequency by step_size in MHz

    except Exception as e:
        logging.error(f"Error during sweeping: {e}")

def _stream_samples(sdr, client_socket, buffer_size, duration_seconds, start_time):
    end_time = start_time + duration_seconds
    initialization_timeout = 10  # 10 seconds timeout for initialization
    initialization_start = time.time()

    while time.time() < end_time:
        try:
            samples = sdr.read_samples(buffer_size)
            if samples.size == 0:
                logging.warning("No samples read from SDR.")
                time.sleep(0.1)
                continue

            # Check if device has initialized (values are not 0.5)
            if np.any(np.abs(samples - 0.5) > 1e-6):
                break

            if time.time() - initialization_start > initialization_timeout:
                logging.warning("Initialization timeout reached. Starting data transmission.")
                break

        except Exception as e:
            logging.error(f"Error during sample reading: {e}")
            break

    actual_start_time = time.time()
    while time.time() < end_time:
        try:
            samples = sdr.read_samples(buffer_size)
            if samples.size == 0:
                logging.warning("No samples read from SDR.")
                time.sleep(0.1)
                continue

            client_socket.sendall(samples.astype(np.complex64).tobytes())
            logging.debug(f"Sent {len(samples)} samples to client")

        except Exception as e:
            logging.error(f"Error during sample streaming: {e}")
            break

    logging.info("Finished streaming samples.")

def main(args):
    server_address = args.server_address
    server_port = args.server_port

    try:
        sdr = HackRF()
    except Exception as e:
        logging.error(f"Failed to initialize HackRF device: {e}")
        return

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_address, server_port))
    server_socket.listen(1)
    logging.info(f"Server listening on {server_address}:{server_port}")

    try:
        while True:
            try:
                client_socket, client_address = server_socket.accept()
                logging.info(f"Connection established with {client_address}")

                tuning_parameters_str = client_socket.recv(4096).decode()
                tuning_parameters = json.loads(tuning_parameters_str)
                logging.info(f"Received tuning parameters: {tuning_parameters}")

                handle_client(client_socket, sdr, tuning_parameters)
                
                client_socket.close()
                logging.info("Client disconnected.")
            except ConnectionResetError:
                logging.info("Client disconnected abruptly.")
                client_socket.close()
                continue
            except Exception as e:
                logging.error(f"Error handling client: {e}")
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Closing server.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        try:
            sdr.close()
            logging.info("HackRF device closed.")
        except Exception as e:
            logging.error(f"Error while closing HackRF device: {e}")
        
        server_socket.close()
        logging.info("Server socket closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HackRF server for streaming data to clients.')
    parser.add_argument('-a', '--server-address', type=str, default='localhost', help='Server IP address')
    parser.add_argument('-p', '--server-port', type=int, default=8888, help='Server port')
    parser.add_argument('-g', '--gain', type=float, default=20, help='HackRF gain')
    args = parser.parse_args()

    main(args)