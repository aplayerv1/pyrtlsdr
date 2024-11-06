import argparse
import socket
import time
import json
import logging
import numpy as np
from pyhackrf2 import HackRF
from logging.handlers import RotatingFileHandler

# Constants for default configuration
DEFAULT_SAMPLE_RATE = 20000000  # 20 MHz
DEFAULT_GAIN = 20
DEFAULT_DURATION = 10
DEFAULT_BUFFER_SIZE = 1024

# Set up logging
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

def configure_sdr(sdr, sample_rate, gain, center_freq, bandwidth):
    """Configure the HackRF SDR with the given parameters."""
    try:
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        sdr.bandwidth = bandwidth
        sdr.lna_gain = gain
        sdr.vga_gain = gain
        sdr.amp_enable = gain > 0
        sdr.amplifier_on = True
        logging.info(f"Configured SDR with sample_rate={sample_rate}, gain={gain}, center_freq={center_freq}, bandwidth={bandwidth}")
    except Exception as e:
        logging.error(f"Error configuring HackRF: {e}")

def handle_client(client_socket, sdr, tuning_parameters):
    """Handle the client connection and stream samples."""
    try:
        start_freq = int(tuning_parameters['start_freq'])
        end_freq = int(tuning_parameters.get('end_freq', start_freq))
        sample_rate = int(tuning_parameters.get('sample_rate', DEFAULT_SAMPLE_RATE))
        gain = int(tuning_parameters.get('gain', DEFAULT_GAIN))
        duration_seconds = int(tuning_parameters.get('duration_seconds', DEFAULT_DURATION))
        buffer_size = int(tuning_parameters.get('buffer_size', DEFAULT_BUFFER_SIZE))

        # Validate parameters
        if end_freq <= start_freq:
            raise ValueError("end_freq must be greater than start_freq")

        center_freq = (start_freq + end_freq) // 2
        bandwidth = end_freq - start_freq

        logging.info(f"Handling client with parameters: start_freq={start_freq}, end_freq={end_freq}, "
                     f"center_freq={center_freq}, bandwidth={bandwidth}, sample_rate={sample_rate}, "
                     f"gain={gain}, duration={duration_seconds}s, buffer_size={buffer_size}")

        configure_sdr(sdr, sample_rate, gain, center_freq, bandwidth)

        start_time = time.time()
        _stream_samples(sdr, client_socket, buffer_size, duration_seconds, start_time)

    except ValueError as ve:
        logging.error(f"Invalid tuning parameters: {ve}")
        client_socket.sendall(b'Error: Invalid tuning parameters\n')
    except Exception as e:
        logging.error(f"Error while handling client: {e}")

def _stream_samples(sdr, client_socket, buffer_size, duration_seconds, start_time):
    """Stream samples from the SDR to the client socket."""
    end_time = start_time + duration_seconds
    initialization_timeout = 10
    initialization_start = time.time()

    logging.info(f"Starting sample streaming for {duration_seconds} seconds")

    # Wait for SDR to initialize and start reading samples
    while time.time() < end_time:
        try:
            samples = sdr.read_samples(buffer_size)
            if samples.size == 0:
                logging.warning("No samples read from SDR.")
                time.sleep(0.1)
                continue

            # Check for SDR initialization
            if np.any(np.abs(samples - 0.5) > 1e-6):
                logging.info("SDR initialization complete")
                break

            if time.time() - initialization_start > initialization_timeout:
                logging.warning("Initialization timeout reached. Starting data transmission.")
                break

        except Exception as e:
            logging.error(f"Error during sample reading: {e}")
            return  # Exit the function if an error occurs

    actual_start_time = time.time()
    samples_sent = 0

    while time.time() < end_time:
        try:
            samples = sdr.read_samples(buffer_size)
            if samples.size == 0:
                logging.warning("No samples read from SDR.")
                time.sleep(0.1)
                continue

            client_socket.sendall(samples.astype(np.complex64).tobytes())
            samples_sent += len(samples)
            logging.debug(f"Sent {len(samples)} samples to client. Total sent: {samples_sent}")

        except socket.error as e:
            if isinstance(e, BrokenPipeError) or e.errno == errno.EPIPE:
                logging.info("Client disconnected. Stopping transmission.")
                break
            else:
                logging.error(f"Socket error during sample streaming: {e}")
                break
        except Exception as e:
            logging.error(f"Error during sample streaming: {e}")
            break

    logging.info(f"Finished streaming samples. Total samples sent: {samples_sent}")

def main(args):
    """Main function to start the HackRF server."""
    server_address = args.server_address
    server_port = args.server_port

    logging.info(f"Starting HackRF server on {server_address}:{server_port}")

    try:
        sdr = HackRF()
        logging.info("HackRF device initialized successfully")
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
                continue
            except json.JSONDecodeError:
                logging.error("Received malformed JSON from client.")
                client_socket.sendall(b'Error: Malformed JSON\n')
                client_socket.close()
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
    parser.add_argument('-g', '--gain', type=float, default=DEFAULT_GAIN, help='HackRF gain')
    args = parser.parse_args()

    main(args)
