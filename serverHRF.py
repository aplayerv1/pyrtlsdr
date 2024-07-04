import argparse
import socket
import time
import json
import logging
import numpy as np
from pyhackrf2 import HackRF

logging.basicConfig(level=logging.INFO)

def configure_sdr(sdr, sample_rate, gain):
    try:
        sdr.sample_rate = int(sample_rate)
        sdr.lna_gain = int(gain)
        sdr.vga_gain = int(gain)
        sdr.amplifier_on = True
    except Exception as e:
        logging.error(f"Error configuring HackRF: {e}")

def handle_client(client_socket, sdr, tuning_parameters):
    start_freq = float(tuning_parameters['start_freq'])
    end_freq = float(tuning_parameters.get('end_freq', start_freq))
    single_freq = tuning_parameters.get('single_freq', False)
    sample_rate = float(tuning_parameters['sample_rate'])
    gain = float(tuning_parameters.get('gain', 20))  # Default gain to 20 if not specified
    duration_seconds = tuning_parameters.get('duration_seconds', 10)  # Default to 10 seconds if not specified
    buffer_size = tuning_parameters.get('buffer_size', 1024)  # Default buffer size

    logging.info(f"Start frequency: {start_freq} Hz, End frequency: {end_freq} Hz, Single frequency mode: {single_freq}, Sample rate: {sample_rate}, Gain: {gain}, Duration: {duration_seconds} seconds, Buffer size: {buffer_size}")

    configure_sdr(sdr, sample_rate, gain)
    
    start_time = time.time()

    if single_freq:
        sdr.center_freq = int(start_freq)
        logging.info(f"Receiving at frequency {start_freq} Hz")
        buffer = []
        while (time.time() - start_time) < duration_seconds:
            samples = sdr.read_samples()
            buffer.extend(samples)
            logging.info(f"Read {len(samples)} samples")
            
            if len(buffer) >= buffer_size:
                try:
                    client_socket.sendall(samples.astype(np.complex64).tobytes())
                    logging.info(f"Sent {len(samples)} samples to client")
                    buffer = []  # Clear buffer after sending
                except Exception as e:
                    logging.error(f"Error sending data to client: {e}")
                    break
        # Send remaining samples in buffer
        if buffer:
            try:
                client_socket.sendall(np.array(buffer).astype(np.complex64).tobytes())
                logging.info(f"Sent remaining {len(buffer)} samples to client")
            except Exception as e:
                logging.error(f"Error sending data to client: {e}")
        logging.info(f"Stopped receiving at frequency {start_freq} Hz")
    else:
        freq_range = range(int(start_freq), int(end_freq) + 1_000_000, 1_000_000)
        logging.info(f"Frequency range: {list(freq_range)}")
        
        for freq in freq_range:
            sdr.center_freq = int(freq)
            logging.info(f"Receiving at frequency {freq} Hz")
            buffer = []
            freq_start_time = time.time()
            while (time.time() - freq_start_time) < duration_seconds:
                samples = sdr.read_samples()
                buffer.extend(samples)
                logging.info(f"Read {len(samples)} samples at frequency {freq} Hz")
                
                if len(buffer) >= buffer_size:
                    try:
                        client_socket.sendall(np.array(buffer).astype(np.complex64).tobytes())
                        logging.info(f"Sent {len(buffer)} samples to client at frequency {freq} Hz")
                        buffer = []  # Clear buffer after sending
                    except Exception as e:
                        logging.error(f"Error sending data to client: {e}")
                        break
                
                if (time.time() - start_time) > duration_seconds:
                    break
            
            # Send remaining samples in buffer
            if buffer:
                try:
                    client_socket.sendall(np.array(buffer).astype(np.complex64).tobytes())
                    logging.info(f"Sent remaining {len(buffer)} samples to client at frequency {freq} Hz")
                except Exception as e:
                    logging.error(f"Error sending data to client: {e}")
            
            logging.info(f"Stopped receiving at frequency {freq} Hz")

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
