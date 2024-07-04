import argparse
import socket
import time
import numpy as np
import json
import logging
from pyhackrf import HackRF

logging.basicConfig(level=logging.INFO)

def configure_sdr(sdr, sample_rate, gain):
    sdr.sample_rate = sample_rate
    sdr.gain = gain

def handle_client(client_socket, sdr, tuning_parameters):
    start_freq = float(tuning_parameters['start_freq'])
    end_freq = float(tuning_parameters.get('end_freq', start_freq))
    single_freq = tuning_parameters.get('single_freq', False)
    sample_rate = float(tuning_parameters['sample_rate'])
    gain = float(tuning_parameters.get('gain', 20))  # Default gain to 20 if not specified
    duration_seconds = tuning_parameters.get('duration_seconds')

    configure_sdr(sdr, sample_rate, gain)
    
    start_time = time.time()

    def rx_callback(samples, context):
        client_socket.sendall(samples.tobytes())
        return 0

    while True:
        if single_freq:
            sdr.center_freq = start_freq
            sdr.start_rx_mode(rx_callback, num_samples=1024)
            time.sleep(1)  # Sleep to allow data to be read and sent
        else:
            for freq in np.arange(start_freq, end_freq + 1e6, 1e6):
                sdr.center_freq = freq
                sdr.start_rx_mode(rx_callback, num_samples=1024)
                time.sleep(1)  # Sleep to allow data to be read and sent
                if duration_seconds and (time.time() - start_time > duration_seconds):
                    break
        
        if duration_seconds and (time.time() - start_time > duration_seconds):
            break

    sdr.stop_rx_mode()

def main(args):
    server_address = args.server_address
    server_port = args.server_port

    sdr = HackRF()

    try:
        sdr.open()
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((server_address, server_port))
        server_socket.listen(1)
        logging.info(f"Server listening on {server_address}:{server_port}")

        while True:
            try:
                client_socket, client_address = server_socket.accept()
                logging.info(f"Connection established with {client_address}")

                tuning_parameters_str = client_socket.recv(4096).decode()
                tuning_parameters = json.loads(tuning_parameters_str)
                logging.info(tuning_parameters)

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
        sdr.close()
        server_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HackRF server for streaming data to clients.')
    parser.add_argument('-a', '--server-address', type=str, default='localhost', help='Server IP address')
    parser.add_argument('-p', '--server-port', type=int, default=8888, help='Server port')
    parser.add_argument('-g', '--gain', type=float, default=20, help='HackRF gain')
    args = parser.parse_args()

    main(args)
