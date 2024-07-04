import argparse
import socket
import time
import json
import logging
import numpy as np
from SoapySDR import *
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

logging.basicConfig(level=logging.INFO)

def configure_sdr(sdr, sample_rate, gain):
    try:
        sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
        sdr.setGain(SOAPY_SDR_RX, 0, gain)
        sdr.setFrequency(SOAPY_SDR_RX, 0, "RF", 1420000000.0)  # Example frequency setting
        sdr.setFrequency(SOAPY_SDR_RX, 0, "BB", 0.0)
    except Exception as e:
        logging.error(f"Error configuring HackRF via SoapySDR: {e}")

def handle_client(client_socket, sdr, tuning_parameters):
    start_freq = float(tuning_parameters['start_freq'])
    end_freq = float(tuning_parameters.get('end_freq', start_freq))
    single_freq = tuning_parameters.get('single_freq', False)
    sample_rate = float(tuning_parameters['sample_rate'])
    gain = float(tuning_parameters.get('gain', 20))  # Default gain to 20 if not specified
    duration_seconds = float(tuning_parameters.get('duration_seconds', 10))  # Default to 10 seconds if not specified

    logging.info(f"Start frequency: {start_freq} Hz, End frequency: {end_freq} Hz, Single frequency mode: {single_freq}, Sample rate: {sample_rate}, Gain: {gain}, Duration: {duration_seconds} seconds")

    configure_sdr(sdr, sample_rate, gain)
    
    start_time = time.time()

    if single_freq:
        sdr.setFrequency(SOAPY_SDR_RX, 0, start_freq)
        logging.info(f"Receiving at frequency {start_freq} Hz")
        while (time.time() - start_time) < duration_seconds:
            samples = np.zeros(1024, np.complex64)  # Example: read 1024 samples
            sdr.readStream(samples, len(samples))
            logging.info(f"Read {len(samples)} samples")
            try:
                client_socket.sendall(samples.tobytes())
                logging.info(f"Sent {len(samples)} samples to client")
            except Exception as e:
                logging.error(f"Error sending data to client: {e}")
                break
        logging.info(f"Stopped receiving at frequency {start_freq} Hz")
    else:
        freq_range = range(int(start_freq), int(end_freq) + 1_000_000, 1_000_000)
        logging.info(f"Frequency range: {list(freq_range)}")
        
        for freq in freq_range:
            sdr.setFrequency(SOAPY_SDR_RX, 0, freq)
            logging.info(f"Receiving at frequency {freq} Hz")
            freq_start_time = time.time()
            while (time.time() - freq_start_time) < duration_seconds:
                samples = np.zeros(1024, np.complex64)  # Example: read 1024 samples
                sdr.readStream(samples, len(samples))
                logging.info(f"Read {len(samples)} samples at frequency {freq} Hz")
                try:
                    client_socket.sendall(samples.tobytes())
                    logging.info(f"Sent {len(samples)} samples to client at frequency {freq} Hz")
                except Exception as e:
                    logging.error(f"Error sending data to client: {e}")
                    break
                if (time.time() - start_time) > duration_seconds:
                    break
            logging.info(f"Stopped receiving at frequency {freq} Hz")

def main(args):
    server_address = args.server_address
    server_port = args.server_port

    try:
        sdr = SoapySDR.Device()
    except Exception as e:
        logging.error(f"Failed to initialize HackRF device via SoapySDR: {e}")
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
            sdr = None  # Ensure sdr is closed properly
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
