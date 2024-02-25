import argparse
import socket
from rtlsdr import RtlSdr
import time
import numpy as np

def main(args):
    server_address = args.server_address
    server_port = args.server_port
    lnb_frequency = args.lnb_frequency

    # Set up RTL-SDR
    sdr = RtlSdr()

    try:
        # Start the server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((server_address, server_port))
        server_socket.listen(1)
        print(f"Server listening on {server_address}:{server_port}")

        while True:
            try:
                client_socket, client_address = server_socket.accept()
                print(f"Connection established with {client_address}")

                # Receive tuning parameters from the client
                tuning_parameters_str = client_socket.recv(4096).decode()
                tuning_parameters = eval(tuning_parameters_str)

                start_freq = float(tuning_parameters['start_freq'])
                end_freq = tuning_parameters.get('end_freq')
                if end_freq is not None:
                  end_freq = float(end_freq)
                single_freq = tuning_parameters.get('single_freq', False)  # Default to False if not provided                sample_rate = float(tuning_parameters['sample_rate'])
                duration_seconds = int(tuning_parameters['duration_seconds'])
                sample_rate = float(tuning_parameters['sample_rate'])
                # Configure RTL-SDR
                sdr.sample_rate = sample_rate
                sdr.gain = 50 # Set gain to automatic
                if single_freq:
                    sdr.center_freq = start_freq  # Adjust the center frequency with LNB offset
                else:
                    # Capture data for a range of frequencies
                    for freq in np.arange(start_freq, end_freq + 1e6, 1e6):  # Step of 1 MHz
                        sdr.center_freq = freq  # Adjust the center frequency with LNB offset
                        # Here, you can read samples and process them
                # Start streaming data
                start_time = time.time()
                while time.time() - start_time < duration_seconds:
                    if single_freq:
                        # Capture data for a single frequency
                        sdr.center_freq = start_freq
                        samples = sdr.read_samples(1024)  # Read samples from the RTL-SDR device
                        client_socket.sendall(samples.tobytes())  # Send samples to the client
                    else:
                        # Capture data for a range of frequencies
                        for freq in range(int(start_freq), int(end_freq) + 1):
                            if time.time() - start_time >= duration_seconds:
                                break  # Exit the loop if duration exceeds
                            sdr.center_freq = freq
                            samples = sdr.read_samples(1024)
                            client_socket.sendall(samples.tobytes())

                # Close client socket
                client_socket.close()
            except ConnectionResetError:
                print("Client disconnected.")
                client_socket.close()
                continue
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing server.")
    finally:
        sdr.close()
        server_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RTL-SDR server for streaming data to clients.')
    parser.add_argument('-a', '--server-address', type=str, default='localhost', help='Server IP address')
    parser.add_argument('-p', '--server-port', type=int, default=8888, help='Server port')
    parser.add_argument('-f', '--lnb-frequency', type=float, default=9750e6, help='LNB frequency in Hz')
    args = parser.parse_args()

    main(args)
