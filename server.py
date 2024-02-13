import argparse
import socket
from rtlsdr import RtlSdr

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
            client_socket, client_address = server_socket.accept()
            print(f"Connection established with {client_address}")

            # Receive tuning parameters from the client
            tuning_parameters = client_socket.recv(1024).decode().split(',')
            sample_rate = float(tuning_parameters[0])
            center_frequency = float(tuning_parameters[1])
            gain = float(tuning_parameters[2])  # Convert gain to float

            # Configure RTL-SDR
            sdr.sample_rate = sample_rate
            sdr.center_freq = center_frequency
            sdr.gain = gain  # Set gain directly

            # Stream data to the client
            while True:
                samples = sdr.read_samples(1024)
                try:
                    client_socket.sendall(samples)
                except BrokenPipeError:
                    print("Client disconnected.")
                    break
                except ConnectionResetError:
                    print("Client disconnected.")
                    break

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
