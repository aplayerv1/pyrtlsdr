import argparse
import os
import numpy as np
import time
import socket
from astropy.io import fits
import json

def capture_data(server_address, start_freq, end_freq, single_freq, sample_rate, duration_seconds, output_dir):
    # Connect to the remote server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)
    print(f"Connected to server at {server_address}")

    # Send parameters to the server
    params = {
        'start_freq': start_freq,
        'end_freq': end_freq,
        'single_freq': single_freq,
        'sample_rate': sample_rate,
        'duration_seconds': duration_seconds  # Include duration_seconds in the parameters
    }
    # Convert the dictionary to a JSON string
    params_str = json.dumps(params)

    # Encode the JSON string and send it over the socket
    client_socket.sendall(params_str.encode())

    # client_socket.sendall(str(params).encode())
    
    # Receive data from the server and save it to a FITS file
    data = b''  # Initialize an empty byte string to store the received data
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    fits_filename = os.path.join(output_dir, f'data_{timestamp}.fits')
    header = fits.Header()
    header['DATE'] = time.strftime("%Y-%m-%d", time.gmtime())
    header['TIME'] = time.strftime("%H:%M:%S", time.gmtime())
    start_time = time.time()  # Record the start time

    # Loop to receive data and append it to the byte string
    while True:
        received_data = client_socket.recv(4096)
        if not received_data:
            break  # Exit the loop if no more data is received

        # Append the received data to the byte string
        data += received_data

    data_array = np.frombuffer(data, dtype=np.uint8)

    # Determine dimensions of the data array
    data_shape = data_array.shape

    # Now create the FITS file and save the data
    # Now create the FITS file and save the data
    with fits.open(fits_filename, mode='append') as hdul:
        hdu = fits.PrimaryHDU(data_array, header=header)
        hdu.header['NAXIS'] = len(data_shape)
        for i, dim in enumerate(data_shape):
            hdu.header[f'NAXIS{i+1}'] = dim

    # Convert the byte string to a NumPy array
    data_array = np.frombuffer(data, dtype=np.uint8)

    # Now create the FITS file and save the data
    with fits.open(fits_filename, mode='append') as hdul:
        hdu = fits.PrimaryHDU(data_array, header=header)
        hdul.append(hdu)

    print(f"Data saved to: {fits_filename}")

    # Close the connection
    client_socket.close()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RTL-SDR Data Capture Client')
    parser.add_argument('server_ip', type=str, help='IP address of the server')
    parser.add_argument('server_port', type=int, help='Port number of the server')
    parser.add_argument('--start-freq', type=float, help='Start frequency in Hz', required=True)
    parser.add_argument('--end-freq', type=float, help='End frequency in Hz')
    parser.add_argument('--single-freq', action='store_true', help='Capture data for a single frequency')

    parser.add_argument('--sample-rate', type=float, help='Sample rate in Hz', default=2.4e6)

    parser.add_argument('--duration', type=int, help='Duration of capture in seconds', default=60)
    parser.add_argument('--output-dir', type=str, help='Directory to save the output file', default='./')
    args = parser.parse_args()

    # If single frequency is specified, ignore end frequency
    if args.single_freq:
        args.end_freq = None

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define the server address as a tuple
    server_address = (args.server_ip, args.server_port)

    # Capture data from the server
    capture_data(server_address, args.start_freq, args.end_freq, args.single_freq,
                 args.sample_rate, args.duration, args.output_dir)
