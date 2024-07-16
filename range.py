import os
import numpy as np
import time
import socket
from astropy.io import fits
import json
import argparse

def capture_data(server_address, start_freq, end_freq, single_freq, sample_rate, duration_seconds, output_dir):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)
    print(f"Connected to server at {server_address}")

    params = {
        'start_freq': start_freq,
        'end_freq': end_freq if not single_freq else start_freq,
        'single_freq': single_freq,
        'sample_rate': sample_rate,
        'duration_seconds': duration_seconds
    }
    client_socket.sendall(json.dumps(params).encode())

    data = bytearray()
    start_time = time.time()
    last_data_time = start_time
    client_socket.settimeout(5.0)  # Set a timeout for recv

    while True:
        try:
            chunk = client_socket.recv(4096)
            if chunk:
                data.extend(chunk)
                last_data_time = time.time()
            else:
                if time.time() - last_data_time > 5:  # 5 seconds timeout after last received data
                    print("No data received for 5 seconds. Assuming transmission complete.")
                    break
                print("No data received. Waiting...")
                time.sleep(1)
        except socket.timeout:
            if time.time() - last_data_time > 5:  # 5 seconds timeout after last received data
                print("Socket timeout and no data received for 5 seconds. Assuming transmission complete.")
                break
            print("Socket timeout. Continuing...")
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    client_socket.close()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    fits_filename = os.path.join(output_dir, f'data_{timestamp}.fits')
    data_array = np.frombuffer(data, dtype=np.complex64)  # Reading the buffer as complex64

    data_array = np.frombuffer(data, dtype=np.complex64)

    if np.any(data_array != 0):
        real_part = data_array.real
        imag_part = data_array.imag

        # Create a Primary HDU for the real part
        hdu_real = fits.PrimaryHDU(data=real_part)
        hdu_real.header['DATE'] = time.strftime("%Y-%m-%d", time.gmtime())
        hdu_real.header['TIME'] = time.strftime("%H:%M:%S", time.gmtime())
        hdu_real.header['PART'] = 'REAL'

        # Create an ImageHDU for the imaginary part
        hdu_imag = fits.ImageHDU(data=imag_part)
        hdu_imag.header['PART'] = 'IMAG'

        # Create an HDUList and write to file
        hdul = fits.HDUList([hdu_real, hdu_imag])
        hdul.writeto(fits_filename, overwrite=True)

        print(f"Data saved to: {fits_filename}")
    else:
        print("Received only zero values. FITS file not created.")

    print(f"Received {len(data)} bytes over {elapsed_time:.2f} seconds")

if __name__ == "__main__":
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

    if args.single_freq:
        args.end_freq = args.start_freq

    os.makedirs(args.output_dir, exist_ok=True)
    server_address = (args.server_ip, args.server_port)
    capture_data(server_address, args.start_freq, args.end_freq, args.single_freq,
                 args.sample_rate, args.duration, args.output_dir)
