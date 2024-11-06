import os
import numpy as np
import time
import socket
from astropy.io import fits
import json
import argparse
import logging

# Constants for default values
DEFAULT_SAMPLE_RATE = 2.4e6  # 2.4 MHz
DEFAULT_DURATION = 60  # seconds

# Set up logging
logging.basicConfig(filename='rtl_sdr_capture.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def capture_data(server_address, start_freq, end_freq, single_freq, sample_rate, duration_seconds, output_dir):
    """Capture data from the RTL-SDR server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(server_address)
            logging.info(f"Connected to server at {server_address}")

            params = {
                'start_freq': start_freq,
                'end_freq': end_freq if not single_freq else start_freq,
                'single_freq': single_freq,
                'sample_rate': sample_rate,
                'duration_seconds': duration_seconds
            }
            client_socket.sendall(json.dumps(params).encode())
            logging.info(f"Sent parameters to server: {params}")

            data = receive_data(client_socket, duration_seconds)

            if data:
                save_data_to_fits(data, start_freq, end_freq, sample_rate, output_dir)
            else:
                logging.warning("No valid data received. FITS file not created.")

    except Exception as e:
        logging.error(f"Error during data capture: {e}")

def receive_data(client_socket, duration_seconds):
    """Receive data from the server."""
    data = bytearray()
    start_time = time.time()
    last_data_time = start_time
    elapsed_time = 0
    client_socket.settimeout(5.0)

    while True:
        try:
            chunk = client_socket.recv(4096)
            if chunk:
                data.extend(chunk)
                last_data_time = time.time()
                logging.debug(f"Received {len(chunk)} bytes of data.")
            else:
                if time.time() - last_data_time > 5:
                    logging.info("No data received for 5 seconds. Assuming transmission complete.")
                    break
                logging.info("No data received. Waiting...")

            elapsed_time = time.time() - start_time
            if elapsed_time >= duration_seconds:
                logging.info(f"Capture duration of {duration_seconds} seconds reached.")
                break

        except socket.timeout:
            if time.time() - last_data_time > 5:
                logging.info("Socket timeout and no data received for 5 seconds. Assuming transmission complete.")
                break
            logging.info("Socket timeout. Continuing to wait for data...")
        except Exception as e:
            logging.error(f"Error receiving data: {e}")
            break

    logging.info(f"Total received data size: {len(data)} bytes")
    return data

def save_data_to_fits(data, start_freq, end_freq, sample_rate, output_dir):
    """Save received data to a FITS file."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    fits_filename = os.path.join(output_dir, f'data_{timestamp}.fits')
    data_array = np.frombuffer(data, dtype=np.complex64)

    if np.any(data_array != 0):
        real_part = data_array.real
        imag_part = data_array.imag

        # Create FITS file
        hdu_real = fits.PrimaryHDU(data=real_part)
        hdu_real.header['DATE'] = time.strftime("%Y-%m-%d", time.gmtime())
        hdu_real.header['TIME'] = time.strftime("%H:%M:%S", time.gmtime())
        hdu_real.header['PART'] = 'REAL'
        hdu_real.header['FREQ'] = (start_freq, 'Start frequency in Hz')
        hdu_real.header['ENDFREQ'] = (end_freq, 'End frequency in Hz')
        hdu_real.header['SFREQ'] = (sample_rate, 'Sample rate in Hz')

        hdu_imag = fits.ImageHDU(data=imag_part)
        hdu_imag.header['PART'] = 'IMAG'

        hdul = fits.HDUList([hdu_real, hdu_imag])
        hdul.writeto(fits_filename, overwrite=True)

        logging.info(f"Data saved to: {fits_filename}")
    else:
        logging.warning("Received only zero values. FITS file not created.")

if __name__ == "__main__":
    logging.info("RTL-SDR Data Capture Client started")

    parser = argparse.ArgumentParser(description='RTL-SDR Data Capture Client')
    parser.add_argument('server_ip', type=str, help='IP address of the server')
    parser.add_argument('server_port', type=int, help='Port number of the server')
    parser.add_argument('--start-freq', type=float, help='Start frequency in Hz', required=True)
    parser.add_argument('--end-freq', type=float, help='End frequency in Hz')
    parser.add_argument('--single-freq', action='store_true', help='Capture data for a single frequency')
    parser.add_argument('--sample-rate', type=float, help='Sample rate in Hz', default=DEFAULT_SAMPLE_RATE)
    parser.add_argument('--duration', type=int, help='Duration of capture in seconds', default=DEFAULT_DURATION)
    parser.add_argument('--output-dir', type=str, help='Directory to save the output file', default='./')
    args = parser.parse_args()

    if args.single_freq:
        args.end_freq = args.start_freq

    os.makedirs(args.output_dir, exist_ok=True)
    server_address = (args.server_ip, args.server_port)
    capture_data(server_address, args.start_freq, args.end_freq, args.single_freq,
                 args.sample_rate, args.duration, args.output_dir)

    logging.info("RTL-SDR Data Capture Client completed")
