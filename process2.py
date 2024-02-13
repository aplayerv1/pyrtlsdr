import numpy as np
import argparse
import socket
from scipy import signal
import json

# Import libraries for HTML generation
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

# Global variables for HTML generation
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Signal Power</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="live_graph"></div>
    <script>
        var signalPower = [];
        var timeStamps = [];
        var layout = {
            title: "Real-time Signal Power",
            xaxis: { title: "Time" },
            yaxis: { title: "Signal Power" }
        };
        var config = { responsive: true };

        function updateGraph() {
            Plotly.newPlot('live_graph', [{ x: timeStamps, y: signalPower, type: 'scatter', mode: 'lines+markers' }], layout, config);
        }

        setInterval(function () {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    signalPower = data.signalPower;
                    timeStamps = data.timeStamps;
                    updateGraph();
                })
                .catch(error => console.error('Error fetching data:', error));
        }, 1000);
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # Construct JSON response with signal power data
            response = {
                'signalPower': signal_power_list,
                'timeStamps': time_stamp_list
            }
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/api':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # Construct JSON response with the latest signal power data, timestamp, raw samples, and processed samples
            response = {
                'signalPower': signal_power_list[-1] if signal_power_list else None,
                'timeStamp': time_stamp_list[-1] if time_stamp_list else None,
                'rawSamples': raw_samples_list[-1].tolist() if raw_samples_list else None,
                'processedSamples': processed_samples_list[-1].tolist() if processed_samples_list else None
            }
            self.wfile.write(json.dumps(response).encode())

# Initialize signal power, time stamp, raw samples, and processed samples lists for HTML
signal_power_list = []
time_stamp_list = []
raw_samples_list = []
processed_samples_list = []

def display_signal_power(samples, sample_rate):
    # Convert samples to floating point numbers
    samples_float = samples.astype(float)

    # Calculate signal power from the received samples
    signal_power = np.mean(np.abs(samples_float) / 2 / 10e8)

    # Append signal power and current timestamp to the lists
    signal_power_list.append(signal_power)
    time_stamp_list.append(time.time())

    # Append raw and processed samples to the lists
    raw_samples_list.append(samples)
    processed_samples = process_samples(samples, sample_rate)  # Fix applied here
    processed_samples_list.append(processed_samples)


def receive_samples_from_server(client_socket):
    # Receive samples from the server
    samples = client_socket.recv(1024)  # Adjust buffer size as needed
    samples_array = np.frombuffer(samples, dtype=np.uint8)  # Convert bytes to numpy array
    return samples_array

def process_samples(samples, sample_rate):
    # Define filter parameters
    cutoff_freq = 1e6  # Cutoff frequency of the low-pass filter in Hz
    nyquist_freq = sample_rate / 2  # Nyquist frequency
    norm_cutoff_freq = cutoff_freq / nyquist_freq
    # Design a high-pass filter using a Butterworth filter
    b, a = signal.butter(4, norm_cutoff_freq, 'high')
    # Apply the filter to the samples
    filtered_samples = signal.filtfilt(b, a, samples)
    return filtered_samples

def close_client_socket(client_socket):
    # Close the client socket
    client_socket.close()

def adjust_samples_for_lnb(samples, lnb_offset):
    adjusted = samples - lnb_offset
    return adjusted

def main(args):
    # Connect to the server and set up parameters
    server_address = args.server_address
    server_port = args.server_port
    lnb_offset = args.lnb_offset  # Adjusted for LNB offset
    frequency = args.frequency
    gain = args.gain
    sample_rate = args.sample_rate

    # Connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_address, server_port))
    print(f"Connected to {server_address}:{server_port}")

    try:
        # Send tuning parameters to the server
        tuning_parameters = f"{sample_rate},{frequency},{gain}"
        client_socket.sendall(tuning_parameters.encode())

        # Set up HTTP server for real-time HTML display
        http_server = HTTPServer(('0.0.0.0', 8000), RequestHandler)
        print('HTTP server started on port 8000')

        # Main data processing loop
        while True:

            samples = receive_samples_from_server(client_socket)

            # Adjust samples for LNB offset
            adjusted_samples = adjust_samples_for_lnb(samples, lnb_offset)

            # Process the adjusted samples using the sample_rate

            # Display real-time monitoring information
            display_signal_power(adjusted_samples, sample_rate)

            # Handle HTTP requests in a non-blocking manner
            http_server.handle_request()

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing connection.")
    finally:
        # Close the client socket
        close_client_socket(client_socket)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process stream data from RTL-SDR server.')
    parser.add_argument('-a', '--server-address', type=str, default='localhost', help='Server IP address')
    parser.add_argument('-p', '--server-port', type=int, default=8888, help='Server port')
    parser.add_argument('-o', '--lnb-offset', type=float, default=9750e6, help='LNB offset frequency in Hz')
    parser.add_argument('-f', '--frequency', type=float, default=100e6, help='Center frequency in Hz')
    parser.add_argument('-g', '--gain', type=float, default='auto', help='Gain setting')
    parser.add_argument('-s', '--sample-rate', type=float, default=2.4e6, help='Sample rate in Hz')

    args = parser.parse_args()

    # Run the main function
    main(args)