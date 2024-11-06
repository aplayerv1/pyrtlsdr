import socket
import numpy as np
import cupy as cp
import pickle
import struct
from numba import cuda
import cupy as cp
import numpy as np

@cuda.jit
def optimized_comb_kernel(signal, freqs, notch_freqs, Q, output):
    idx = cuda.grid(1)
    
    # GTX 1080 Ti optimized parameters
    if idx < signal.shape[0]:
        shared_freq = cuda.shared.array(shape=(512,), dtype=cuda.float64)
        
        for i in range(0, len(notch_freqs), 512):
            if idx < min(512, len(notch_freqs) - i):
                shared_freq[idx] = notch_freqs[i + idx]
            cuda.syncthreads()
            
            for j in range(min(512, len(notch_freqs) - i)):
                notch = shared_freq[j]
                freq_diff = abs(freqs[idx] - notch)
                if freq_diff < Q:
                    signal[idx] *= (freq_diff / Q)
            cuda.syncthreads()
            
def handle_comb_filter(data, fs, notch_freq, Q):
    # GPU processing on server side
    d_data = cp.asarray(data)
    fft_data = cp.fft.fft(d_data)
    freqs = cp.fft.fftfreq(len(d_data), 1/fs)
    notch_freqs = cp.arange(notch_freq, 0.5 * fs, notch_freq)
    
    # Apply optimized kernel
    blocks = (len(d_data) + 512 - 1) // 512
    optimized_comb_kernel[blocks, 512](fft_data, freqs, notch_freqs, Q)
    
    result = cp.asnumpy(cp.real(cp.fft.ifft(fft_data)))
    return result

def receive_data(conn):
    # Receive data size first
    size_data = conn.recv(8)
    size = struct.unpack('!Q', size_data)[0]
    
    # Receive the actual data
    data = b''
    while len(data) < size:
        packet = conn.recv(size - len(data))
        data += packet
    
    return pickle.loads(data)

def start_gpu_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 9999))
    server.listen(5)
    print("GPU Server running on port 9999")
    
    while True:
        conn, addr = server.accept()
        print(f"Connection from {addr}")
        
        # Receive parameters
        params = receive_data(conn)
        data, fs, notch_freq, Q = params
        
        # Process on GPU
        result = handle_comb_filter(data, fs, notch_freq, Q)
        
        # Send back results
        conn.sendall(pickle.dumps(result))
        conn.close()
        
        