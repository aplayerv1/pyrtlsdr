import os
import sys
import re
import argparse
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.signal import spectrogram, butter, lfilter
from astropy.io import fits
from tqdm import tqdm
from datetime import datetime  # Changed import

def read_fits_chunk(filename, chunk_size, chunk_idx):
    """Read a specific chunk of data from a FITS file."""
    with fits.open(filename, memmap=True) as hdulist:
        data = hdulist[0].data.astype(np.float64)
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(data))
        return data[start_idx:end_idx]

def denoise_signal(signal, cutoff_freq, fs):
    """Apply a low-pass filter to remove noise from the signal."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, signal)
    return y

def process_chunk(args):
    filename, chunk_size, chunk_idx, fs, cutoff_freq, nperseg = args

    # Get the total number of chunks
    with fits.open(filename, memmap=True) as hdulist:
        data_size = len(hdulist[0].data)
    num_chunks = data_size // chunk_size + (data_size % chunk_size > 0)

    data_chunk = read_fits_chunk(filename, chunk_size, chunk_idx)

    # Calculate the start and end times for the current chunk
    start_time = chunk_idx * chunk_size / fs
    end_time = (chunk_idx + 1) * chunk_size / fs

    # Calculate the overlap with the next chunk
    if chunk_idx < num_chunks - 1:
        next_start_time = (chunk_idx + 1) * chunk_size / fs
        overlap = int((end_time - next_start_time) * fs)
    else:
        overlap = 0

    try:
        f, t, Sxx = spectrogram(data_chunk, fs=fs, nperseg=nperseg, noverlap=overlap, nfft=nperseg, mode='magnitude')
        print(f"Chunk {chunk_idx}: t.min() = {t.min() + start_time}, t.max() = {t.max() + start_time}")
    except ValueError as e:
        print(f"ValueError: {e}. Adjusting nperseg and noverlap.")
        nperseg = min(nperseg, len(data_chunk))
        noverlap = min(nperseg // 2, len(data_chunk) - 1)
        f, t, Sxx = spectrogram(data_chunk, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nperseg, mode='magnitude')
        print(f"Chunk {chunk_idx}: t.min() = {t.min() + start_time}, t.max() = {t.max() + start_time}")

    return f, t + start_time, Sxx

def extract_datetime_from_filename(filename):
    """Extract datetime from filename with flexible pattern matching."""
    basename = os.path.basename(filename)
    
    # Multiple patterns to match different filename formats
    patterns = [
        r'data_(\d{8})_(\d{6})',  # YYYYMMDD_HHMMSS format
        r'data_(\d{6})_(\d{6})',  # YYMMDD_HHMMSS format
        r'(\d{8})_(\d{6})',       # Just YYYYMMDD_HHMMSS
        r'(\d{6})_(\d{6})'        # Just YYMMDD_HHMMSS
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            date_str, time_str = match.groups()
            try:
                if len(date_str) == 6:
                    return datetime.strptime(f"{date_str}_{time_str}", "%y%m%d_%H%M%S")
                else:
                    return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            except ValueError:
                continue
    
    current_time = datetime.now()
    logging.warning(f"Using current datetime {current_time} for filename {basename}")
    return current_time


def main():


    parser = argparse.ArgumentParser(description='Generate a spectrogram from a FITS file.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input FITS file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory for the spectrogram image')
    parser.add_argument('--fs', type=float, default=2.4e6, help='Sampling frequency in Hz (default: 2.4e6)')
    parser.add_argument('--chunk-size', type=int, default=1024*1024, help='Chunk size for reading FITS file (default: 1048576)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count(), help='Number of worker processes (default: all CPU cores)')
    parser.add_argument('--cutoff-freq', type=float, default=1e6, help='Cutoff frequency for low-pass filter (default: 1e6)')
    parser.add_argument('--nperseg', type=int, default=2048, help='Length of each segment for the spectrogram (default: 2048)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist")
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Extract datetime from the input filename
    datetime_from_filename = extract_datetime_from_filename(args.input)
    formatted_datetime = datetime_from_filename.strftime("%Y%m%d_%H%M%S")

    # Get the total number of chunks
    with fits.open(args.input, memmap=True) as hdulist:
        data_size = len(hdulist[0].data)
    num_chunks = data_size // args.chunk_size + (data_size % args.chunk_size > 0)

    # Adjust nperseg if it is larger than chunk size
    if args.nperseg > args.chunk_size:
        print(f"nperseg ({args.nperseg}) is greater than chunk size ({args.chunk_size}), adjusting nperseg to {args.chunk_size}")
        args.nperseg = args.chunk_size

    # Create arguments for multiprocessing
    tasks = [(args.input, args.chunk_size, chunk_idx, args.fs, args.cutoff_freq, args.nperseg) for chunk_idx in range(num_chunks)]

    # Use multiprocessing to process chunks in parallel and add a progress bar
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_chunk, tasks), total=num_chunks))

    # Combine results
    combined_frequencies = results[0][0]
    combined_times = np.concatenate([result[1] for result in results])
    combined_spectrogram = np.concatenate([result[2] for result in results], axis=1)

    # Generate and save the combined spectrogram
    output_path = os.path.join(args.output, f"heatmap_{formatted_datetime}.png")

    plt.figure(figsize=(20, 10))  # Adjust figure size to match the expected heatmap dimensions
    plt.imshow(10 * np.log10(combined_spectrogram), aspect='auto', 
               extent=[combined_times.min(), combined_times.max(), combined_frequencies.min(), combined_frequencies.max()], 
               cmap='viridis', origin='lower')  # Ensure origin is set to 'lower' to match the reference
    plt.colorbar(label='Intensity [dB]')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Spectrogram saved to {output_path}")
    sys.exit(0)
if __name__ == "__main__":
    main()
