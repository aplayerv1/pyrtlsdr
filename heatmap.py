import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, lfilter
from astropy.io import fits
import datetime
import multiprocessing as mp
from tqdm import tqdm
import re

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
    data_chunk = read_fits_chunk(filename, chunk_size, chunk_idx)
    # denoised_chunk = denoise_signal(data_chunk, cutoff_freq, fs)
    try:
        f, t, Sxx = spectrogram(data_chunk, fs=fs, nperseg=nperseg, noverlap=nperseg//2, scaling='density', mode='magnitude')
    except ValueError as e:
        print(f"ValueError: {e}. Adjusting nperseg and noverlap.")
        nperseg = min(nperseg, len(data_chunk))
        noverlap = min(nperseg//2, len(data_chunk) - 1)
        f, t, Sxx = spectrogram(data_chunk, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
    return f, t + chunk_idx * (len(data_chunk) / fs), Sxx

def extract_datetime_from_filename(filename):
    """Extract datetime from the filename in the format data_YYMMDD_HHMMSS."""
    basename = os.path.basename(filename)
    match = re.search(r'\w+_(\d{8}_\d{6})', basename)
    if match:
        datetime_str = match.group(1)
        return datetime.datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
    else:
        raise ValueError("Filename does not match the expected format 'data_YYMMDD_HHMMSS'")

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

if __name__ == "__main__":
    main()
