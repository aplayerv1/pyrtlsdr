import numpy as np
import wave
import argparse

# Function to read FFT data from fft.txt and process into sound
def process_fft_to_sound(input_file, output_file, samplerate):
    # Read FFT data from fft.txt
    fft_data = []
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the first line (header)
            fft_value = float(line.strip().split(',')[1])  # Assuming the data format: Frequency(Hz), Real Part, Imaginary Part
            fft_data.append(fft_value)

    # Perform inverse FFT to reconstruct the time-domain signal
    reconstructed_signal = np.fft.ifft(fft_data).real.astype(np.float32)

    # Write the reconstructed audio signal to a WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(reconstructed_signal.tobytes())

    print(f"Audio saved to {output_file}")

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description='Process FFT data into sound and save to a WAV file.')
    parser.add_argument('input_file', type=str, help='Path to input FFT data file (e.g., fft.txt)')
    parser.add_argument('output_file', type=str, help='Path to output WAV file (e.g., output.wav)')
    parser.add_argument('--samplerate', type=int, default=44100, help='Sample rate of the output WAV file (default: 44100)')

    args = parser.parse_args()

    # Call function to process FFT data into sound and save to WAV file
    process_fft_to_sound(args.input_file, args.output_file, args.samplerate)

if __name__ == "__main__":
    main()
