import os
import csv
from astropy.io import fits
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_frequency_ranges(file_path='frequency_ranges.csv'):
    frequency_ranges = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frequency_ranges.append({
                'start': float(row['start_freq']),
                'end': float(row['end_freq']),
                'center': float(row['center_freq']),
                'name': row['name']
            })
    return frequency_ranges

def generate_correct_filename(fits_file, frequency_ranges):
    with fits.open(fits_file) as hdul:
        header = hdul[0].header
        start_freq = header['FREQ']
        date = header['DATE'].replace('-', '')
        time = header['TIME'].replace(':', '')
        
        for range_info in frequency_ranges:
            if range_info['start'] <= start_freq <= range_info['end']:
                return f"data_{date}_{time}_{range_info['name']}.fits"
        
        # If no match found, use the actual frequency
        return f"data_{date}_{time}_{start_freq/1e6:.2f}MHz.fits"

def rename_fits_files(raw_dir, frequency_ranges):
    for filename in os.listdir(raw_dir):
        if filename.endswith('.fits'):
            old_path = os.path.join(raw_dir, filename)
            new_filename = generate_correct_filename(old_path, frequency_ranges)
            new_path = os.path.join(raw_dir, new_filename)
            
            if filename != new_filename:
                os.rename(old_path, new_path)
                logging.info(f"Renamed: {filename} -> {new_filename}")
            else:
                logging.info(f"Filename correct: {filename}")

def check_file():
    raw_dir = "/home/server/rtl/pyrtl/raw"  # Update this path as needed
    frequency_ranges = read_frequency_ranges('frequency_ranges.csv')
    rename_fits_files(raw_dir, frequency_ranges)
    logging.info("Filename check and update complete.")
    
