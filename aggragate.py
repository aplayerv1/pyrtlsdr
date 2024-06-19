import os
import re
import argparse
import logging
from astropy.io import fits
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DURATION_PER_FILE = 1200  # 20 minutes

def extract_time_from_filename(filename):
    pattern = r'data_\d{8}_(\d{6})\.fits'
    match = re.match(pattern, filename)
    if match:
        time_str = match.group(1)
        return time_str
    else:
        return None

def process_and_append(filepath, output_path, first_file):
    filename = os.path.basename(filepath)
    logger.info(f"Processing file: {filename}")
    
    try:
        with fits.open(filepath, mode='readonly') as hdul_input:
            if first_file:
                # Write the primary HDU
                hdul_input[0].writeto(output_path, overwrite=True)
                # Append other HDUs
                for hdu in hdul_input[1:]:
                    if hdu.data is not None:
                        with fits.open(output_path, mode='append') as hdul_output:
                            hdul_output.append(hdu)
            else:
                # Append all HDUs
                for hdu in hdul_input:
                    if hdu.data is not None:
                        with fits.open(output_path, mode='append') as hdul_output:
                            hdul_output.append(hdu)
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")

def aggregate_data(data_directory, output_directory):
    today = datetime.now().strftime('%Y%m%d')
    files_by_time = {}

    for filename in os.listdir(data_directory):
        if filename.endswith('.fits'):
            filepath = os.path.join(data_directory, filename)
            time_str = extract_time_from_filename(filename)
            if time_str:
                if time_str not in files_by_time:
                    files_by_time[time_str] = []
                files_by_time[time_str].append(filepath)

    for time_str, filepaths in files_by_time.items():
        if len(filepaths) >= 2:
            logger.info(f"Aggregating data for time: {time_str}")
            aggregate_filename = f'aggregate_{today}_{time_str}.fits'
            output_path = os.path.join(output_directory, aggregate_filename)

            first_file = True
            file_count = 0
            for filepath in filepaths:
                process_and_append(filepath, output_path, first_file)
                first_file = False
                file_count += 1

            # Calculate total duration
            total_duration = file_count * DURATION_PER_FILE

            # Create a directory based on the aggregate filename
            base_name = os.path.splitext(aggregate_filename)[0]
            timestamped_dir = os.path.join(output_directory, base_name)
            os.makedirs(timestamped_dir, exist_ok=True)
            final_output_path = os.path.join(timestamped_dir, aggregate_filename)
            os.rename(output_path, final_output_path)
            logger.info(f"Aggregated file saved to: {final_output_path}")

            # Add header to the final aggregated file
            with fits.open(final_output_path, mode='update') as hdul:
                hdul[0].header['DATE'] = datetime.now().strftime('%Y-%m-%d')
                hdul[0].header['DURATION'] = total_duration  # Add the total duration header

def main():
    parser = argparse.ArgumentParser(description="Aggregate FITS data files.")
    parser.add_argument("--data_directory", "-i", type=str, required=True, help="Path to the directory containing FITS data files.")
    parser.add_argument("--output_directory", "-o", type=str, default=".", help="Path to the directory for saving the aggregated data. Default is the current directory.")
    args = parser.parse_args()

    data_directory = args.data_directory
    output_directory = args.output_directory

    if not os.path.isdir(data_directory):
        logger.error(f"The provided data directory does not exist or is not a directory: {data_directory}")
        return

    if not os.path.exists(output_directory):
        logger.info(f"The output directory does not exist. Creating directory: {output_directory}")
        os.makedirs(output_directory)

    aggregate_data(data_directory, output_directory)

if __name__ == "__main__":
    main()
