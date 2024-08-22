import concurrent.futures
import os
import subprocess
import time
from datetime import datetime
import logging
import queue
import configparser
import re
# Set up logging

logging.basicConfig(filename='radio_astronomy.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

file_queue = queue.Queue()
range_queue = queue.Queue()

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')

SETTINGS = config['settings']
IP = SETTINGS.get('ip')
PORT = SETTINGS.getint('port')
DURATION = SETTINGS.getint('duration')
SRF = SETTINGS.getfloat('srf')
TOL = SETTINGS.getfloat('tol')
CHUNK = SETTINGS.getint('chunk')
WORKERS = SETTINGS.getint('workers')
LAT = SETTINGS.getfloat('lat')
LON = SETTINGS.getfloat('lon')
BASE_DIR = SETTINGS.get('base_directory')
NAS_IMAGES_DIR = SETTINGS.get('nas_images_dir')
NAS_RAW_DIR = SETTINGS.get('nas_raw_dir')


def calculate_sampling_rate(ffreq, lfreq, duration, tolerance):
    bandwidth = lfreq - ffreq
    highest_freq = max(ffreq, lfreq)
    min_sampling_rate = 2 * highest_freq
    srf = max(min_sampling_rate, 2 * bandwidth + tolerance)
    return srf

def extract_frequency_from_filename(filename):
    # Example: file name might be "data_20240815_1420MHz_HI.fits"
    match = re.search(r'(\d+(\.\d+)?)MHz', filename)
    if match:
        return float(match.group(1)) * 1e6  # Convert MHz to Hz
    return None

def run_range(ffreq, lfreq, srf, duration, ip, port, max_attempts=3, retry_delay=10):
    for attempt in range(max_attempts):
        try:
            logging.info(f"Starting range process: {ffreq} to {lfreq} (Attempt {attempt + 1})")
            subprocess.run([
                "python3", "range.py", ip, str(port),
                "--start-freq", str(ffreq), "--end-freq", str(lfreq),
                "--duration", str(duration), "--sample-rate", str(srf)
            ], check=True)
            logging.info(f"Completed range process: {ffreq} to {lfreq}")
            return  # Success, exit the function
        except subprocess.CalledProcessError as e:
            logging.error(f"Range process {ffreq} to {lfreq} failed (Attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Max attempts reached. Range process {ffreq} to {lfreq} failed.")

def process_and_heatmap(file, sfreq, srf, tol, chunk, duration_hours, lat, lon, workers, low_cutoff, high_cutoff):
    try:
        logging.info(f"Processing file: {file}")
        logging.info(f"Current frequency range: {low_cutoff/1e6:.2f} MHz to {high_cutoff/1e6:.2f} MHz")
        logging.info(f"sfreq: {sfreq}")
        filename_w = os.path.splitext(file)[0]
        subprocess.run([
            "python3", "process5.py", "-i", f"{BASE_DIR}/raw/{file}", "-o", f"{BASE_DIR}/images/{filename_w}/",
            "--tolerance", str(tol), "--chunk_size", str(chunk), "--fs", str(srf),
            "--center-frequency", str(sfreq), "--duration", str(duration_hours),
            "--latitude", str(lat), "--longitude", str(lon),
            "--low-cutoff", str(low_cutoff), "--high-cutoff", str(high_cutoff)
        ], check=True)

        logging.info(f"Generating heatmap for file: {file}")
        heatmap_result = subprocess.run([
            "python3", "heatmap.py", "-i", f"{BASE_DIR}/raw/{file}", "-o", f"{BASE_DIR}/images/{filename_w}/",
            "--fs", str(srf), "--num-workers", str(workers), "--nperseg", "2048"
        ], capture_output=True, text=True)
        
        if heatmap_result.returncode == 0:
            logging.info(f"Heatmap generated successfully for file: {file}")
        else:
            logging.error(f"Heatmap generation failed for file: {file}")
            logging.error(f"Heatmap error output: {heatmap_result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Processing or heatmap generation for file {file} failed: {e}")
    finally:
        logging.info(f"Completed processing and heatmap generation for file: {file}")
        with open('processed_files.log', 'a') as log_file:
            log_file.write(f"{datetime.now().isoformat()} - Processed: {file}\n")

def process_file_queue():
    max_retries = 3
    retry_delay = 60  # seconds

    while True:
        try:
            file_info = file_queue.get(block=False)
            retries = 0
            while retries < max_retries:
                try:
                    process_and_heatmap(**file_info)
                    break  # Successfully processed, exit retry loop
                except IOError as e:
                    retries += 1
                    logging.warning(f"I/O error processing {file_info['file']} (attempt {retries}/{max_retries}): {e}")
                    if retries < max_retries:
                        logging.info(f"Re-queueing {file_info['file']} for retry in {retry_delay} seconds")
                        time.sleep(retry_delay)
                    else:
                        logging.error(f"Max retries reached for {file_info['file']}, moving to next file")
            file_queue.task_done()
        except queue.Empty:
            time.sleep(1)
            if file_queue.empty():
                break

def process_range_queue():
    while True:
        try:
            ffreq, lfreq, srf, duration, ip, port = range_queue.get(block=False)
            run_range(ffreq, lfreq, srf, duration, ip, port)
            range_queue.task_done()
        except queue.Empty:
            time.sleep(1)

def process_frequency_range(ffreq, lfreq, sfreq, fileappend, low_cutoff, high_cutoff):
    logging.info(f"Processing frequency range: {ffreq} to {lfreq}")

    # Debugging
    logging.debug(f"BASE_DIR: {BASE_DIR}")
    
    # Read processed files from .dat file
    processed_files = set()
    try:
        with open('processed_files.dat', 'r') as dat_file:
            processed_files = set(line.strip() for line in dat_file)
    except FileNotFoundError:
        logging.info("processed_files.dat not found. Creating a new one.")
    
    # Debugging
    logging.debug(f"Processed files: {processed_files}")
    
    # Check for unprocessed files
    raw_files = [f for f in os.listdir(f"{BASE_DIR}/raw") if f.endswith('.fits')]
    
    # Debugging
    logging.debug(f"Raw files found: {raw_files}")
    
    # Extract frequency from file names and match against frequency ranges
    unprocessed_files = []
    for file in raw_files:
        if file not in processed_files:
            file_freq = extract_frequency_from_filename(file)
            if file_freq and ffreq <= file_freq <= lfreq:
                unprocessed_files.append(file)
    
    # Debugging
    logging.debug(f"Unprocessed files: {unprocessed_files}")
    
    if unprocessed_files:
        logging.info(f"Found {len(unprocessed_files)} unprocessed files: {unprocessed_files}")
        for file in unprocessed_files:
            file_queue.put({
                'file': file,
                'sfreq': sfreq,
                'srf': calculate_sampling_rate(ffreq, lfreq, DURATION, TOL),
                'tol': TOL,
                'chunk': CHUNK,
                'duration_hours': DURATION / 3600,
                'lat': LAT,
                'lon': LON,
                'workers': WORKERS,
                'low_cutoff': low_cutoff,
                'high_cutoff': high_cutoff
            })
            # Add to processed files
            processed_files.add(file)
    
    duration = DURATION
    duration_hours = duration / 3600

    srf = calculate_sampling_rate(ffreq, lfreq, DURATION, TOL)

    range_queue.put((ffreq, lfreq, SRF, duration, IP, PORT))

    # Wait until file is available
    max_wait_time = 30  # Increased wait time for file creation
    start_time = time.time()
    while True:
        files = [f for f in os.listdir('.') if f.endswith('.fits')]
        if files:
            latest_file = max(files, key=os.path.getctime)
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            new_filename = f"data_{timestamp}_{fileappend}.fits"
            try:
                os.rename(latest_file, f"{BASE_DIR}/raw/{new_filename}")
                logging.info(f"File renamed and moved: {new_filename}")
                break  # Exit the loop after successful rename
            except OSError as e:
                logging.error(f"Failed to rename file {latest_file}: {e}")
        if time.time() - start_time > max_wait_time:
            logging.error(f"Timeout waiting for .fits file to be created")
            return
        time.sleep(1)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    new_filename = f"data_{timestamp}_{fileappend}.fits"
   
    try:
        os.rename(latest_file, f"{BASE_DIR}/raw/{new_filename}")
    except OSError as e:
        logging.error(f"Failed to rename file {latest_file} to raw/{new_filename}: {e}")
        return list(unprocessed_files)

    file_queue.put({
                'file': file,
                'sfreq': sfreq,
                'srf': srf,
                'tol': TOL,
                'chunk': CHUNK,
                'duration_hours': duration_hours,
                'lat': LAT,
                'lon': LON,
                'workers': WORKERS,
                'low_cutoff': low_cutoff,
                'high_cutoff': high_cutoff
    })

    with open('processed_files.dat', 'w') as dat_file:
        for file in processed_files:
            dat_file.write(f"{file}\n")

def cleanup_and_sync():
    logging.info("Starting cleanup and synchronization process")
    
    try:
        # Remove empty directories
        subprocess.run(f"find {BASE_DIR}/images/ -type d -empty -delete", shell=True, check=True)
        subprocess.run(f"find {BASE_DIR}/raw/ -type d -empty -delete", shell=True, check=True)

        # Navigate back to the script's directory
        os.chdir(BASE_DIR)

        # Sync processed images to NAS
        subprocess.run(f"rsync -avh --update images/ {NAS_IMAGES_DIR}/", shell=True, check=True)

        # Sync raw data to NAS
        subprocess.run(f"rsync -avh --update raw/ {NAS_RAW_DIR}/", shell=True, check=True)

        # Clean up raw and images directories
        subprocess.run(f"find {BASE_DIR}/raw/ -type f -mmin +1 -exec rm -r {{}} \;", shell=True, check=True)
        subprocess.run(f"find {BASE_DIR}/images/ -type f -mmin +1 -exec rm -r {{}} \;", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Cleanup and sync failed: {e}")

    logging.info("Cleanup and synchronization process completed")
    exit()

if __name__ == "__main__":
    logging.info("Starting radio astronomy processing script")
    os.chdir(BASE_DIR)
    os.makedirs("raw", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    frequency_ranges = [
        (1420.20e6, 1420.60e6, 1420.40e6, "1420MHz_HI", 1420.00e6, 1420.80e6),
        (407.8e6, 408.2e6, 408e6, "408MHz_Haslam", 407.00e6, 409.00e6),
        (150.8e6, 151.2e6, 151e6, "151MHz_6C", 150.00e6, 152.00e6),
        (30.0e6, 80.0e6, 50.0e6, "50MHz_8C", 20.0e6, 80.0e6),
        (322.8e6, 323.2e6, 323e6, "323MHz_Deuterium", 322.0e6, 324.0e6),
        (1610.6e6, 1611.0e6, 1610.8e6, "1611MHz_OH", 1609.0e6, 1612.0e6),
        (1665.2e6, 1665.6e6, 1665.4e6, "1665MHz_OH", 1664.0e6, 1667.0e6),
        (1667.2e6, 1667.6e6, 1667.4e6, "1667MHz_OH", 1666.0e6, 1668.0e6),
        (1720.2e6, 1720.6e6, 1720.4e6, "1720MHz_OH", 1719.0e6, 1721.0e6),
        (2290.8e6, 2291.2e6, 2291e6, "2291MHz_H2CO", 2289.0e6, 2292.0e6),
        (2670.8e6, 2671.2e6, 2671e6, "2671MHz_RRL", 2669.0e6, 2672.0e6),
        (3260.8e6, 3261.2e6, 3261e6, "3261MHz_CH", 3259.0e6, 3262.0e6),
        (3335.8e6, 3336.2e6, 3336e6, "3336MHz_CH", 3334.0e6, 3337.0e6),
        (3349.0e6, 3349.4e6, 3349.2e6, "3349MHz_CH", 3348.0e6, 3350.0e6),
        (4829.4e6, 4830.0e6, 4829.7e6, "4830MHz_H2CO", 4828.0e6, 4831.0e6),
        (5289.6e6, 5290.0e6, 5289.8e6, "5290MHz_OH", 5288.0e6, 5291.0e6),
        (5885.0e6, 5885.4e6, 5885.2e6, "5885MHz_CH3OH", 5884.0e6, 5886.0e6),
        (400.0e6, 800.0e6, 600.0e6, "600MHz_Pulsar", 390.0e6, 810.0e6),
        (1400.0e6, 1400.4e6, 1400.2e6, "1400MHz_Pulsar", 1399.0e6, 1401.0e6),
        (327.0e6, 327.4e6, 327.2e6, "327MHz_Pulsar", 326.0e6, 328.0e6),
        (74.0e6, 74.4e6, 74.2e6, "74MHz_Pulsar", 73.0e6, 75.0e6),
        (408.5e6, 408.9e6, 408.7e6, "408.7MHz_Pulsar", 408.0e6, 409.0e6),
        (800.0e6, 900.0e6, 850.0e6, "850MHz_Pulsar", 790.0e6, 910.0e6),
        (1500.0e6, 1500.4e6, 1500.2e6, "1500MHz_Pulsar", 1499.0e6, 1501.0e6),
        (1427.0e6, 1427.4e6, 1427.2e6, "1427MHz_HI", 1426.0e6, 1428.0e6),
        (550.0e6, 600.0e6, 575.0e6, "575MHz_HCN", 540.0e6, 610.0e6),
        (5500.0e6, 5600.0e6, 5550.0e6, "5550MHz_H2O", 5450.0e6, 5650.0e6),
        (40.0e6, 41.0e6, 40.5e6, "40.5MHz_Galactic_Synchrotron", 39.0e6, 42.0e6),
        (60.0e6, 65.0e6, 62.5e6, "62.5MHz_Low_Frequency_Interference", 55.0e6, 70.0e6),
        (80.0e6, 85.0e6, 82.5e6, "82.5MHz_Extragalactic_Radio_Lobes", 75.0e6, 90.0e6),
        (20.0e6, 30.0e6, 25.0e6, "25MHz_Solar_Radio_Bursts", 15.0e6, 35.0e6),
        (45.0e6, 50.0e6, 47.5e6, "47.5MHz_Interstellar_Absorption", 40.0e6, 55.0e6),
        (95.0e6, 100.0e6, 97.5e6, "97.5MHz_Solar_Coronal_Loops", 90.0e6, 105.0e6),
        (100.0e6, 6000.0e6, 3000.0e6, "Gyrosynchrotron_Emission", 50.0e6, 6050.0e6),
        (10.0e6, 100.0e6, 50.0e6, "Solar_Type_I_Burst", 5.0e6, 105.0e6),
        (20.0e6, 450.0e6, 100.0e6, "Solar_Type_II_Burst", 15.0e6, 455.0e6),
        (10.0e6, 500.0e6, 150.0e6, "Solar_Type_III_Burst", 5.0e6, 505.0e6),
        (20.0e6, 500.0e6, 200.0e6, "Solar_Type_IV_Burst", 15.0e6, 505.0e6),
        (20.0e6, 200.0e6, 100.0e6, "Solar_Type_V_Burst", 15.0e6, 205.0e6),
        (10.0e6, 1000.0e6, 500.0e6, "Plasma_Emission", 5.0e6, 1005.0e6),
        (1000.0e6, 6000.0e6, 3000.0e6, "Thermal_Bremsstrahlung_Emission", 900.0e6, 6100.0e6),
        (30.0e6, 300.0e6, 150.0e6, "Non_Thermal_Continuum_Emission", 25.0e6, 305.0e6)
    ]


    file_tracking = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        executor.submit(process_file_queue)
        executor.submit(process_range_queue)
        future_to_range = {executor.submit(process_frequency_range, *args): args for args in frequency_ranges}
        for future in concurrent.futures.as_completed(future_to_range):
            args = future_to_range[future]
            try:
                files = future.result()
                file_tracking[args[3]] = files
                logging.info(f"Completed range process for {args[3]}")
            except Exception as e:
                logging.error(f"Range process {args[3]} generated an exception: {e}")

    cleanup_and_sync()