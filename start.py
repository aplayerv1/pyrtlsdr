import concurrent.futures
import os
import subprocess
import time
from datetime import datetime
import logging
import queue
import configparser

logging.basicConfig(filename='radio_astronomy.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

file_queue = queue.Queue()

config = configparser.ConfigParser()
config.read('config.ini')

SETTINGS = config['settings']

IP = SETTINGS.get('ip')
PORT = SETTINGS.getint('port')
SRF = SETTINGS.getfloat('srf')
TOL = SETTINGS.getfloat('tol')
CHUNK = SETTINGS.getint('chunk')
WORKERS = SETTINGS.getint('workers')
LAT = SETTINGS.getfloat('lat')
LON = SETTINGS.getfloat('lon')
BASE_DIR = SETTINGS.get('base_directory')
NAS_IMAGES_DIR = SETTINGS.get('nas_images_dir')
NAS_RAW_DIR = SETTINGS.get('nas_raw_dir')

def run_range(ffreq, lfreq, srf, duration, ip, port):
    try:
        logging.info(f"Starting range process: {ffreq} to {lfreq}")
        subprocess.run(["python3", "range.py", ip, str(port), "--start-freq", str(ffreq), "--end-freq", str(lfreq), "--duration", str(duration), "--sample-rate", str(srf)], check=True)
        logging.info(f"Completed range process: {ffreq} to {lfreq}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Range process {ffreq} to {lfreq} failed: {e}")

def process_and_heatmap(file, sfreq, srf, tol, chunk, duration_hours, lat, lon, workers):
    try:
        logging.info(f"Processing file: {file}")
        filename_w = os.path.splitext(file)[0]
        subprocess.run(["python3", "process5.py", "-i", f"{BASE_DIR}/raw/{file}", "-o", f"{BASE_DIR}/images/{filename_w}/",
                        "--tolerance", str(tol), "--chunk_size", str(chunk), "--fs", str(srf), "--center-frequency", str(sfreq),
                        "--duration", str(duration_hours), "--latitude", str(lat), "--longitude", str(lon)], check=True)

        logging.info(f"Generating heatmap for file: {file}")
        heatmap_result = subprocess.run(["python3", "heatmap.py", "-i", f"{BASE_DIR}/raw/{file}", "-o", f"{BASE_DIR}/images/{filename_w}/",
                        "--fs", str(srf), "--num-workers", str(workers), "--nperseg", "2048"], capture_output=True, text=True)
        
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
    while True:
        try:
            file_info = file_queue.get(block=False)
            process_and_heatmap(**file_info)
            file_queue.task_done()
        except queue.Empty:
            time.sleep(1)

def process_frequency_range(ffreq, lfreq, sfreq, fileappend):
    logging.info(f"Processing frequency range: {ffreq} to {lfreq}")
    duration = 3600
    duration_hours = duration / 3600

    run_range(ffreq, lfreq, SRF, duration, IP, PORT)

    while True:
        files = [f for f in os.listdir('.') if f.endswith('.fits')]
        if files:
            latest_file = max(files, key=os.path.getctime)
            break
        time.sleep(1)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    new_filename = f"data_{timestamp}_{fileappend}.fits"
    
    try:
        os.rename(latest_file, f"{BASE_DIR}/raw/{new_filename}")
    except OSError as e:
        logging.error(f"Failed to rename file {latest_file} to raw/{new_filename}: {e}")
        return []

    file_queue.put({
        'file': new_filename,
        'sfreq': sfreq,
        'srf': SRF,
        'tol': TOL,
        'chunk': CHUNK,
        'duration_hours': duration_hours,
        'lat': LAT,
        'lon': LON,
        'workers': WORKERS
    })

    logging.info(f"Queued processing for frequency range: {ffreq} to {lfreq}")
    return [new_filename]

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

if __name__ == "__main__":
    logging.info("Starting radio astronomy processing script")
    os.chdir(BASE_DIR)
    os.makedirs("raw", exist_ok=True)
    os.makedirs("images", exist_ok=True)


    frequency_ranges = [
        (1420.20e6, 1420.60e6, 1420.40e6, "1420MHz_HI"),
        (407.8e6, 408.2e6, 408e6, "408MHz_Haslam"),
        (150.8e6, 151.2e6, 151e6, "151MHz_6C"),
        (30.0e6, 80.0e6, 50.0e6, "50MHz_8C"),
        (322.8e6, 323.2e6, 323e6, "323MHz_Deuterium"),
        (1610.6e6, 1611.0e6, 1610.8e6, "1611MHz_OH"),
        (1665.2e6, 1665.6e6, 1665.4e6, "1665MHz_OH"),
        (1667.2e6, 1667.6e6, 1667.4e6, "1667MHz_OH"),
        (1720.2e6, 1720.6e6, 1720.4e6, "1720MHz_OH"),
        (2290.8e6, 2291.2e6, 2291e6, "2291MHz_H2CO"),
        (2670.8e6, 2671.2e6, 2671e6, "2671MHz_RRL"),
        (3260.8e6, 3261.2e6, 3261e6, "3261MHz_CH"),
        (3335.8e6, 3336.2e6, 3336e6, "3336MHz_CH"),
        (3349.0e6, 3349.4e6, 3349.2e6, "3349MHz_CH"),
        (4829.4e6, 4830.0e6, 4829.7e6, "4830MHz_H2CO"),
        (5289.6e6, 5290.0e6, 5289.8e6, "5290MHz_OH"),
        (5885.0e6, 5885.4e6, 5885.2e6, "5885MHz_CH3OH"),
        (400.0e6, 800.0e6, 600.0e6, "600MHz_Pulsar"),
        (1400.0e6, 1400.4e6, 1400.2e6, "1400MHz_Pulsar"),
        (327.0e6, 327.4e6, 327.2e6, "327MHz_Pulsar"),
        (74.0e6, 74.4e6, 74.2e6, "74MHz_Pulsar"),
        (408.5e6, 408.9e6, 408.7e6, "408.7MHz_Pulsar"),
        (800.0e6, 900.0e6, 850.0e6, "850MHz_Pulsar"),
        (1500.0e6, 1500.4e6, 1500.2e6, "1500MHz_Pulsar"),
        (1427.0e6, 1427.4e6, 1427.2e6, "1427MHz_HI"),
        (550.0e6, 600.0e6, 575.0e6, "575MHz_HCN"),
        (5500.0e6, 5600.0e6, 5550.0e6, "5550MHz_H2O"),
        (40.0e6, 41.0e6, 40.5e6, "40.5MHz_Galactic_Synchrotron"),
        (60.0e6, 65.0e6, 62.5e6, "62.5MHz_Low_Frequency_Interference"),
        (80.0e6, 85.0e6, 82.5e6, "82.5MHz_Extragalactic_Radio_Lobes"),
        (20.0e6, 30.0e6, 25.0e6, "25MHz_Solar_Radio_Bursts"),
        (45.0e6, 50.0e6, 47.5e6, "47.5MHz_Interstellar_Absorption"),
        (95.0e6, 100.0e6, 97.5e6, "97.5MHz_Solar_Coronal_Loops"),
        (100.0e6, 6000.0e6, 3000.0e6, "Gyrosynchrotron_Emission"),
        (10.0e6, 100.0e6, 50.0e6, "Solar_Type_I_Burst"),
        (20.0e6, 450.0e6, 100.0e6, "Solar_Type_II_Burst"),
        (10.0e6, 500.0e6, 150.0e6, "Solar_Type_III_Burst"),
        (20.0e6, 500.0e6, 200.0e6, "Solar_Type_IV_Burst"),
        (20.0e6, 200.0e6, 100.0e6, "Solar_Type_V_Burst"),
        (10.0e6, 1000.0e6, 500.0e6, "Plasma_Emission"),
        (1000.0e6, 6000.0e6, 3000.0e6, "Thermal_Bremsstrahlung_Emission"),
        (30.0e6, 300.0e6, 150.0e6, "Non_Thermal_Continuum_Emission")
    ]

    file_tracking = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(process_file_queue)
        future_to_range = {executor.submit(process_frequency_range, *args): args for args in frequency_ranges}
        for future in concurrent.futures.as_completed(future_to_range):
            args = future_to_range[future]
            try:
                files = future.result()
                file_tracking[args[3]] = files
                logging.info(f"Completed range process for {args[3]}")
            except Exception as exc:
                logging.error(f'{args[3]} generated an exception: {exc}')

    file_queue.join()  # Wait for all processing to complete

    for range_name, files in file_tracking.items():
        logging.info(f"Files created for {range_name}: {files}")

    # Perform cleanup and synchronization after all processing is done
    cleanup_and_sync()

    logging.info("Radio astronomy processing script completed")