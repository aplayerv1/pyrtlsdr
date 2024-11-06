import concurrent.futures
import os
import subprocess
import sys
import time
from datetime import datetime
import logging
import queue
import configparser
import re
import threading
import csv
import psutil
from check_file import check_file
from astropy.io import fits

# Set up logging

logging.basicConfig(filename='radio_astronomy.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
process_semaphore = threading.Semaphore(1)

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


# Check if processed_files.dat exists and clear it
processed_files_path = 'processed_files.dat'
if os.path.exists(processed_files_path):
    with open(processed_files_path, 'w') as file:
        file.write('')
    logging.info("Cleared processed_files.dat")
else:
    logging.info("processed_files.dat does not exist. A new file will be created when processing starts.")

def rename_and_move_file(latest_file, fileappend):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    new_filename = f"data_{timestamp}_{fileappend}.fits"
    try:
        os.rename(latest_file, f"{BASE_DIR}/raw/{new_filename}")
        logging.info(f"File renamed and moved: {new_filename}")
    except OSError as e:
        logging.error(f"Failed to rename file {latest_file} to raw/{new_filename}: {e}")

def calculate_sampling_rate(ffreq, lfreq, duration, tolerance):
    bandwidth = lfreq - ffreq
    highest_freq = max(ffreq, lfreq)
    min_sampling_rate = 2 * highest_freq
    srf = max(min_sampling_rate, 2 * bandwidth + tolerance)
    return srf

def extract_info_from_fits(file_path):
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        frequency = header.get('FREQ', None)
        date_obs = header.get('DATE', None)
        time_obs = header.get('TIME', None)
    return frequency, date_obs, time_obs

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
  with process_semaphore:
    try:
        logging.info(f"Processing file: {file}")
        logging.info(f"Current frequency range: {low_cutoff/1e6:.2f} MHz to {high_cutoff/1e6:.2f} MHz")
        logging.info(f"sfreq: {sfreq}")
        filename_w = os.path.splitext(file)[0]
        process = subprocess.run([
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
        
        if process.returncode == 0:
            logging.info(f"Processing completed successfully for file: {file}")
            
            # Check if the frequency is close to 1420MHz (allowing for some tolerance)
            # if abs(sfreq - 1420e6) < 1e6:  # Within 1MHz of 1420MHz
            #     logging.info(f"Starting AI process for 1420MHz signal: {file}")
            #     ai_result = subprocess.run([
            #         "/home/server/miniconda3/envs/tf_gpu/bin/python3", "aim.py", "-f", f"{BASE_DIR}/raw/{file}", "-o", f"{BASE_DIR}/images/{filename_w}/",
            #         "-c", "1024", "-n", str(workers), "--nperseg", "2048"
            #     ], capture_output=True, text=True)
                
            #     if ai_result.returncode == 0:
            #         logging.info(f"AI process completed successfully for file: {file}")
            #     else:
            #         logging.error(f"AI process failed for file: {file}")
            #         logging.error(f"AI error output: {ai_result.stderr}")
            # else:
            #     logging.info(f"Skipping AI process for non-1420MHz signal: {file}")
        else:
            logging.error(f"Processing failed for file: {file}")
            logging.error(f"Processing error output: {process.stderr}")


        logging.info(f"Aggragate Data")

        if process.returncode == 0:
            logging.info(f"Processing completed successfully for file: {file}")
        else:
            logging.error(f"Processing failed for file: {file}")
            logging.error(f"Processing error output: {process.stderr}")
    
        
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
    logging.info("process_file_queue started")
    empty_count = 0
    while True:
        try:
            file_info = file_queue.get(block=False)
            empty_count = 0  # Reset counter when an item is found
            logging.info(f"Processing file: {file_info['file']}")
            process_and_heatmap(**file_info)
            file_queue.task_done()
            logging.info(f"Completed processing file: {file_info['file']}")
        except queue.Empty:
            empty_count += 1
            if empty_count % 10 == 0:  # Log every 10th empty check
                logging.info(f"Queue empty for {empty_count} consecutive checks")
            time.sleep(0.1)  # Short sleep to prevent busy waiting
        except Exception as e:
            logging.error(f"Error processing file: {e}", exc_info=True)
        finally:
            log_queue_size()

def process_file_queue():
    logging.info("process_file_queue started")
    while True:
        try:
            file_info = file_queue.get(block=False)
            logging.info(f"Processing file: {file_info['file']}")
            process_and_heatmap(**file_info)
            file_queue.task_done()
            logging.info(f"Completed processing file: {file_info['file']}")
        except queue.Empty:
            logging.info("File queue is empty. Waiting for new files...")
            time.sleep(5)  # Wait for 5 seconds before checking again
        except Exception as e:
            logging.error(f"Error processing file: {e}", exc_info=True)
        finally:
            log_queue_size()



def log_queue_contents():
    logging.info("Current queue contents:")
    for item in list(file_queue.queue):
        logging.info(f"  {item}")
        
def log_queue_size():
    size = file_queue.qsize()
    logging.info(f"Current file queue size: {size}")
    if size > 0:
        logging.info("Next items in queue:")
        for i, item in enumerate(list(file_queue.queue)[:5]):
            logging.info(f"  {i+1}: {item['file']}")

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
    logging.debug(f"BASE_DIR: {BASE_DIR}")
   
    processed_files = set()
    try:
        with open('processed_files.dat', 'r') as dat_file:
            processed_files = set(line.strip() for line in dat_file)
        logging.info(f"Loaded {len(processed_files)} processed files from .dat file")
    except FileNotFoundError:
        logging.info("processed_files.dat not found. Creating a new one.")
   
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.fits')]
    logging.debug(f"Files found in BASE_DIR: {files}")

    for file in files:
        thread = threading.Thread(target=rename_and_move_file, args=(file, fileappend))
        thread.start()
        thread.join()  # Wait for each file to be processed before moving to the next

    logging.info(f"Processed {len(files)} files")

    raw_files = [f for f in os.listdir(f"{BASE_DIR}/raw") if f.endswith('.fits')]
    logging.info(f"Found {len(raw_files)} raw .fits files")
   
    unprocessed_files = []
    for file in raw_files:
        file_path = os.path.join(f"{BASE_DIR}/raw", file)
        frequency, date_obs, time_obs = extract_info_from_fits(file_path)
        if frequency and date_obs and time_obs and ffreq <= frequency <= lfreq:
            new_file_path = rename_file_based_on_info(file_path, frequency, date_obs, time_obs)
            unprocessed_files.append(os.path.basename(new_file_path))
        else:
            logging.debug(f"File {file} frequency {frequency} not in range {ffreq} to {lfreq}")

    logging.info(f"Found {len(unprocessed_files)} unprocessed files: {unprocessed_files}")
   
    for file in unprocessed_files:
        logging.info(f"Queueing file for processing: {file}")
        file_queue.put({
            'file': file,
            'sfreq': sfreq,
            'srf': calculate_sampling_rate(ffreq, lfreq, DURATION, TOL),
            'tol': TOL,
            'chunk': CHUNK,
            'duration_hours': DURATION,
            'lat': LAT,
            'lon': LON,
            'workers': WORKERS,
            'low_cutoff': low_cutoff,
            'high_cutoff': high_cutoff
        })
        processed_files.add(file)
        logging.info(f"Added {file} to processed files set")
   
    duration = DURATION
    duration_hours = duration

    srf = calculate_sampling_rate(ffreq, lfreq, DURATION, TOL)

    range_queue.put((ffreq, lfreq, SRF, duration, IP, PORT))
    logging.info(f"Range queue updated. Starting file wait loop.")
   
    max_wait_time = 30
    start_time = time.time()
    while True:
        files = [f for f in os.listdir(BASE_DIR) if f.endswith('.fits')]
        logging.debug(f"Files found in BASE_DIR: {files}")

        if files:
            latest_file = max(files, key=os.path.getctime)
            logging.info(f"Latest file found: {latest_file}")

            thread = threading.Thread(target=rename_and_move_file, args=(latest_file, fileappend))
            thread.start()

            break  # Exit the loop after starting the rename thread

        if time.time() - start_time > max_wait_time:
            logging.error(f"Timeout waiting for .fits file to be created")
            return

        time.sleep(1)

    file_queue.put({
                'file': file,
                'sfreq': sfreq,
                'srf': srf,
                'tol': TOL,
                'chunk': CHUNK,
                'duration_hours': duration,
                'lat': LAT,
                'lon': LON,
                'workers': WORKERS,
                'low_cutoff': low_cutoff,
                'high_cutoff': high_cutoff
    })

    with open('processed_files.dat', 'w') as dat_file:
        for file in processed_files:
            dat_file.write(f"{file}\n")

    thread.join()  # Wait for the rename thread to complete before returning


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

def cleanup_and_sync_with_timeout():
    cleanup_thread = threading.Thread(target=cleanup_and_sync)
    cleanup_thread.start()
    cleanup_thread.join(timeout=60)  # 10 minutes timeout
    if cleanup_thread.is_alive():
        logging.error("Cleanup and sync process did not complete within the timeout period")

def read_frequency_ranges(file_path='frequency_ranges.csv'):
    frequency_ranges = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frequency_ranges.append((
                float(row['start_freq']),
                float(row['end_freq']),
                float(row['center_freq']),
                row['name'],
                float(row['low_cutoff']),
                float(row['high_cutoff'])
            ))
    return frequency_ranges

def rename_file_based_on_info(file_path, frequency, date_obs, time_obs):
    for start, end, center, name, low_cutoff, high_cutoff in frequency_ranges:
        if start <= frequency <= end:
            date_time = f"{date_obs}T{time_obs}"
            # Changed %y to %Y for 4-digit year format
            date_str = datetime.strptime(date_time, '%Y-%m-%dT%H:%M:%S').strftime("%Y%m%d_%H%M%S")
            new_filename = f"data_{date_str}_{name}.fits"
            new_path = os.path.join(os.path.dirname(file_path), new_filename)
            os.rename(file_path, new_path)
            return new_path
    return file_path

if __name__ == "__main__":
    logging.info("Starting radio astronomy processing script")
    os.chdir(BASE_DIR)
    os.makedirs("raw", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    check_file()
    frequency_ranges = read_frequency_ranges()
    file_tracking = {}

    logging.info("Starting concurrent execution of range processes")

    # Start the process_file_queue in a separate thread
    threading.Thread(target=process_file_queue, daemon=True).start()

    no_activity_count = 0
    max_no_activity = 5  # Exit after 5 consecutive checks with no activity

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        logging.info(f"Submitting file queue processing task")
        executor.submit(process_file_queue)
        
        logging.info(f"Submitting range queue processing task")
        executor.submit(process_range_queue)
        
        logging.info(f"Submitting {len(frequency_ranges)} frequency range processing tasks")
        future_to_range = {executor.submit(process_frequency_range, *args): args for args in frequency_ranges}
        
        while True:
            done, not_done = concurrent.futures.wait(future_to_range.keys(), timeout=60)  # 1 minute timeout
            
            new_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.fits')]
            processes_running = any(p.name() in ['python', 'process5.py', 'range.py', 'heatmap.py'] for p in psutil.process_iter(['name']))

            if not not_done and not new_files and not processes_running and file_queue.empty() and range_queue.empty():
                no_activity_count += 1
                logging.info(f"No new files or active processes. Count: {no_activity_count}")
                if no_activity_count >= max_no_activity:
                    logging.info("No activity detected for a while. Exiting.")
                    break
            else:
                no_activity_count = 0

            logging.info(f"Completed futures: {len(done)}, Incomplete futures: {len(not_done)}")

            time.sleep(60)  # Wait for 1 minute before checking again

        for future, args in future_to_range.items():
            try:
                files = future.result()
                file_tracking[args[3]] = files
            except Exception as e:
                logging.error(f"Range process {args[3]} generated an exception: {e}", exc_info=True)

    logging.info("All range processes completed")
    logging.info(f"Total processed ranges: {len(file_tracking)}")
    logging.info("Starting cleanup and synchronization")
    cleanup_and_sync_with_timeout()
    logging.info("Script execution completed.")
    sys.exit(0)