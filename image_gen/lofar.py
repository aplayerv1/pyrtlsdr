import logging
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import cupy as cp
from tqdm import tqdm
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
import os
from datetime import datetime
from multiprocessing import Pool
import gc
from scipy import signal
import cupy as cp
from cupyx.scipy.special import erf

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_coordinates_gpu(times, location, altitude, azimuth):
    logging.debug(f"Entering calculate_coordinates_gpu with times: {times}")
    altaz = cp.array([altitude.value, azimuth.value])
    logging.debug(f"AltAz: {altaz}")
   
    if times.isscalar:
        julian_dates = cp.array([times.jd])
    else:
        julian_dates = cp.array(times.jd)
   
    logging.debug(f"Julian dates: {julian_dates[:5]}... (first 5)")

    lst = cp.remainder(julian_dates * 360.0 / 86164.0905 + location.lon.deg, 360.0)
    ha = lst - altaz[1]

    ra = cp.arctan2(cp.sin(ha), cp.cos(ha) * cp.sin(location.lat.rad) + cp.tan(altaz[0]) * cp.cos(location.lat.rad))
    dec = cp.arcsin(cp.sin(location.lat.rad) * cp.sin(altaz[0]) - cp.cos(location.lat.rad) * cp.cos(altaz[0]) * cp.cos(ha))

    logging.debug(f"Calculated RA: {cp.rad2deg(ra)[:5]}... (first 5)")
    logging.debug(f"Calculated Dec: {cp.rad2deg(dec)[:5]}... (first 5)")
    return cp.rad2deg(ra), cp.rad2deg(dec)

def map_signal_to_sky_chunk(ras_chunk, decs_chunk, signal_chunk, ra_bins, dec_bins):
    logging.debug(f"Entering map_signal_to_sky_chunk with chunks of size: {len(ras_chunk)}")
    sky_map_chunk = cp.zeros((len(dec_bins) - 1, len(ra_bins) - 1), dtype=cp.float32)
   
    for ra, dec, signal in zip(ras_chunk, decs_chunk, signal_chunk):
        ra_idx = cp.searchsorted(ra_bins, ra) - 1
        dec_idx = cp.searchsorted(dec_bins, dec) - 1
       
        if 0 <= ra_idx < len(ra_bins) - 1 and 0 <= dec_idx < len(dec_bins) - 1:
            sky_map_chunk[dec_idx, ra_idx] += signal
   
    logging.debug(f"Sky map chunk created with shape: {sky_map_chunk.shape}")
    return sky_map_chunk

def detect_transients(time_domain_data, threshold=3):
    logging.debug(f"Detecting transients with threshold: {threshold}")
    mean = np.mean(time_domain_data)
    std = np.std(time_domain_data)
    transients = np.where(time_domain_data > mean + threshold * std)[0]
    logging.debug(f"Detected {len(transients)} transient events")
    return transients

def generate_spectrogram(time_domain_data, times, output_dir, date, time, duration_seconds):
    logging.debug("Generating spectrogram")
    plt.figure(figsize=(10, 6))
    time = time.replace(":","")
    unix_times = times.unix
    logging.debug(f"Unix times: {unix_times[:5]}... (first 5)")
   
    total_duration_minutes = duration_seconds / 60  # Convert to minutes
    
    # Calculate the sampling frequency
    fs = len(time_domain_data) / duration_seconds
    
    f, t, Sxx = signal.spectrogram(time_domain_data, fs=fs)
    logging.debug(f"Spectrogram shape: f={f.shape}, t={t.shape}, Sxx={Sxx.shape}")
    
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [min]')
    plt.title(f'Spectrogram - {date} {time}')
    plt.colorbar(label='Power/Frequency [dB/Hz]')
    plt.xlim(0, total_duration_minutes)  # Set x-axis limit to total duration in minutes
    plt.xticks(np.linspace(0, total_duration_minutes, 6))  # Set 6 evenly spaced ticks
    spectrogram_file = os.path.join(output_dir, f'lofar_spectrogram_{date}_{time}.png')
    plt.savefig(spectrogram_file)
    plt.close()
    logging.info(f"Spectrogram saved to: {spectrogram_file}")

def parse_date(date_str):
    logging.debug(f"Parsing date string: {date_str}")
    if len(date_str) == 6:
        year = int('20' + date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
    elif len(date_str) == 8:
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
    else:
        logging.warning(f"Invalid date format: {date_str}. Using current date.")
        return datetime.now()
    parsed_date = datetime(year, month, day)
    logging.debug(f"Parsed date: {parsed_date}")
    return parsed_date

def generate_lofar_image(filtered_fft_values, time_domain_data, freq, output_dir, date, time, lat, lon, duration_hours, chunk_size=1000000):
    logging.info('Starting LOFAR image generation.')
    logging.debug(f'Input parameters: date={date}, time={time}, lat={lat}, lon={lon}, duration_hours={duration_hours}')
    ra_bins = cp.linspace(0, 360, 180)
    dec_bins = cp.linspace(-90, 90, 90)

    sky_map = cp.zeros((len(dec_bins) - 1, len(ra_bins) - 1), dtype=cp.float32)
    with tqdm(total=1, desc='Generating LOFAR Image:') as pbar:
        os.makedirs(output_dir, exist_ok=True)
        try:
            date_obj = parse_date(date)
            time = time.replace(':', '')
            logging.debug(f'Parsed date: {date_obj}')
            start_datetime = datetime.combine(date_obj.date(), datetime.strptime(time, '%H%M%S').time())

            logging.debug(f'start_datetime: {start_datetime}')
            start_time = Time(start_datetime.strftime('%Y-%m-%dT%H:%M:%S'), format='isot')
            logging.debug(f'start_time: {start_time}')

            logging.debug(f'Creating time array with duration: {duration_hours} hours')
            time_intervals = np.linspace(0, duration_hours * 3600, len(filtered_fft_values))
            times = Time([t.isot for t in (start_time + time_intervals * u.second)])
            logging.debug(f'Times array created with shape: {times.shape}')
            location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg)

            altitude = 90 * u.deg
            azimuth = 0 * u.deg

            logging.debug('Calculating coordinates using GPU.')
            ra_gpu, dec_gpu = calculate_coordinates_gpu(times, location, altitude, azimuth)
            ras = cp.asnumpy(ra_gpu)
            decs = cp.asnumpy(dec_gpu)
            logging.debug(f'Coordinates calculated: {ras[:5]}, {decs[:5]}... (showing first 5)')

            logging.debug(f'Starting chunked processing with chunk size: {chunk_size}')

            ras_gpu = cp.array(ras, dtype=cp.float32)
            decs_gpu = cp.array(decs, dtype=cp.float32)
            filtered_fft_values_gpu = cp.array(filtered_fft_values)

            for start in tqdm(range(0, len(filtered_fft_values), chunk_size), desc='Processing Chunks'):
                end = min(start + chunk_size, len(filtered_fft_values))
                ras_chunk = ras_gpu[start:end]
                decs_chunk = decs_gpu[start:end]
                signal_chunk = filtered_fft_values_gpu[start:end]
                sky_map_chunk = map_signal_to_sky_chunk(ras_chunk, decs_chunk, signal_chunk, ra_bins, dec_bins)
                sky_map += sky_map_chunk
                del sky_map_chunk
                cp._default_memory_pool.free_all_blocks()
                gc.collect()

            del ras_gpu, decs_gpu, filtered_fft_values_gpu
            cp._default_memory_pool.free_all_blocks()
            gc.collect()

            logging.debug('Finished processing chunks. Starting analysis.')

            transient_events = detect_transients(time_domain_data)
            generate_spectrogram(time_domain_data, times, output_dir, date, time,duration_hours)

            logging.debug('Starting image generation.')

            non_zero = cp.nonzero(sky_map)
            if non_zero[0].size > 0 and non_zero[1].size > 0:
                ra_min, ra_max = ra_bins[non_zero[1].min()], ra_bins[non_zero[1].max()]
                dec_min, dec_max = dec_bins[non_zero[0].min()], dec_bins[non_zero[0].max()]
            else:
                logging.warning("No non-zero values found in sky_map. Using full range.")
                ra_min, ra_max = ra_bins[0], ra_bins[-1]
                dec_min, dec_max = dec_bins[0], dec_bins[-1]

            ra_buffer, dec_buffer = 2.0, 2.0
            ra_min = max(0, ra_min - ra_buffer)
            ra_max = min(360, ra_max + ra_buffer)
            dec_min = max(-90, dec_min - dec_buffer)
            dec_max = min(90, dec_max + dec_buffer)

            zoom_ra_center, zoom_dec_center = 180, 0
            zoom_factor = 1.0
            ra_min_zoomed = max(ra_min, zoom_ra_center - zoom_factor / 2)
            ra_max_zoomed = min(ra_max, zoom_ra_center + zoom_factor / 2)
            dec_min_zoomed = max(dec_min, zoom_dec_center - zoom_factor / 2)
            dec_max_zoomed = min(dec_max, zoom_dec_center + zoom_factor / 2)

            plt.figure(figsize=(8, 4))
            sky_map_min = sky_map.min().get()
            sky_map_max = sky_map.max().get()

            logging.debug(f"Sky map min: {sky_map_min}, max: {sky_map_max}")

            if sky_map_min == sky_map_max:
                logging.warning("Sky map has uniform values. Adjusting for visualization.")
                sky_map_min -= 1
                sky_map_max += 1

            norm = LogNorm(vmin=max(sky_map_min, 1e-10), vmax=sky_map_max)
            plt.imshow(sky_map.get(), origin='lower', norm=norm, cmap='hot',
                    extent=[ra_min_zoomed, ra_max_zoomed, dec_min_zoomed, dec_max_zoomed], aspect='auto')
            plt.colorbar(label='Normalized Signal Intensity')
            plt.xlabel('Right Ascension (degrees)')
            plt.ylabel('Declination (degrees)')
            plt.title(f'LOFAR Fixed Pointing Observation\n'
                      f'Date: {date}, Time: {time}, Duration: {duration_hours:.2f} seconds\n'
                      f'Latitude: {lat:.2f}°')

            plt.ylim(dec_min_zoomed, dec_max_zoomed)
            plt.plot(zoom_ra_center, zoom_dec_center, 'ro', markersize=10)

            logging.debug('Plotting transient events and spectral lines.')

            for event in transient_events:
                event_time = times[event]
                event_ra, event_dec = calculate_coordinates_gpu(event_time, location, altitude, azimuth)
                plt.plot(event_ra.get(), event_dec.get(), 'g*', markersize=10)

            output_file = os.path.join(output_dir, f'lofar_sky_map_enhanced_{date}_{time}.png')
            plt.savefig(output_file)
            plt.close()

            logging.info(f"Enhanced LOFAR image saved to: {output_file}")
            pbar.update(1)
        except Exception as e:
            logging.error(f"An error occurred while generating LOFAR image: {str(e)}", exc_info=True)
            raise
        finally:
            del filtered_fft_values
            del sky_map
            gc.collect()
            logging.debug('Memory cleared.')
