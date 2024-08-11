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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_coordinates(time, location, altitude, azimuth):
    altaz = AltAz(obstime=time, location=location)
    skycoord = SkyCoord(alt=altitude, az=azimuth, frame=altaz)
    icrs_coord = skycoord.icrs
    return icrs_coord.ra.deg, icrs_coord.dec.deg

def calculate_coordinates_wrapper(args):
    return calculate_coordinates(*args)

def map_signal_to_sky(ras, decs, signal_data, ra_bins, dec_bins):
    sky_map = cp.zeros((len(dec_bins) - 1, len(ra_bins) - 1), dtype=cp.float32)
    
    for ra, dec, signal in zip(ras, decs, signal_data):
        ra_idx = cp.searchsorted(ra_bins, ra) - 1
        dec_idx = cp.searchsorted(dec_bins, dec) - 1
        
        if 0 <= ra_idx < len(ra_bins) - 1 and 0 <= dec_idx < len(dec_bins) - 1:
            sky_map[dec_idx, ra_idx] += signal
    
    return sky_map

def generate_lofar_image(filtered_fft_values, output_dir, date, time, lat, lon, duration_hours):
    logging.info('Starting LOFAR image generation.')

    with tqdm(total=1, desc='Generating LOFAR Image:') as pbar:
        os.makedirs(output_dir, exist_ok=True)
        try:
            logging.debug(f'Input parameters: date={date}, time={time}, lat={lat}, lon={lon}, duration_hours={duration_hours}')
            date_obj = datetime.strptime(date, '%Y%m%d')
            start_datetime = datetime.combine(date_obj.date(), datetime.strptime(time, '%H%M%S').time())
            start_time = Time(start_datetime)

            logging.debug(f'Creating time array with duration: {duration_hours} hours')
            times = start_time + np.linspace(0, duration_hours, len(filtered_fft_values)) * u.hour
            location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg)

            altitude = 90 * u.deg
            azimuth = 0 * u.deg

            logging.debug('Calculating coordinates in parallel.')
            with Pool() as pool:
                coords = pool.map(calculate_coordinates_wrapper, [(t, location, altitude, azimuth) for t in times])
            
            ras, decs = zip(*coords)
            ras = cp.array(ras, dtype=cp.float32)
            decs = cp.array(decs, dtype=cp.float32)

            ra_bins = cp.linspace(0, 360, 180)
            dec_bins = cp.linspace(-90, 90, 90)
            
            fft_values_normalized = (cp.abs(filtered_fft_values) - cp.min(cp.abs(filtered_fft_values))) / (cp.max(cp.abs(filtered_fft_values)) - cp.min(cp.abs(filtered_fft_values)))

            logging.debug('Mapping signal data to sky map.')
            sky_map = map_signal_to_sky(ras, decs, fft_values_normalized, ra_bins, dec_bins)

            non_zero = cp.nonzero(sky_map)
            ra_min, ra_max = ra_bins[non_zero[1].min()], ra_bins[non_zero[1].max()]
            dec_min, dec_max = dec_bins[non_zero[0].min()], dec_bins[non_zero[0].max()]

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
            norm = LogNorm(vmin=sky_map.min()+1, vmax=sky_map.max())
            plt.imshow(sky_map.get(), origin='lower', norm=norm, cmap='hot',
                    extent=[ra_min_zoomed, ra_max_zoomed, dec_min_zoomed, dec_max_zoomed], aspect='auto')
            plt.colorbar(label='Normalized Signal Intensity')
            plt.xlabel('Right Ascension (degrees)')
            plt.ylabel('Declination (degrees)')
            plt.title(f'LOFAR Fixed Pointing Observation\n'
                      f'Date: {date}, Time: {time}, Duration: {duration_hours:.2f} hours\n'
                      f'Latitude: {lat:.2f}°')

            plt.ylim(dec_min_zoomed, dec_max_zoomed)
            plt.plot(zoom_ra_center, zoom_dec_center, 'ro', markersize=10)

            output_file = os.path.join(output_dir, f'lofar_sky_map_zoom_{date}_{time}.png')
            plt.savefig(output_file)
            plt.close()

            logging.info(f"LOFAR image saved to: {output_file}")
            pbar.update(1)
        except Exception as e:
            logging.error(f"An error occurred while generating LOFAR image: {str(e)}")
            raise
        finally:
            del filtered_fft_values
            del sky_map
            gc.collect()
            logging.debug('Memory cleared.')