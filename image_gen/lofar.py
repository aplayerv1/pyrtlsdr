import logging

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u
import os
import logging
from datetime import datetime

def generate_lofar_image(signal_data, output_dir, date, time, lat, lon, duration_hours):

    with tqdm(total=1, desc='Generating LOFAR Image:') as pbar:
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Set up logging
    
            # Initialize parameters
            date_obj = datetime.strptime(date, '%Y%m%d')
            start_datetime = datetime.combine(date_obj.date(), datetime.strptime(time, '%H%M%S').time())
            start_time = Time(start_datetime)
    
            # Create a time array spanning the observation duration
            times = start_time + np.linspace(0, duration_hours, len(signal_data)) * u.hour
            location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg)

            altitude = 90 * u.deg
            azimuth = 0 * u.deg

            ras = []
            decs = []

            for t in times:
                altaz = AltAz(obstime=t, location=location)
                skycoord = SkyCoord(alt=altitude, az=azimuth, frame=altaz)
                icrs_coord = skycoord.icrs
                ras.append(icrs_coord.ra.deg)
                decs.append(icrs_coord.dec.deg)

            ras = np.array(ras)
            decs = np.array(decs)
    
            # Create an empty sky map
            ra_bins = np.linspace(0, 360, 360)
            dec_bins = np.linspace(-90, 90, 180)
    
            sky_map = np.zeros((len(dec_bins), len(ra_bins)))
    
            # Normalize signal data
            signal_data_normalized = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data))
    
            # Map the signal data to the sky map
            for i, (ra, dec, signal) in enumerate(zip(ras, decs, signal_data_normalized)):
                ra_idx = np.digitize(ra, ra_bins) - 1
                dec_idx = np.digitize(dec, dec_bins) - 1
        
                if 0 <= ra_idx < len(ra_bins) and 0 <= dec_idx < len(dec_bins):
                    sky_map[dec_idx, ra_idx] += signal
    
            # Find the non-zero region
            non_zero = np.nonzero(sky_map)
            ra_min, ra_max = ra_bins[non_zero[1].min()], ra_bins[non_zero[1].max()]
            dec_min, dec_max = dec_bins[non_zero[0].min()], dec_bins[non_zero[0].max()]

            # Add a small buffer to the declination range
            ra_buffer = 2.0  # degrees, adjust as needed
            dec_buffer = 2.0  # degrees, adjust as needed
            ra_min = max(0, ra_min - ra_buffer)
            ra_max = min(360, ra_max + ra_buffer)
            dec_min = max(-90, dec_min - dec_buffer)
            dec_max = min(90, dec_max + dec_buffer)

            # Zoom into a specific region (example: center on RA=180, Dec=0 with a zoom factor of 1 degree)
            zoom_ra_center = 180
            zoom_dec_center = 0
            zoom_factor = 1.0  # degrees, adjust as needed
    
            ra_min_zoomed = zoom_ra_center - zoom_factor / 2
            ra_max_zoomed = zoom_ra_center + zoom_factor / 2
            dec_min_zoomed = zoom_dec_center - zoom_factor / 2
            dec_max_zoomed = zoom_dec_center + zoom_factor / 2
    
            # Ensure the zoomed-in region stays within the overall limits
            ra_min_zoomed = max(ra_min, ra_min_zoomed)
            ra_max_zoomed = min(ra_max, ra_max_zoomed)
            dec_min_zoomed = max(dec_min, dec_min_zoomed)
            dec_max_zoomed = min(dec_max, dec_max_zoomed)

            # Plot the sky map with zoom
            plt.figure(figsize=(12, 6))
            norm = LogNorm(vmin=sky_map.min()+1, vmax=sky_map.max())
            plt.imshow(sky_map, origin='lower', norm=LogNorm(vmin=sky_map.min()+1, vmax=sky_map.max()), cmap='hot', 
                    extent=[ra_min_zoomed, ra_max_zoomed, dec_min_zoomed, dec_max_zoomed], aspect='auto')
            plt.colorbar(label='Normalized Signal Intensity')
            plt.xlabel('Right Ascension (degrees)')
            plt.ylabel('Declination (degrees)')
            plt.title(f'LOFAR Fixed Pointing Observation\n'
                      f'Date: {date}, Time: {time}, Duration: {duration_hours:.2f} hours\n'
                      f'Latitude: {lat:.2f}°')

            # Ensure the y-axis limits are set explicitly
            plt.ylim(dec_min_zoomed, dec_max_zoomed)

            # Example: Mark specific point with a red dot (RA=180, Dec=0)
            plt.plot(zoom_ra_center, zoom_dec_center, 'ro', markersize=10)

            output_file = os.path.join(output_dir, f'lofar_sky_map_zoom_{date}_{time}.png')
            plt.savefig(output_file)
            plt.close()

            logging.info(f"Zoomed-in RA range: {ra_min_zoomed:.2f} to {ra_max_zoomed:.2f}")
            logging.info(f"Zoomed-in Dec range: {dec_min_zoomed:.2f} to {dec_max_zoomed:.2f}")

            pbar.update(1)
        except Exception as e:
            logging.error(f"An error occurred while generating LOFAR image: {str(e)}")
            raise