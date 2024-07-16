import logging
import os
from matplotlib import pyplot as plt
import numpy as np

def create_position_velocity_diagram(signal_data, sampling_rate, output_dir, date, time):
    velocity = np.fft.fftfreq(len(signal_data), d=1/sampling_rate)
    position = np.linspace(-10, 10, len(signal_data))
    intensity = np.abs(signal_data)  # Assuming signal_data is already processed appropriately

    # Create a 2D histogram for position-velocity diagram
    hist, xedges, yedges = np.histogram2d(position, velocity, bins=100, weights=intensity)

    plt.figure(figsize=(10, 6))
    plt.imshow(hist.T, extent=[np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)],
               origin='lower', cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Intensity')
    plt.xlabel('Position (arcsec)')
    plt.ylabel('Velocity (km/s)')
    plt.title('Position-Velocity Diagram')
    plt.grid(True)
    plt.tight_layout()

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'position_velocity_diagram_{date}_{time}.png'))
    
    plt.show()