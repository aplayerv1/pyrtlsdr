import numpy as np
from astropy.io import fits

# Function to read binary data from file
def read_binary_file(filename):
    with open(filename, 'rb') as f:
        binary_data = f.read()
    return binary_data

# Function to convert binary data to FITS
def binary_to_fits(binary_data, output_filename):

    # Step 2: Convert binary data to NumPy array (example assuming 1D array of integers)
    # Adjust this step based on the format of your binary data
    numpy_array = np.frombuffer(binary_data, dtype=np.int32)

    # Step 3: Create a FITS HDU with the NumPy array as the data
    hdu = fits.PrimaryHDU(numpy_array)

    # Step 4: Optionally, add header information to the FITS HDU
    hdu.header['OBSERVER'] = 'John Doe'
    hdu.header['DATE'] = '2024-02-21'

    # Step 5: Write the FITS HDU to a file
    hdu.writeto(output_filename, overwrite=True)

# Example usage
if __name__ == "__main__":
    # Input binary filename
    binary_filename = "data_20240223_231656.fits"
    
    # Output FITS filename
    output_filename = "data_20240223_231656_2.fits"

    
with fits.open('data_20240223_231656.fits', mode='update') as hdul:
    # Access the header of the primary HDU
    header = hdul[0].header

    # Add or modify header keywords as needed
    header['NEWKEY'] = 'New value'
    header['COMMENT'] = 'This is a comment'
    header['EXPTIME'] = 10.0  # Example of adding a new keyword with a value
    header['DATE'] = '2024-01-01'
    # Save changes back to the FITS file
    hdul.flush()

    # Read binary data from file
    binary_data = read_binary_file(binary_filename)

    # Convert binary data to FITS
    binary_to_fits(binary_data, output_filename)
