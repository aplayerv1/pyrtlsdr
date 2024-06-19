from astropy.io import fits

# Open the FITS file
with fits.open('data_20240220_203408.fits') as hdul:
    # Get the header
    header = hdul[0].header

    # Print all the keywords in the header
    print(header)