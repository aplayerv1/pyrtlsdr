# Radio Astronomy with SDR

This project directory contains scripts and files for capturing, processing, and analyzing radio frequency data.

## Hardware
### SDR Hardware
    Raspberry Pi: Provides a compact and versatile platform to run pyrtlsdr on, especially useful for portable or embedded SDR applications. With low power consumption, it’s a great choice for field operations or remote monitoring setups.

    HackRF One: A software-defined radio capable of transmission and reception from 1 MHz to 6 GHz. When combined with pyrtlsdr, the HackRF One can perform a wide range of signal processing tasks across this frequency range, making it a flexible option for hobbyists and professionals alike.

    Bias Tee (10 MHz - 6 GHz): A bias tee is useful for powering external devices, like LNA (Low Noise Amplifiers), directly from the radio through the antenna port. When using HackRF One with pyrtlsdr, a bias tee can be enabled in setups that require additional amplification or filtering directly at the antenna input.


## Files and Scripts

## Installation

To set up the project environment, use the provided `install.sh` script:

1. Open a terminal in the project directory.
2. Make the script executable: `chmod +x install.sh`
3. Run the installation script: `./install.sh`

This script will install all necessary dependencies and set up the required environment for the project.

## Usage

 - `start.py`: Initiates data capture and processing for specific frequency ranges.

### Python Scripts:

- `start.py`: Initiates data capture and processing for specific frequency ranges.
- `process5.py`: Processes data files for analysis.
- `aim.py`: Handles AI-related tasks for signal processing and analysis.
- `advanced_signal_processing.py`: Contains advanced signal processing functions.
- `preprocess.py`: Preprocesses raw data for further analysis.
- `heatmap.py`: Generates heatmaps from captured data.

### Other Files:

- `README.md`: This documentation file.
- `config.ini`: Configuration file containing IP, port, and other settings.

## Directories

- `images/`: Stores output images or visualizations.
- `raw/`: Directory for storing raw data files.
- `aggregated/`: Directory for storing aggregated data files.

### Processing Data:

- Use `process5.py` to prepare and analyze data files.
- `advanced_signal_processing.py` provides additional signal processing capabilities.

### Visualization:

- `heatmap.py` generates heatmaps based on processed data stored in `images/`.

## Configuration in config.ini

All key settings for the project are stored in the `config.ini` file. Below is an example configuration:

        ```ini
        [settings]
        ip = 10.10.1.17
        port = 8886
        duration = 10
        SRF = 20e6
        tol = 1600000
        chunk = 2048
        workers = 4
        lat = 41.157940
        lon = 8.464160
        base_directory = /home/server/rtl/pyrtl
        nas_images_dir = /mnt/nas/tests/processed
        nas_raw_dir = /mnt/nas/tests/capture

        Key Parameters:
        ip: IP address for server connections.
        port: Port number for server connections.
        duration: Duration of data capture sessions (in seconds).
        SRF: Sampling rate frequency.
        tol: Tolerance value for frequency adjustments.
        chunk: Data chunk size for processing.
        workers: Number of worker threads or processes.
        lat, lon: Latitude and longitude coordinates of the observation site.
        base_directory: Base directory for local data storage.
        nas_images_dir: Network-attached storage directory for processed images.
        nas_raw_dir: Network-attached storage directory for raw data capture.


## Changing Settings in config.ini
        To modify any of the settings in the `config.ini` file, follow these steps:
        1. Open the `config.ini` file in a text editor.
        2. Locate the setting you want to change.
        3. Modify the value of the setting to the desired value.
        4. Save the changes to the `config.ini` file.
        5. Restart the script or application that uses the modified settings.
        Note: Make sure to restart the script or application after making any changes to the `config.ini` file to ensure that the new settings are applied.

## Frequency Adjustment

The `frequency_ranges.csv` includes a frequency adjustment feature. To use this feature, follow these steps:

        1. Open the `frequency_ranges.csv` file in a text editor.
        2. Locate the frequency range you want to adjust.
        3. Modify the frequency value in the `frequency` column.
        4. Save the changes to the `frequency_ranges.csv` file.
        5. Restart the script or application that uses the modified frequency range.

        (start, stop, center, name, low_cutoff, high_cutoff)

        (0, 10000000, 5000000, "Filename_10MHz", 4000000, 6000000)
        
## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or submit a pull request.