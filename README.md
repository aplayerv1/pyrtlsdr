# My Project

This is a brief description of my project.

## Installation

- Clone the repository: `git clone https://github.com/aplayerv1/pyrtlsdr.git`
- Install dependencies: `pip3 install -r requirements`
- RTL USB in USB PORT 
## Usage

Here's how you can use this project:

1. Run `python3 server.py` to start the application. 
2. Run `python3 process2.py` to connect to server and display basic metric signal strength
all data of the usb is in the `http://localhost:8000/api`

        3. Open your web browser and navigate to `http://localhost:8888`.
        arguments for process2.py
            -a or --server-address IP-ADDRESS, default: localhost
            -p or --server-port  PORTNUMBER, default: 8888
            -o or --lnb-offset if you have a LNB plugged in default: 9750e6 which is in hz
            -f or --frequency, default:100e6 `example for 1.2 GHZ = 1200.00e6`
            -g or --gain, default auto


4. To process data from a binary file captured from rtl_sdr run `python3 process.py` don't forget that arguments are required.

the raw file name must be NAME_YYYYMMDD_HHMMMM.bin

arguments for process.py
    -i or --input, `example python3 process.py -i pathoffile`
    -o or --output, `example python3 process.py -i pathoffolder -o ./` for local directory
    -s or --sampling_rate, default: 2.4e6 `which is 2.4mhz`
    -c or --center_frequency, default: 1420.30e6 `which is 1.4203 ghz`

`example python3 process.py -i file.bin -o ./ -f 1420.30e6 -s 2.4e6`

output of files are:
    1. heatmap
    2. frequency spectrum
    3. signal strength
    4. preprocessed heatmap


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

