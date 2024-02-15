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


4. To process data from a binary file captured from rtlsd.py `python3 rtlsd.py` don't forget that arguments are required.

to capture data using rtlsd.py arguments are 
                      
                      -f or --frequency 
                      -t or --time in seconds
                      --output-dir default=./
                      --sample-rate default=2.4e6
                      --gain default=auto doesnt work for me so a number is required

file will be saved as raw_data_YYYYMMDD_HHMMSS.bin

python process.py the raw file name must be NAME_YYYYMMDD_HHMMMM.bin
                
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

5. aiml.py is ai machine learning althought I am still new at that, It runs but I don't think its working correctly any help would be appreciated
                   
                arguments for aiml.py
                        -a or --server-address default=localhost
                        -p or --server-port default=8888
                        -o or --lnb-offset default=9750e6
                        -f or --frequency default=100e6
                        -g or --gain default=auto but auto doesnt work for me
                        -s or --sampling-frequency default=2.4e6
                        -c or --cutoff-frequency default=1000 cutoff frequency for noise removal
                        -n or --notch-bandwidth default= 100 <--- not implemented
                
                `example python3 process.py -i file.bin -o ./ -f 1420.30e6 -s 2.4e6`

6. range.py is the same as rtlsd.py but instead of just one frenquency it has a low and a high for capture within range

                   arguments for range.py
                       --start-freq
                       --end-freq
                       --sample-rate
                       --duration in seconds
                       --output-dir
           the file created will be binary_raw_YYYYMMDD_HHMMSS.bin

7. analysis.py isn't much but a debug tool for the raw files all it does is print peaks of the signal

                   arguments for analysis.py
                   -i or --input path to binary file must be NAME_YYYYMMDD_HHMMSS.bin
                   -t or --threshold default=0.5


                    
## The server with the conjunction of process2.py
Is running at https://dos.mvia.ca the api is at https://dos.mvia.ca/api

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

