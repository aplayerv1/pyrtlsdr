#!/bin/bash

cd /home/server/rtl/pyrtl

# Configuration variables for 408 MHz center frequency
ffreq=407.8e6
lfreq=408.2e6
sfreq=408e6
duration=1200
srf=2.4e6
tol=1.6e6
chunk=1024
ip=10.10.1.143
port=8885

# Capture data over a range of frequencies
echo "Range $ffreq to $lfreq"
python3 range.py $ip $port --start-freq $ffreq --end-freq $lfreq --duration $duration --sample-rate $srf

# List all FITS files in the current directory
for file in *.fits; do
    if [[ -f "$file" ]]; then
        echo "Processing file: $file"
        mv $file raw/
        filename_w="${file%.*}"
        # Append 408MHz designation to the filename
        filename_w_408="${filename_w}_408MHz"

        mkdir -p images/$filename_w_408

        # Preprocess the file using preprocess.py
        echo "Starting Preprocess"
        python3 preprocess.py -i raw/$file -o raw/

        # Process the file using process5.py
        echo "Starting Processing"
        pf=$filename_w".fits.txt"
        python3 process5.py -f raw/$pf -i raw/$filename_w".fits" -o images/$filename_w_408/ --start_time 0 --end_time $duration --tolerance $tol --chunk_size $chunk --fs $srf

        # Generate Heatmap
        echo "Starting Heatmap"
        python3 heatmap.py -i raw/$filename_w".fits" -o images/$filename_w_408/ --fs $srf --chunk-size $chunk --num-workers 16

        # Clean up temporary files
        rm -r /home/server/rtl/pyrtl/raw/*.txt
        cd /home/server/rtl/pyrtl

        # Sync processed images to NAS
        rsync -avh --update images/ /mnt/nas/tests/processed/

        # Sync raw data to NAS
        rsync -avh --update raw/ /mnt/nas/tests/capture/

        # Clean up raw and images directories
        rm -r /home/server/rtl/pyrtl/raw/*
        rm -r /home/server/rtl/pyrtl/images/*
    fi
done
