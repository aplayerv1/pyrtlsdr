#!/bin/bash

cd /home/server/rtl/pyrtl

# Configuration variables
ffreq=1420.20e6
lfreq=1420.60e6
sfreq=1420.40e6
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
        mkdir -p images/$filename_w

        # # Preprocess the file using preprocess.py
        echo "Starting Preprocess"
        python3 preprocess.py -i raw/$file -o raw/

        # # # Process the file using process5.py
        echo "Starting Processing"
        pf=$filename_w".fits.txt"
        python3 process5.py -f raw/$pf -i raw/$filename_w".fits" -o images/$filename_w/ --start_time 0 --end_time $duration --tolerance $tol --chunk_size $chunk --fs $srf
        echo "Starting Heatmap"
        python3 heatmap.py -i raw/$filename_w".fits" -o images/$filename_w/ --fs $srf --chunk-size $chunk --num-workers 16
	rm -r /home/server/rtl/pyrtl/raw/*.txt
        cd /home/server/rtl/pyrtl
        rsync -avh --update images/ /mnt/nas/tests/processed/
        rsync -avh --update raw/ /mnt/nas/tests/capture/
        rm -r /home/server/rtl/pyrtl/raw/*
        rm -r /home/server/rtl/pyrtl/images/*
    fi
done
