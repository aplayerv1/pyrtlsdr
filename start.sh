#!/bin/bash

cd /home/server/rtl/pyrtl

# Configuration variables
ffreq=1420.20e6
lfreq=1420.60e6
sfreq=1420.40e6
duration=1200
srf=2.4e6
tol=1.6e6
chunk=2048
ip=10.10.1.143
port=8885
workers=4

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
        python3 preprocess.py -i raw/$file -o raw/ --center_frequency $sfreq

        # # # Process the file using process5.py
        pf=$filename_w".fits.txt"

        # Process the file using process5.py
        echo "Starting Sound"
        # Assuming correct pattern for processed file

        python3 tosound.py "raw/$pf" "sound/${filename_w}.wav" --samplerate 48000 

        echo "Starting Processing"
        python3 process5.py -f raw/$pf -i raw/$filename_w".fits" -o images/$filename_w/ --start_time 0 --end_time $duration --tolerance $tol --chunk_size $chunk --fs $srf

        echo "Starting Heatmap"
        python3 heatmap.py -i raw/$filename_w".fits" -o images/$filename_w/ --fs $srf --chunk-size $chunk --num-workers $workers

	    rm -r /home/server/rtl/pyrtl/raw/*.txt

        cd /home/server/rtl/pyrtl

        # Sync processed images to NAS
        rsync -avh --update images/ /mnt/nas/tests/processed/

        # Sync raw data to NAS
        rsync -avh --update raw/ /mnt/nas/tests/capture/

        #Sybc sound data to NAS
        rsync -avh --update sound/ /mnt/nas/tests/sound/


    fi
done
