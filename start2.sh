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
workers=4

# Capture data over a range of frequencies
echo "Range $ffreq to $lfreq"
python3 range.py $ip $port --start-freq $ffreq --end-freq $lfreq --duration $duration --sample-rate $srf

# List all FITS files in the current directory
for file in *.fits; do
    if [[ -f "$file" ]]; then
        echo "Processing file: $file"

        # Remove the .fits extension from the filename
        filename_w="${file%.fits}"

        # Append _408MHz to the filename
        filename_w_408="${filename_w}_408MHz"

        # Rename the file with _408MHz appended
        mv "$file" "raw/${filename_w_408}.fits"
        
        # Confirm the file is in raw/ directory
        ls raw/

        # Create directory to store processed images
        mkdir -p "images/${filename_w_408}"

        # Preprocess the file using preprocess.py
        echo "Starting Preprocess"
        python3 preprocess.py -i "raw/${filename_w_408}.fits" -o raw/ --center_frequency $sfreq
        
        # Check if the file is accessible
        ls raw/

        # Process the file using process5.py
        echo "Starting Sound"
        pf="${filename_w_408}.fits.txt"  # Assuming correct pattern for processed file


        python tosound.py "raw/$pf" "sound/${filename_w_408}.wav" --samplerate 48000 

        
        echo "Starting Processing"
        python3 process5.py -f "raw/$pf" -i "raw/${filename_w_408}.fits" -o "images/${filename_w_408}/" --start_time 0 --end_time $duration --tolerance $tol --chunk_size $chunk --fs $srf

        # Generate Heatmap
        echo "Starting Heatmap"
        python3 heatmap.py -i "raw/${filename_w_408}.fits" -o "images/${filename_w_408}/" --fs $srf --chunk-size $chunk --num-workers $workers

        # Clean up temporary files
        rm -r /home/server/rtl/pyrtl/raw/*.txt
        
        # Navigate back to the script's directory
        cd /home/server/rtl/pyrtl

        # Sync processed images to NAS
        rsync -avh --update images/ /mnt/nas/tests/processed/

        # Sync raw data to NAS
        rsync -avh --update raw/ /mnt/nas/tests/capture/

        #Sybc sound data to NAS
        rsync -avh --update sound/ /mnt/nas/tests/sound/


        # Clean up raw and images directories
        rm -r /home/server/rtl/pyrtl/raw/*
        rm -r /home/server/rtl/pyrtl/images/*
        rm -r /home/server/rtl/pyrtl/sound/*

        # Navigate back to the original directory for the next iteration
        cd -
    fi
done
