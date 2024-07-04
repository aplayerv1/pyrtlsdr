#!/bin/bash

# Function to process a frequency range
process_frequency_range() {
    local ffreq=$1
    local lfreq=$2
    local sfreq=$3
    local fileappend=$4

    # Configuration variables
    duration=1800
    srf=20e6
    tol=1.6e6
    chunk=2048
    ip=10.10.1.17
    port=8885
    workers=24

    # Calculate dynamic bandwidth as half of the center frequency
    sfreq_decimal=$(printf "%.2f" "$sfreq") # Convert sfreq to decimal notation
    bandwidth=$(echo "scale=2; $sfreq_decimal / 2" | bc -l)
    
    echo "Center Frequency (sfreq): $sfreq"
    echo "Center Frequency in decimal (sfreq_decimal): $sfreq_decimal"
    echo "Calculated Bandwidth: $bandwidth"

    # Configuration variables for rsync
    nas_base_dir="/mnt/disk2T/rtl"
    nas_images_dir="$nas_base_dir/processed"
    nas_raw_dir="$nas_base_dir/capture"
    nas_sound_dir="$nas_base_dir/sound"

    # Capture data over a range of frequencies
    echo "Range $ffreq to $lfreq"
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    start_time_unix=$(date -d "$start_time" +%s)
    echo "Start Time: $start_time"
    python3 range.py $ip $port --start-freq $ffreq --end-freq $lfreq --duration $duration --sample-rate $srf

    # List all FITS files in the current directory
    for file in *.fits; do
        if [[ -f "$file" ]]; then
            echo "Processing file: $file"

            # Remove the .fits extension from the filename
            filename_w="${file%.fits}"

            # Append the file append string to the filename
            filename_w_appended="${filename_w}_${fileappend}"

            # Rename the file with the appended string
            mv "$file" "raw/${filename_w_appended}.fits"
            
            # Create directory to store processed images
            mkdir -p "images/${filename_w_appended}"

            # Preprocess the file using preprocess.py
            echo "Starting Preprocess"
            python3 preprocess.py -i "raw/${filename_w_appended}.fits" -o raw/ --center_frequency $sfreq --bandwidth $bandwidth -s $srf
            
            # Process the file using process5.py
            echo "Starting Sound"
            pf="${filename_w_appended}.fits.txt"  # Assuming correct pattern for processed file

            python3 tosound.py "raw/$pf" "sound/${filename_w_appended}.wav" --samplerate 48000

            echo "Starting Processing"
            python3 process5.py -f "raw/$pf" -i "raw/${filename_w_appended}.fits" -o "images/${filename_w_appended}/" --start_time 0 --end_time $duration --tolerance $tol --chunk_size $chunk --fs $srf --center-frequency $sfreq

            # Generate Heatmap
            echo "Starting Heatmap"
            python3 heatmap.py -i "raw/${filename_w_appended}.fits" -o "images/${filename_w_appended}/" --fs $srf --num-workers $workers --nperseg 2048

            # Clean up temporary files
            rm -r /home/server/rtl/pyrtl/raw/*.txt
            
            # Navigate back to the script's directory
            cd /home/server/rtl/pyrtl

            # Sync processed images to NAS
            rsync -avh --update images/ "$nas_images_dir/"

            # Sync raw data to NAS
            rsync -avh --update raw/ "$nas_raw_dir/"

            # Sync sound data to NAS
            rsync -avh --update sound/ "$nas_sound_dir/"

            # Clean up raw and images directories
            find /home/server/rtl/pyrtl/raw/ -type f -mmin +1 -exec rm -r {} \;
            find /home/server/rtl/pyrtl/images/ -type f -mmin +1 -exec rm -r {} \;
            find /home/server/rtl/pyrtl/sound/ -type f -mmin +1 -exec rm -r {} \;
        fi
    done

    end_time=$(date '+%Y-%m-%d %H:%M:%S')
    end_time_unix=$(date -d "$end_time" +%s)
    duration_seconds=$((end_time_unix - start_time_unix))

    # Convert duration to hours, minutes, and seconds
    hours=$((duration_seconds / 3600))
    minutes=$(( (duration_seconds % 3600) / 60 ))
    seconds=$((duration_seconds % 60))

    echo "End Time: $end_time"
    echo "Duration: ${hours}h ${minutes}m ${seconds}s"
}

cd /home/server/rtl/pyrtl

# Process each frequency range
process_frequency_range 1420.20e6 1420.60e6 1420.40e6 "1420MHz"
process_frequency_range 407.8e6 408.2e6 408e6 "408MHz"
process_frequency_range 150.8e6 151.2e6 151e6 "151MHz"
process_frequency_range 30.0e6 80.0e6 50.0e6 "50MHz"
