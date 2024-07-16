#!/bin/bash

# Function to process a frequency range
process_frequency_range() {
    local ffreq=$1
    local lfreq=$2
    local sfreq=$3
    local fileappend=$4

    # Configuration variables
    duration=100
    duration_hours=$(echo "scale=4; $duration / 3600" | bc)
    srf=20e6
    tol=1.6e6
    chunk=2048
    ip=10.10.1.17
    port=8885
    workers=24
    lat=41.604730
    lon=-8.464160

    # Calculate dynamic bandwidth as half of the center frequency
    sfreq_decimal=$(printf "%.2f" "$srf") # Convert sfreq to decimal notation
    bandwidth=$(echo "scale=2; $sfreq_decimal / 2" | bc -l)
    
    echo "Center Frequency (sfreq): $sfreq"
    echo "Center Frequency in decimal (sfreq_decimal): $sfreq_decimal"
    echo "Calculated Bandwidth: $bandwidth"

    # Configuration variables for rsync
    nas_base_dir="/mnt/nas/tests"
    nas_images_dir="$nas_base_dir/processed"
    nas_raw_dir="$nas_base_dir/capture"
    nas_sound_dir="$nas_base_dir/sound"

    # Capture data over a range of frequencies
    echo "Range $ffreq to $lfreq"
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    start_time_unix=$(date -d "$start_time" +%s)
    echo "Start Time: $start_time"
    python3 range.py $ip $port --start-freq $ffreq --end-freq $lfreq --duration "$duration" --sample-rate $srf

    # List all FITS files in the current directory
    for file in *.fits; do
        latest_fits=$(ls -t *.fits | head -n1)

        if [[ -f "$latest_fits" ]]; then
            echo "Processing file: $latest_fits"

            # Remove the .fits extension from the filename
            filename_w="${latest_fits%.fits}"

            # Append the file append string to the filename
            filename_w_appended="${filename_w}_${fileappend}"

            # Move the file to the raw directory
            mv "$latest_fits" "raw/${filename_w_appended}.fits"
            
            # Create directory to store processed images
            mkdir -p "images/${filename_w_appended}"

            echo "Starting Processing"
            python3 process5.py -i "raw/${filename_w_appended}.fits" -o "images/${filename_w_appended}/" \
                --tolerance $tol --chunk_size $chunk --fs $srf --center-frequency $sfreq \
                --duration $duration_hours --latitude $lat --longitude $lon
            # Generate Heatmap
            echo "Starting Heatmap"
            python3 heatmap.py -i "raw/${filename_w_appended}.fits" -o "images/${filename_w_appended}/" --fs $srf --num-workers $workers --nperseg 2048

            find /home/server/rtl/pyrtl/images/ -type d -empty -delete
            find /home/server/rtl/pyrtl/raw/ -type d -empty -delete
             
            # Navigate back to the script's directory
            cd /home/server/rtl/pyrtl

            # # Sync processed images to NAS
            # rsync -avh --update images/ "$nas_images_dir/"

            # # Sync raw data to NAS
            # rsync -avh --update raw/ "$nas_raw_dir/"


            # # Clean up raw and images directories
            # find /home/server/rtl/pyrtl/raw/ -type f -mmin +1 -exec rm -r {} \;
            # find /home/server/rtl/pyrtl/images/ -type f -mmin +1 -exec rm -r {} \;

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
process_frequency_range 1420.20e6 1420.60e6 1420.40e6 "1420MHz_HI"
# process_frequency_range 407.8e6 408.2e6 408e6 "408MHz_Haslam"
# process_frequency_range 150.8e6 151.2e6 151e6 "151MHz_6C"
process_frequency_range 30.0e6 80.0e6 50.0e6 "50MHz_8C"

# # Additional astronomical frequencies up to 6 GHz
# process_frequency_range 322.8e6 323.2e6 323e6 "323MHz_Deuterium"
# process_frequency_range 1610.6e6 1611.0e6 1610.8e6 "1611MHz_OH"
# process_frequency_range 1665.2e6 1665.6e6 1665.4e6 "1665MHz_OH"
# process_frequency_range 1667.2e6 1667.6e6 1667.4e6 "1667MHz_OH"
# process_frequency_range 1720.2e6 1720.6e6 1720.4e6 "1720MHz_OH"
# process_frequency_range 2290.8e6 2291.2e6 2291e6 "2291MHz_H2CO"
# process_frequency_range 2670.8e6 2671.2e6 2671e6 "2671MHz_RRL"
# process_frequency_range 3260.8e6 3261.2e6 3261e6 "3261MHz_CH"
# process_frequency_range 3335.8e6 3336.2e6 3336e6 "3336MHz_CH"
# process_frequency_range 3349.0e6 3349.4e6 3349.2e6 "3349MHz_CH"
# process_frequency_range 4829.4e6 4830.0e6 4829.7e6 "4830MHz_H2CO"
# process_frequency_range 5289.6e6 5290.0e6 5289.8e6 "5290MHz_OH"
# process_frequency_range 5885.0e6 5885.4e6 5885.2e6 "5885MHz_CH3OH"
# process_frequency_range 400.0e6 800.0e6 600.0e6 "600MHz_Pulsar"
# process_frequency_range 1400.0e6 1400.4e6 1400.2e6 "1400MHz_Pulsar"
# process_frequency_range 327.0e6 327.4e6 327.2e6 "327MHz_Pulsar"
# process_frequency_range 74.0e6 74.4e6 74.2e6 "74MHz_Pulsar"
# process_frequency_range 408.5e6 408.9e6 408.7e6 "408.7MHz_Pulsar"
# process_frequency_range 800.0e6 900.0e6 850.0e6 "850MHz_Pulsar"
# process_frequency_range 1500.0e6 1500.4e6 1500.2e6 "1500MHz_Pulsar"
# process_frequency_range 1427.0e6 1427.4e6 1427.2e6 "1427MHz_HI"
# process_frequency_range 550.0e6 600.0e6 575.0e6 "575MHz_HCN"
# process_frequency_range 5500.0e6 5600.0e6 5550.0e6 "5550MHz_H2O"

# # Additional astronomical frequencies below 100 MHz
# process_frequency_range 40.0e6 41.0e6 40.5e6 "40.5MHz_Galactic_Synchrotron"
# process_frequency_range 60.0e6 65.0e6 62.5e6 "62.5MHz_Low_Frequency_Interference"
# process_frequency_range 80.0e6 85.0e6 82.5e6 "82.5MHz_Extragalactic_Radio_Lobes"
# process_frequency_range 20.0e6 30.0e6 25.0e6 "25MHz_Solar_Radio_Bursts"
# process_frequency_range 45.0e6 50.0e6 47.5e6 "47.5MHz_Interstellar_Absorption"
# process_frequency_range 95.0e6 100.0e6 97.5e6 "97.5MHz_Solar_Coronal_Loops"
