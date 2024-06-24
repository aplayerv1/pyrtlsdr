#!/bin/bash

srf=2.4e6
chunk=1024
workers=24
# Navigate to the working directory
cd /home/server/rtl/pyrtl

# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Create a timestamped directory
output_dir="/mnt/disk2T/rtl/combined"
mkdir -p "$output_dir"

# # Aggregate data
echo "Starting Aggregate"
python3 aggragate.py -i /mnt/nas/tests/capture -o "$output_dir"

# Check if the aggregation was successful
if [ $? -ne 0 ]; then
    echo "Aggregation failed, exiting."
    exit 1
fi

# Array to store directories containing .fits files
declare -a dirs_with_fits

# Find all subdirectories containing .fits files
while IFS= read -r -d '' new_output_dir; do
    if [[ -n $(find "$new_output_dir" -maxdepth 1 -name "*.fits" -print -quit) ]]; then
        dirs_with_fits+=("$new_output_dir")
    fi
done < <(find "$output_dir" -mindepth 1 -type d -print0)

# Loop through each directory containing .fits files
for dir_with_fits in "${dirs_with_fits[@]}"; do
    # Preprocess aggregated files in the current output directory
    echo "Starting Process 2 for directory: $dir_with_fits"
    find "$dir_with_fits" -type f -name '*.fits' | while read -r file; do
        if [[ -f "$file" ]]; then
            base_name=$(basename "$file" .fits)
            python3 preprocess.py -i "$file" -o "$dir_with_fits/"
            if [ $? -ne 0 ]; then
                echo "Preprocessing failed for $file, exiting."
                exit 1
            fi
        fi
    done
done

echo "Starting Processing of Combined"
find "$output_dir" -type f -name 'aggregate_????????_??????.fits' | while read -r fits_file; do
    txt_file="${fits_file%.fits}.fits.txt"
    echo "Processing $fits_file"
    echo "Checking if corresponding .txt file exists at: $txt_file"
    if [[ -f "$txt_file" ]]; then
        base_name=$(basename "$fits_file" .fits)
        echo "Base name: $base_name"
        filename_w="$base_name"
        pf="$filename_w.fits.txt"
        input_file="$fits_file"
        output_file="images/$filename_w/"
        mkdir -p "$output_file"

        # Read the duration from the FITS header
        duration=$(python3 -c "
from astropy.io import fits
import sys

fits_file = sys.argv[1]
with fits.open(fits_file) as hdul:
    print(int(hdul[0].header['DURATION']))

" "$input_file")

        if [ $? -ne 0 ]; then
            echo "Failed to extract duration for $input_file, skipping."
            continue
        fi

        python3 process5.py -f "$txt_file" -i "$input_file" -o "$output_file" --start_time 0 --end_time "$duration" --tolerance 1.6e6
        if [ $? -ne 0 ]; then
            echo "Processing failed for $fits_file, exiting."
            exit 1
        fi

        echo "Starting Heatmap Process for $input_file"
        python3 heatmap.py -i "$input_file" -o "$output_file" --fs "$srf"  --num-workers $workers
        if [ $? -ne 0 ]; then
            echo "Heatmap processing failed for $input_file, exiting."
            exit 1
        fi

        # Rsync the processed files to the target directory
        rsync -av "images/" /mnt/nas/tests/processed/
        if [ $? -ne 0 ]; then
            echo "Rsync failed for $output_file, exiting."
            exit 1
        fi


    else
        echo "Corresponding .txt file for $fits_file not found, skipping."
    fi
done

# Clean up combined directory
echo "Cleaning up combined directory"
rm -r "$output_dir"/*.*
if [ $? -ne 0 ]; then
    echo "Cleanup failed, exiting."
    exit 1
fi

echo "All tasks completed successfully."
