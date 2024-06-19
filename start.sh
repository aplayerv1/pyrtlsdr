<<<<<<< HEAD
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
=======
val=1420.25e6
val2=1420.35e6
val3=1430.30e6
while true
do
    # Commands or scripts to execute indefinitely
    # For example, you can print a message
    echo "This loop will continue indefinitely"
    python3 range.py IP 8886 --start-freq $val3 --single-freq  --duration 900
    python3 range.py IP 8886 --start-freq $val --end-freq $val2 --duration 900

for file in *.fits; do
    echo $file
    # Check if the current file is a regular file
    if [ -f "$file" ]; then
        # Perform actions with the file name
        echo "Processing file: $file"

	    filename_w="${file%.*}"

	    mkdir -p images/$filename_w

        # Process the file using preprocess.py
        python3 process.py -i $file -o images/$filename_w/
        mv $file raw/
        # Assign the file name to the variable var
    fi
done
    sleep 10
done
>>>>>>> 0858e9ca9b448c30a3baea59d8a38216c90a9f50
