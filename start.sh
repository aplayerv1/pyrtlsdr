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
