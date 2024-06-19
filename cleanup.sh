#!/bin/bash

# List of files to keep
declare -a keep_files=(
    "aggragate.py"
    "heatmap.py"
    "process5.py"
    "range.py"
    "server.py"
    "ai.sh"
    "start.sh"
    "aggragate.sh"
)

# Directory where Python scripts are located
scripts_dir="/home/server/rtl/pyrtl"

# Iterate through all Python files in the directory
for file in "$scripts_dir"/*.py; do
    filename=$(basename "$file")
    # Check if the file should be kept
    if [[ ! " ${keep_files[@]} " =~ " ${filename} " ]]; then
        echo "Removing $filename"
        rm -f "$file"
    fi
done
