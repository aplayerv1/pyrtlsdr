#!/bin/bash

# Function to check if a command is available
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check and install missing packages (if needed)
install_packages() {
    if ! command_exists "$1"; then
        echo "Installing $1..."
        sudo apt-get update
        sudo apt-get install -y "$1"
    fi
}

# Check and install necessary packages
install_packages python3  # Replace with your Python version if needed
install_packages python3-pip  # Install pip for Python
install_packages libhackrf-dev  # Install libhackrf development files (if available)
install_packages libhackrf0  # Install libhackrf runtime library

# Update pip and install or upgrade pyhackrf2
echo "Updating pip..."
pip3 install --upgrade pip

echo "Installing or upgrading pyhackrf2..."
pip3 install --upgrade pyhackrf2

# Check if libhackrf.so.0 is accessible
echo "Checking libhackrf.so.0..."
if ! ldconfig -p | grep -q libhackrf.so.0; then
    echo "libhackrf.so.0 not found or not properly configured."
    echo "Ensure libhackrf is installed and check LD_LIBRARY_PATH."
    exit 1
fi

echo "Script completed successfully."
