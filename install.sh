#!/bin/bash

# Install Miniconda (if not already installed)
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Create and activate the tf_gpu environment
conda create -n tf_gpu python=3.8 -y
conda activate tf_gpu

# Install required packages
conda install -y numpy matplotlib scipy astropy
conda install -y -c conda-forge cupy
conda install -y psutil configparser tensorflow-gpu argparse
conda install -y numba
conda install -y -c conda-forge pywt
conda install -y scikit-learn

# Install additional packages that might not be available through conda
pip install concurrent-futures
pip install EMD-signal
pip install PyEMD
pip install pyhackrf2

# Install packages for the new imports
conda install -y astropy

# Create necessary directories
mkdir -p raw images

echo "Installation complete. Activate the environment with: conda activate tf_gpu"
