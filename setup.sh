#!/bin/bash
set -e  

echo "Setting up Conda environment......"
conda env create -f environment.yml

eval "$(conda shell.bash hook)"
conda activate fsaa_pfms
pip install --no-deps git+https://github.com/BrianPulfer/fsaa.git

echo "Setup completed! Use 'conda activate fsaa_pfms' to get started."