#!/bin/bash

# Set the HOME environment variable if it's not already set
export HOME=/mnt/cephfs/home/amb/esainz  # Replace with your actual home directory path
# Activate the Conda environment
source /mnt/cephfs/home/amb/esainz/miniconda3/bin/activate pinn # Replace with your actual conda env
# Debugging: Print environment variables to verify
echo "HOME is set to: $HOME"
echo "PATH is set to: $PATH"
echo "Python path: $(which python)"

# Execute the Python script with train argument and additional training parameters
python main.py 