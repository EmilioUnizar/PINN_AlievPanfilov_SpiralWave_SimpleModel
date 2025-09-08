#!/bin/bash

# Set the HOME environment variable if it's not already set
export HOME=/mnt/cephfs/home/amb/esainz 

# Activate the Conda environment
source /mnt/cephfs/home/amb/esainz/miniconda3/bin/activate pinn # Replace with your actual conda env

# Execute the Python script with train argument and additional training parameters EJECUTABLE A COPIAR (EJEMPLO)
wandb agent '801707-university-of-zaragoza/AP- Marta/xthewd1v' 
