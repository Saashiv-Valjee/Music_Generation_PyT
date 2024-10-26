# music generation with transformer using pytorch and maestro Dataset
# 1: setup Environment and import libraries

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import music21
import os
import glob
import random
import logging

from src.data_processing import get_data_loader

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting up project space
if not os.path.exists('data'): os.makedirs('data')
if not os.path.exists('models'): os.makedirs('models')
if not os.path.exists('src'): os.makedirs('src')
if not os.path.exists('outputs'): os.makedirs('outputs')

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    data_dir = 'data/maestro-v3.0.0/'
    batch_size = 16
    sequence_length = 100

    # Create the data loader
    data_loader = get_data_loader(data_dir, batch_size=batch_size, sequence_length=sequence_length)

    # Check if data loader is created successfully
    if data_loader is None:
        print("Data loader could not be created.")
        return

    # Iterate over the data loader and print batch shapes
    for batch in data_loader:
        if batch is None:
            continue
        inputs, targets = batch
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")

        # Optionally, print the first input and target sequences
        idx2note = data_loader.dataset.idx2note
        input_notes = [idx2note[idx.item()] for idx in inputs[0]]
        target_notes = [idx2note[idx.item()] for idx in targets[0]]

        print(f"First input sequence (indices): {inputs[0]}")
        print(f"First input sequence (notes): {input_notes}")
        print(f"First target sequence (indices): {targets[0]}")
        print(f"First target sequence (notes): {target_notes}")
        break  # Test only one batch

if __name__ == '__main__':
    main()