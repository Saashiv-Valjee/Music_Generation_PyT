# main.py

import logging
from src.data_processing import get_data_loader
import music21

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed output
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

if __name__ == '__main__':
    main()
