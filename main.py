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

    # Iterate over the data loader and print batch shapes
    for batch in data_loader:
        if batch is None:
            continue
        inputs, targets = batch
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")

        # Optionally, print the first input and target sequences
        idx2note = data_loader.dataset.idx2note

        # Since idx2note now maps indices to MIDI note numbers, we need to convert them to note names
        input_indices = inputs[0].tolist()
        target_indices = targets[0].tolist()

        # Function to convert MIDI numbers to note names
        def midi_to_note_name(midi_number):
            if midi_number == 128:  # REST token
                return 'REST'
            else:
                return music21.note.Note(midi_number).nameWithOctave

        input_notes = [midi_to_note_name(idx2note[idx]) for idx in input_indices]
        target_notes = [midi_to_note_name(idx2note[idx]) for idx in target_indices]

        print(f"First input sequence (indices): {input_indices}")
        print(f"First input sequence (notes): {input_notes}")
        print(f"First target sequence (indices): {target_indices}")
        print(f"First target sequence (notes): {target_notes}")
        break  # Test only one batch

if __name__ == '__main__':
    main()
