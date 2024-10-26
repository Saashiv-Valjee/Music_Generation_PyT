# src/data_processing.py

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import music21
import random
import logging
import pickle

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MaestroDataset(Dataset):
    def __init__(self, data_dir, sequence_length=100, transform=None):
        """
        Initialize the MaestroDataset.

        Args:
            data_dir (str): Path to the directory containing MIDI files.
            sequence_length (int): Length of each input sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Retrieve all MIDI file paths recursively from the data directory
        self.midi_files = glob.glob(os.path.join(data_dir, '**/*.mid*'), recursive=True)

        # Build the vocabulary of unique MIDI note numbers and rests
        self.note2idx, self.idx2note = self.build_vocab()

        # Preprocess the data to convert MIDI files into sequences of token indices
        self.data = self.preprocess_data()

    def build_vocab(self):
        """
        Build a vocabulary mapping from MIDI note numbers/rests to indices and vice versa.

        Returns:
            note2idx (dict): Dictionary mapping MIDI note numbers/rests to unique indices.
            idx2note (dict): Dictionary mapping indices back to MIDI note numbers/rests.
        """
        # Path to the vocabulary file
        vocab_path = os.path.join(self.data_dir, 'vocab.pkl')

        # Check if the vocabulary file exists
        if os.path.exists(vocab_path):
            # Load the vocabulary from the file
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            note2idx = vocab['note2idx']
            idx2note = vocab['idx2note']
            logger.info("Vocabulary loaded from disk.")
        else:
            logger.info("Building vocabulary without reading MIDI files...")
            # Define the possible MIDI note numbers (0-127)
            midi_notes = list(range(128))  # MIDI note numbers from 0 to 127
            # Assign a unique integer to 'REST', e.g., 128
            rest_token = 128
            midi_notes.append(rest_token)

            # Create mappings from notes to indices and indices to notes
            note2idx = {note: idx for idx, note in enumerate(midi_notes)}
            idx2note = {idx: note for note, idx in note2idx.items()}

            logger.info(f"Vocabulary size: {len(note2idx)}")

            # Save the vocabulary to disk
            with open(vocab_path, 'wb') as f:
                pickle.dump({'note2idx': note2idx, 'idx2note': idx2note, 'rest_token': rest_token}, f)
            logger.info("Vocabulary saved to disk.")

        return note2idx, idx2note

    def preprocess_data(self):
        """
        Preprocess the MIDI files to convert them into sequences of token indices.

        Returns:
            sequences (list): List of sequences, where each sequence is a list of token indices.
        """
        # Path to the processed data file
        processed_data_path = os.path.join(self.data_dir, 'processed_data.pt')

        # Check if the processed data file exists
        if os.path.exists(processed_data_path):
            # Load the processed data from disk
            sequences = torch.load(processed_data_path, weights_only=True)
            logger.info("Processed data loaded from disk.")
        else:
            logger.info("Preprocessing data...")
            sequences = []  # List to store all sequences

            # Get the rest token from the vocabulary
            rest_token = self.note2idx[128]  # 'REST' token is assigned 128

            # Iterate over all MIDI files to create sequences
            count = 0
            for midi_file in self.midi_files:
                count += 1
                logger.info(f'Processing file {count}/{len(self.midi_files)}: {midi_file}')
                sequence = []  # List to store the sequence for the current MIDI file
                try:
                    # Parse the MIDI file into a music21 stream object
                    stream = music21.converter.parse(midi_file)
                    # Iterate over all notes and rests in the stream
                    for element in stream.flatten().notesAndRests:
                        if element.isNote:
                            # Use the MIDI pitch number as the token
                            token = self.note2idx.get(element.pitch.midi)
                        elif element.isRest:
                            # Use 'REST' token
                            token = rest_token
                        else:
                            # Skip elements that are neither notes nor rests
                            continue

                        if token is not None:
                            # Append the token index to the sequence
                            sequence.append(token)
                except Exception as e:
                    # Log a warning if there's an error processing a MIDI file
                    logger.warning(f"Error processing {midi_file}: {e}")
                    continue

                # Only include sequences that are longer than the sequence length
                if len(sequence) > self.sequence_length:
                    sequences.append(sequence)
                else:
                    logger.debug(f"Sequence in {midi_file} is too short; skipping.")

            logger.info(f"Total sequences collected: {len(sequences)}")

            # Save the processed data to disk
            torch.save(sequences, processed_data_path)
            logger.info("Processed data saved to disk.")

        return sequences

    def __len__(self):
        """
        Return the total number of sequences in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Generate one sample of data.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            input_seq (Tensor): Tensor of input token indices.
            target_seq (Tensor): Tensor of target token indices.
        """
        sequence = self.data[idx]
        # Calculate the maximum possible starting index for slicing
        max_start_idx = len(sequence) - self.sequence_length - 1

        if max_start_idx <= 0:
            # If the sequence is too short, log a debug message and return None
            logger.debug(f"Sequence at index {idx} is too short; skipping.")
            return None  # Skip sequences that are too short

        # Randomly select a starting index for slicing the sequence
        start_idx = random.randint(0, max_start_idx)

        # Slice the sequence to get input and target sequences
        input_seq = sequence[start_idx:start_idx + self.sequence_length]
        target_seq = sequence[start_idx + 1:start_idx + self.sequence_length + 1]

        # Convert the sequences to PyTorch tensors
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)

        return input_seq, target_seq

def get_data_loader(data_dir, batch_size=32, sequence_length=100, shuffle=True):
    """
    Create a DataLoader for the MaestroDataset.

    Args:
        data_dir (str): Path to the directory containing MIDI files.
        batch_size (int): Number of samples per batch.
        sequence_length (int): Length of each input sequence.
        shuffle (bool): Whether to shuffle the dataset at every epoch.

    Returns:
        data_loader (DataLoader): DataLoader object for the dataset.
    """
    # Instantiate the dataset
    dataset = MaestroDataset(data_dir, sequence_length=sequence_length)

    def collate_fn(batch):
        """
        Custom collate function to handle batches with None entries.

        Args:
            batch (list): List of samples returned by __getitem__.

        Returns:
            inputs (Tensor): Batch of input sequences.
            targets (Tensor): Batch of target sequences.
        """
        # Remove None entries from the batch
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None  # Return None if the batch is empty

        # Unzip the batch into inputs and targets
        inputs, targets = zip(*batch)
        # Stack the inputs and targets into tensors
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        return inputs, targets

    # Create the DataLoader with the custom collate function
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Shuffle the data at every epoch if True
        collate_fn=collate_fn,  # Calling a method, not a process to be applied to an element
        drop_last=True  # Drop the last batch if it's incomplete
    )
    return data_loader
