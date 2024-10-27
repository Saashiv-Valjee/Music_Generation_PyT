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
    DURATIONS = {
        0.25: 'SIXTEENTH',
        0.5: 'EIGHTH',
        1.0: 'QUARTER',
        2.0: 'HALF',
        4.0: 'WHOLE'
    }

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

        # Preprocess the data to convert MIDI files into sequences of token indices
        self.data, self.unique_tokens = self.preprocess_data()

        # Build the vocabulary of unique tokens
        self.note2idx, self.idx2note = self.build_vocab()

        # Convert sequences of tokens to sequences of indices
        self.data = self.tokens_to_indices(self.data)

    def preprocess_data(self):
        """
        Preprocess the MIDI files to convert them into sequences of tokens.

        Returns:
            sequences (list): List of sequences, where each sequence is a list of tokens.
            unique_tokens (set): Set of unique tokens found in the data.
        """
        # Path to the processed data file
        processed_data_path = os.path.join(self.data_dir, 'processed_data.pkl')

        # Check if the processed data file exists
        if os.path.exists(processed_data_path):
            # Load the processed data from disk
            with open(processed_data_path, 'rb') as f:
                data_dict = pickle.load(f)
            sequences = data_dict['sequences']
            unique_tokens = data_dict['unique_tokens']
            logger.info("Processed data loaded from disk.")
        else:
            logger.info("Preprocessing data...")
            sequences = []  # List to store all sequences
            unique_tokens = set()

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
                        if element.quarterLength == 0.0:
                            continue  # Skip zero-duration elements

                        # Map the duration to the nearest category
                        duration = min(self.DURATIONS.keys(), key=lambda x: abs(x - element.quarterLength))
                        duration_name = self.DURATIONS[duration]

                        if element.isNote:
                            token = f'NOTE_{element.pitch.midi}_DURATION_{duration_name}'
                        elif element.isRest:
                            token = f'REST_DURATION_{duration_name}'
                        else:
                            continue

                        sequence.append(token)
                        unique_tokens.add(token)
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
            with open(processed_data_path, 'wb') as f:
                pickle.dump({'sequences': sequences, 'unique_tokens': unique_tokens}, f)
            logger.info("Processed data saved to disk.")

        return sequences, unique_tokens

    def build_vocab(self):
        """
        Build a vocabulary mapping from tokens to indices and vice versa.

        Returns:
            note2idx (dict): Dictionary mapping tokens to unique indices.
            idx2note (dict): Dictionary mapping indices back to tokens.
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
            logger.info("Building vocabulary...")

            unique_tokens = self.unique_tokens

            # Create mappings from tokens to indices and vice versa
            note2idx = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
            idx2note = {idx: token for token, idx in note2idx.items()}

            logger.info(f"Vocabulary size: {len(note2idx)}")

            # Save the vocabulary to disk
            with open(vocab_path, 'wb') as f:
                pickle.dump({'note2idx': note2idx, 'idx2note': idx2note}, f)
            logger.info("Vocabulary saved to disk.")

        return note2idx, idx2note

    def tokens_to_indices(self, sequences):
        """
        Convert sequences of tokens to sequences of indices.

        Args:
            sequences (list): List of sequences, where each sequence is a list of tokens.

        Returns:
            sequences_idx (list): List of sequences, where each sequence is a list of indices.
        """
        sequences_idx = []
        for sequence in sequences:
            sequence_idx = [self.note2idx[token] for token in sequence if token in self.note2idx]
            sequences_idx.append(sequence_idx)
        return sequences_idx

    def __len__(self):
        """
        Return the total number of sequences in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Calculate the maximum possible starting index for slicing
        max_start_idx = len(sequence) - self.sequence_length - 1

        if max_start_idx <= 0:
            logger.debug(f"Sequence at index {idx} is too short; skipping.")
            return None  # Skip sequences that are too short

        start_idx = random.randint(0, max_start_idx)

        # Slice the sequence to get input and target sequences
        input_seq = sequence[start_idx:start_idx + self.sequence_length]
        target_seq = sequence[start_idx + 1:start_idx + self.sequence_length + 1]

        # Convert to tensors
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
