# src/evaluate.py

import os
import torch
import torch.nn as nn
import numpy as np
import random
import music21
import logging

# Import your data loader and model
from src.data_processing import MaestroDataset
from models.model import MusicTransformer

# Import hyperparameters from config.py
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device (CPU or GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset to access vocabulary mappings
dataset = MaestroDataset(config.data_dir, sequence_length=config.sequence_length)
note2idx = dataset.note2idx
idx2note = dataset.idx2note
vocab_size = len(note2idx)
logger.info(f'Vocabulary size: {vocab_size}')

# Initialize the model
model = MusicTransformer(
    vocab_size=vocab_size,
    embedding_dim=config.embedding_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    dim_feedforward=config.dim_feedforward,
    dropout=config.dropout,
    max_len=config.max_len
).to(DEVICE)

# Load the trained model checkpoint
if os.path.exists(config.checkpoint_path):
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=DEVICE))
    logger.info(f'Loaded model checkpoint from {config.checkpoint_path}')
else:
    logger.error(f'Checkpoint not found at {config.checkpoint_path}')
    exit(1)

# Set model to evaluation mode
model.eval()

def generate_music(model, seed_sequence, max_length, temperature=1.0):
    """
    Generate a music sequence using the trained model.

    Args:
        model: Trained MusicTransformer model.
        seed_sequence (list): List of token indices to start the generation.
        max_length (int): Maximum length of the generated sequence.
        temperature (float): Sampling temperature for controlling randomness.

    Returns:
        List[int]: Generated sequence of token indices.
    """
    generated = seed_sequence.copy()
    input_sequence = seed_sequence.copy()

    for _ in range(max_length):
        # Prepare input tensor
        input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(DEVICE)  # Shape: (1, seq_length)

        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)  # Output shape: (1, seq_length, vocab_size)
            next_token_logits = output[0, -1, :]  # Get logits for the last time step

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Convert logits to probabilities
            probabilities = torch.softmax(next_token_logits, dim=-1).cpu().numpy()

            # Sample the next token from the probability distribution
            next_token = np.random.choice(len(probabilities), p=probabilities)

        # Append the predicted token to the generated sequence
        generated.append(next_token)

        # Update the input sequence (keep the sequence length constant)
        input_sequence.append(next_token)
        input_sequence = input_sequence[-config.sequence_length:]  # Keep the last 'sequence_length' tokens

    return generated

def main():
    # Select a random seed sequence from the dataset
    seed_idx = random.randint(0, len(dataset) - 1)
    seed_input, _ = dataset[seed_idx]  # Get a random sequence from the dataset
    seed_sequence = seed_input.tolist()  # Convert tensor to list

    logger.info(f'Generating music with seed sequence of length {len(seed_sequence)}')

    # Generate music
    generated_sequence = generate_music(
        model,
        seed_sequence=seed_sequence,
        max_length=config.max_generation_length,
        temperature=config.temperature  # Adjust temperature as desired
    )

    logger.info(f'Generated sequence length: {len(generated_sequence)}')

    # Separate seed sequence and generated notes
    generated_notes_sequence = generated_sequence[len(seed_sequence):]

    # Convert token indices to tokens (note strings)
    seed_tokens = [idx2note[token_idx] for token_idx in seed_sequence]
    generated_tokens = [idx2note[token_idx] for token_idx in generated_notes_sequence]

    # Print seed sequence and generated notes
    print("\nSeed Sequence:")
    seed_notes_display = []
    for token in seed_tokens:
        if token.startswith('NOTE'):
            parts = token.split('_')
            midi_number = int(parts[1])
            duration_name = parts[-1]
            note_name = music21.note.Note(midi_number).nameWithOctave
            seed_notes_display.append(f'{note_name}_{duration_name}')
        elif token.startswith('REST'):
            duration_name = token.split('_')[-1]
            seed_notes_display.append(f'REST_{duration_name}')
        else:
            seed_notes_display.append('UNKNOWN')
    print(seed_notes_display)

    print("\nGenerated Notes:")
    generated_notes_display = []
    for token in generated_tokens:
        if token.startswith('NOTE'):
            parts = token.split('_')
            midi_number = int(parts[1])
            duration_name = parts[-1]
            note_name = music21.note.Note(midi_number).nameWithOctave
            generated_notes_display.append(f'{note_name}_{duration_name}')
        elif token.startswith('REST'):
            duration_name = token.split('_')[-1]
            generated_notes_display.append(f'REST_{duration_name}')
        else:
            generated_notes_display.append('UNKNOWN')
    print(generated_notes_display)

    # Convert tokens to music21 notes
    notes = []
    # Combine seed and generated tokens
    all_tokens = seed_tokens + generated_tokens

    # Reverse mapping of durations
    duration_mapping = {v: k for k, v in dataset.DURATIONS.items()}

    for token in all_tokens:
        if token.startswith('NOTE'):
            parts = token.split('_')
            midi_number = int(parts[1])
            duration_name = parts[-1]
            duration = duration_mapping.get(duration_name, 1.0)  # Default to 1.0 if not found
            note = music21.note.Note(midi_number)
            note.duration.quarterLength = duration
            notes.append(note)
        elif token.startswith('REST'):
            duration_name = token.split('_')[-1]
            duration = duration_mapping.get(duration_name, 1.0)  # Default to 1.0 if not found
            rest = music21.note.Rest()
            rest.duration.quarterLength = duration
            notes.append(rest)
        else:
            # Handle unknown tokens if necessary
            continue

    # Create a music21 stream
    midi_stream = music21.stream.Stream(notes)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(config.output_midi_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the MIDI file
    midi_stream.write('midi', fp=config.output_midi_path)
    logger.info(f'Generated MIDI file saved to {config.output_midi_path}')

if __name__ == '__main__':
    main()
