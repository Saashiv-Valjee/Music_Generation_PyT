# config.py

import os

# Data parameters
data_dir = 'data/maestro-v3.0.0/'
sequence_length = 128  # Adjust based on your computational resources
batch_size = 32

'''# complex_Model hyperparameters
embedding_dim = 128
num_heads = 8
num_layers = 6
dim_feedforward = 512
dropout = 0.1
max_len = sequence_length  # Should match sequence_length'''

# simple_Model hyperparameters
embedding_dim = 128
num_heads = 4
num_layers = 2
dim_feedforward = 512
dropout = 0.1
max_len = 5000  # Should be at least as long as the longest sequence

# Training parameters
learning_rate = 1e-4
num_epochs = 500
grad_clip = 1.0
log_interval = 100
checkpoint_dir = 'checkpoints'

# Evaluation parameters
checkpoint_path = os.path.join(checkpoint_dir, f'music_transformer_epoch{num_epochs}.pth')  # Adjust if necessary
output_midi_path = 'outputs/generated_music.mid'
max_generation_length = 256  # Maximum length of the generated sequence
temperature = 1.0  # Sampling temperature
