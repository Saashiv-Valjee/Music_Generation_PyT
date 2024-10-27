# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

# Import your data loader and model
from src.data_processing import get_data_loader
from models.model import MusicTransformer

# Import hyperparameters from config.py
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device (CPU or GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    # Prepare data loader
    data_loader = get_data_loader(config.data_dir, batch_size=config.batch_size, sequence_length=config.sequence_length)
    dataset = data_loader.dataset
    vocab_size = len(dataset.note2idx)
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

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir='logs')

    model.train()  # Set model to training mode

    global_step = 0  # Counter for logging

    for epoch in range(1, config.num_epochs + 1):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}/{config.num_epochs}")

        for batch_idx, batch in progress_bar:
            if batch is None:
                continue
            inputs, targets = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Reshape outputs and targets for loss computation
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # Update parameters
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            # Log metrics
            if global_step % config.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                logger.info(f'Epoch [{epoch}/{config.num_epochs}], Step [{batch_idx}/{len(data_loader)}], Loss: {avg_loss:.4f}')
                writer.add_scalar('Loss/train', avg_loss, global_step)

        # Save checkpoint after each epoch
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)
        checkpoint_path = os.path.join(config.checkpoint_dir, f'music_transformer_epoch{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f'Model checkpoint saved at {checkpoint_path}')

    writer.close()

def main():
    train()

if __name__ == '__main__':
    main()
