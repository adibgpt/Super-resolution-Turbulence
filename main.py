import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data_loader import FlowFieldDataset
from lightning_model import FlowFieldModel

# Set hyperparameters and paths
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
INPUT_PATH = "./data/"  # Assuming data is in a folder named 'data'
OUTPUT_PATH = "./outputs/"
CHECKPOINT_DIR = os.path.join(OUTPUT_PATH, "checkpoints/")
LOG_DIR = os.path.join(OUTPUT_PATH, "logs/")

# Ensure necessary directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    # Initialize datasets and dataloaders
    train_dataset = FlowFieldDataset(input_path=INPUT_PATH, mode="train")
    val_dataset = FlowFieldDataset(input_path=INPUT_PATH, mode="val")
    test_dataset = FlowFieldDataset(input_path=INPUT_PATH, mode="test")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Initialize model and logger
    model = FlowFieldModel(learning_rate=LEARNING_RATE)
    logger = TensorBoardLogger(LOG_DIR, name="tensorboard")

    # Configure checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
