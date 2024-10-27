import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_loader import FlowFieldDataset
from lightning_model import FlowFieldModel

# Paths and filenames
INPUT_PATH = "./data/"  # Path to dataset
OUTPUT_PATH = "./outputs/"
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoints/best_model.ckpt")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_model(checkpoint_path):
    """
    Load the model from the specified checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
    Returns:
        FlowFieldModel: Loaded model in evaluation mode.
    """
    model = FlowFieldModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def run_inference():
    # Load the model
    model = load_model(CHECKPOINT_PATH)

    # Initialize test dataset and dataloader
    test_dataset = FlowFieldDataset(input_path=INPUT_PATH, mode="test")
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Run inference
    predictions = {}
    ids = []
    progress_bar = tqdm(test_dataloader, desc="Running Inference")

    for idx, batch in enumerate(progress_bar):
        id, inputs = batch
        outputs = model(inputs)
        outputs = outputs.permute(0, 2, 3, 1)  # Reorder dimensions
        predictions[idx] = outputs.cpu().detach().numpy().flatten(order="C").astype(np.float32)
        ids.append(id.cpu().detach().numpy()[0])

    # Convert predictions to DataFrame
    df = pd.DataFrame.from_dict(predictions, orient="index")
    df["id"] = ids

    # Move 'id' column to the front
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols].reset_index(drop=True)

    # Save predictions to a timestamped CSV file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(OUTPUT_PATH, f"predictions_{timestamp}.csv")
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

if __name__ == "__main__":
    run_inference()
