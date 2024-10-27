import torch
from datetime import datetime

def save_model_weights(model, output_path="outputs", suffix=""):
    """
    Save only the model weights.

    Args:
        model (torch.nn.Module): The model to save.
        output_path (str): Directory where the model weights will be saved.
        suffix (str): Suffix for the filename, if any (e.g., "_best" or accuracy score).
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    weights_filename = f"model_weights{suffix}_{timestamp}.pth"
    torch.save(model.state_dict(), f"{output_path}/{weights_filename}")
    print(f"Model weights saved as {weights_filename} in {output_path}.")

def save_full_model(model, output_path="outputs", suffix=""):
    """
    Save the full model including weights and architecture.

    Args:
        model (torch.nn.Module): The model to save.
        output_path (str): Directory where the model will be saved.
        suffix (str): Suffix for the filename, if any.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_filename = f"model_full{suffix}_{timestamp}.pth"
    torch.save(model, f"{output_path}/{model_filename}")
    print(f"Full model saved as {model_filename} in {output_path}.")

def load_model_weights(model, weights_path):
    """
    Load model weights from a specified file.

    Args:
        model (torch.nn.Module): Model architecture to load weights into.
        weights_path (str): Path to the model weights file.
    """
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set to evaluation mode
    print(f"Model weights loaded from {weights_path}.")
    return model

def load_full_model(model_path):
    """
    Load the full model (architecture + weights) from a specified file.

    Args:
        model_path (str): Path to the full model file.
    Returns:
        torch.nn.Module: Loaded model.
    """
    model = torch.load(model_path)
    model.eval()  # Set to evaluation mode
    print(f"Full model loaded from {model_path}.")
    return model
