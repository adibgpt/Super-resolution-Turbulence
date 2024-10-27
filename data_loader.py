import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

class FlowFieldDataset(Dataset):
    """
    Custom Dataset for loading flow field data with low-resolution (LR) and high-resolution (HR) fields.
    
    Args:
        input_path (str): Path to the dataset directory.
        mode (str): Mode of the dataset, one of ["train", "val", "test"].
    """
    def __init__(self, input_path, mode):
        assert mode in ["train", "val", "test"], "Mode should be 'train', 'val', or 'test'."
        self.mode = mode
        self.csv_file = pd.read_csv(os.path.join(input_path, f"{mode}.csv"))
        
        # Define paths for low-resolution and high-resolution data
        self.LR_path = os.path.join(input_path, "flowfields", "LR", mode)
        self.HR_path = os.path.join(input_path, "flowfields", "HR", mode)

        # Normalization values
        self.mean = np.array([0.24, 28.0, 28.0, 28.0])
        self.std = np.array([0.068, 48.0, 48.0, 48.0])

    def transform(self, x):
        """
        Apply transformations to the data, including normalization.
        
        Args:
            x (np.array): Input array of flow field data.
        Returns:
            torch.Tensor: Transformed tensor.
        """
        return Compose([ToTensor(), Normalize(self.mean, self.std, inplace=True)])(x)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: Transformed low-resolution input and high-resolution target tensors.
        """
        # Read low-resolution inputs
        rho_i = np.fromfile(os.path.join(self.LR_path, self.csv_file["rho_filename"][idx]), dtype="<f4").reshape(16, 16)
        ux_i = np.fromfile(os.path.join(self.LR_path, self.csv_file["ux_filename"][idx]), dtype="<f4").reshape(16, 16)
        uy_i = np.fromfile(os.path.join(self.LR_path, self.csv_file["uy_filename"][idx]), dtype="<f4").reshape(16, 16)
        uz_i = np.fromfile(os.path.join(self.LR_path, self.csv_file["uz_filename"][idx]), dtype="<f4").reshape(16, 16)
        X = np.stack([rho_i, ux_i, uy_i, uz_i], axis=2)

        if self.mode == "test":
            # For test mode, only return the input data and ID
            id = self.csv_file["id"][idx]
            return id, self.transform(X)

        # Read high-resolution targets
        rho_o = np.fromfile(os.path.join(self.HR_path, self.csv_file["rho_filename"][idx]), dtype="<f4").reshape(128, 128)
        ux_o = np.fromfile(os.path.join(self.HR_path, self.csv_file["ux_filename"][idx]), dtype="<f4").reshape(128, 128)
        uy_o = np.fromfile(os.path.join(self.HR_path, self.csv_file["uy_filename"][idx]), dtype="<f4").reshape(128, 128)
        uz_o = np.fromfile(os.path.join(self.HR_path, self.csv_file["uz_filename"][idx]), dtype="<f4").reshape(128, 128)
        Y = np.stack([rho_o, ux_o, uy_o, uz_o], axis=2)
        
        return self.transform(X), self.transform(Y)
