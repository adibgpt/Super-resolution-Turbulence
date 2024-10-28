
# Flow Field Prediction with PyTorch and PyTorch Lightning

This repository contains a PyTorch implementation of a flow field prediction model designed for the Turbulent Super-Resolution purposes. The model uses a modified EDSR architecture with Squeeze-and-Excitation (SE) blocks to enhance feature representation. It includes scripts for training, inference, and saving/loading models.

![Flow Field Prediction](https://github.com/adibgpt/Super-resolution-Turbulence/blob/48f9ead8d55294f7d7d36cdf2dcff28fc7de72dc/2D%20Snapshot.png)

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Dataset Description](#dataset-description)
- [Evaluation Metric](#evaluation-metric)
- [Requirements](#requirements)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Model Utilities](#model-utilities)
- [Acknowledgments](#acknowledgments)

## Problem Statement

The problem will be provided with a dataset containing low-resolution 2D flowfield images. The task is to develop state-of-the-art super-resolution models capable of reconstructing high-resolution versions of these flowfields. The challenge lies in preserving the fine details and structures in the flowfields while increasing the resolution, which is critical for maintaining the accuracy of simulations and analyses.

## Dataset

The dataset for this problem consists of pairs of low-resolution and high-resolution 2D slices of 3D direct numerical simulation data. Each flowfield slice is represented as 4 channels (density and 3 velocity components). The dataset is divided into training and validation sets, with a separate test set used for final evaluation.

## Dataset Description

### Files
- **train.csv** - Contains information about the training set.
- **val.csv** - Contains information about the validation set.
- **test.csv** - Contains information about the test set.
- **flowfields** - Directory containing flowfield data files.

### Columns in `train.csv`, `val.csv`, and `test.csv`
- **id** - Unique identifier for each flowfield file.
- **rho_filename** - Filename for the density component in `flowfields`.
- **ux_filename** - Filename for the x-velocity component in `flowfields`.
- **uy_filename** - Filename for the y-velocity component in `flowfields`.
- **uz_filename** - Filename for the z-velocity component in `flowfields`.

### Columns in `sample_submission.csv`
- **id** - Unique identifier for each flowfield file.
- **0, 1, ...** - Flattened array of predictions, representing the high-resolution output from the model.

## Evaluation Metric

The primary evaluation metric for this problem is Mean Squared Error (MSE) between the ground truth and the predicted high-resolution images. Additionally, further metrics such as Structural Similarity Index Measure (SSIM) of pixels and subgrid-scale stress evaluations will be provided to assess the model.

## Requirements

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Setup

1. **Dataset**: Place your dataset in a folder named `Dataset` in the root directory. The dataset should include:
    - `train.csv`, `val.csv`, `test.csv` (CSV files listing the dataset files)
    - Low-resolution and high-resolution flow field data in the `flowfields/LR` and `flowfields/HR` directories, respectively.
   
2. **Outputs Directory**: An `outputs` directory will be created automatically for storing logs, checkpoints, and predictions.

## Project Structure

```
.
├── data_loader.py         # Data loading and preprocessing
├── lightning_model.py     # Lightning module for training and validation
├── model.py               # Model architecture (EDSR with SE blocks)
├── model_utils.py         # Utility functions for saving/loading models
├── main.py                # Main script for training the model
├── inference.py           # Script for running inference on test data
├── requirements.txt       # Required packages
└── README.md              # Project documentation
```

## Training the Model

To train the model, simply run:
```bash
python main.py
```

This will:
- Train the model using the specified hyperparameters (defined in `main.py`).
- Log metrics to the `outputs/logs` directory, viewable via TensorBoard.
- Save the best model checkpoint based on validation loss in `outputs/checkpoints`.

## Running Inference

Once the model is trained, you can run inference on the test set:
```bash
python inference.py
```

This will:
- Load the model from the latest checkpoint in `outputs/checkpoints`.
- Run predictions on the test set.
- Save predictions to a timestamped CSV file in the `outputs` directory.

## Model Utilities

The `model_utils.py` file includes helper functions:
- **Saving and Loading Model Weights**:
  ```python
  from model_utils import save_model_weights, load_model_weights
  save_model_weights(model, output_path="outputs", suffix="_best")
  loaded_model = load_model_weights(model, weights_path="outputs/model_weights_best.pth")
  ```
- **Saving and Loading the Full Model**:
  ```python
  from model_utils import save_full_model, load_full_model
  save_full_model(model, output_path="outputs", suffix="_best")
  loaded_model = load_full_model("outputs/model_full_best.pth")
  ```

## Acknowledgments

This code was developed by Adib Bazgir and Rama Chandra Praneeth Madugula for the Turbulent Super-Resolution problem. Future versions may include models based on FNO, GANs, and other architectures for flow field prediction.
