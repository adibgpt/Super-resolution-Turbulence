import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from model import Model

class FlowFieldModel(LightningModule):
    """
    PyTorch Lightning Module for training the modified EDSR-based model.

    Args:
        learning_rate (float): Learning rate for the optimizer.
    """
    def __init__(self, learning_rate=0.001):
        super(FlowFieldModel, self).__init__()

        self.model = Model()
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()  # Mean Squared Error for regression tasks

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_loss = self.loss_fn(outputs, targets)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
