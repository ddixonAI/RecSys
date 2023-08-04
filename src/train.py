# Import Modules
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import model
import config


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for (x_user, x_bus), y in loader:
        x_user = x_user.to(device, dtype=torch.float32)
        x_bus = x_bus.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x_user, x_bus)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)

    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for (x_user, x_bus), y in loader:
            x_user = x_user.to(device, dtype=torch.float32)
            x_bus = x_bus.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x_user, x_bus)
            loss = loss_fn(y_pred, y)

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)

    return epoch_loss


def run():
    # Training code here

    return


if __name__ == "__main__":
    run()
