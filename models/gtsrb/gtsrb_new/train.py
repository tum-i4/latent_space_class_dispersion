import json
import sys
import time
from typing import Tuple, Any, Literal, Optional, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader


def batched(images: torch.tensor, labels: torch.tensor, batch_size: int):
    for i in range(0, len(labels), batch_size):
        end = min(len(labels), i + batch_size)

        yield images[i:end], labels[i:end]


def calc_validation(net: nn.Module, loader: DataLoader, criterion, device: str):
    net.eval()
    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        total = 0

        for img, label in loader:
            with torch.autocast(device_type=device):
                logit = net(img)
                loss = criterion(logit, label)

            correct += (torch.max(logit, dim=1)[1] == label).sum().item()
            total_loss += loss.item()
            total += len(label)

    return total_loss / total, correct / total


def train(net: nn.Module, data: Dict[str, torch.tensor], out_file: Path, num_epochs: int = 50, **kwargs):
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    dataset_train = TensorDataset(data['img_train'], data['lbl_train'])
    loader_train = DataLoader(dataset_train, batch_size=1024, shuffle=True)

    dataset_valid = TensorDataset(data['img_valid'], data['lbl_valid'])
    loader_valid = DataLoader(dataset_valid, 1024, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    history = []
    val_history = []

    for epoch in range(0, num_epochs):
        net.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (img, label) in enumerate(loader_train):
            optimizer.zero_grad()

            with torch.autocast(device_type=device):
                logit = net(img)
                logit = torch.log_softmax(logit, dim=1)
                loss = criterion(logit, label)

            correct += (torch.max(logit, dim=1)[1] == label).sum().item()
            total_loss += loss.item()
            total += len(label)

            loss.backward()
            optimizer.step()

            # if batch_idx >= 2000:
            #     break

        val_loss, val_acc = calc_validation(net, loader_valid, criterion, device)
        val_history.append((val_loss, val_acc))

        history.append((total_loss / total, correct / total))
        print(
            f'Epoch: [{epoch:04}]',
            f'Loss: {total_loss/ total:15.10f}',
            f'Accuracy: {correct/total:07.3%} ({correct: 6}/{total})',
            ' | ',
            f'V_Loss: {val_loss:13.10f}',
            f'V_Accuracy: {val_acc:07.3%}',
        )
    end = time.time()

    torch.save(net.state_dict(), out_file)

    out_dir = out_file.parent
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, out_dir / 'checkpoint.tar')

    net_script = torch.jit.script(net)
    net_script.save(out_dir / "model_script.pth")

    with (out_dir / "training.json").open("w") as fp:
        json.dump({
            "duration": end - start,
            "final_train_acc": history[-1][1],
            "final_valid_acc": val_history[-1][1],
            "history": history,
            "val_history": val_history,
        }, fp, indent=2)
